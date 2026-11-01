// Preparation only: pack the solved page graph into exact Metal stage scopes.
// A stage barrier orders an encoder prefix, not one resource. Keep that prefix
// only when every affected consumer already depends on all of its frontier.
struct MetalScopeSearch {
    NSArray *history;
    NSMutableData *visited;
    NSMutableData *positions;
    NSUInteger generation;
};

static BOOL MetalScopeCovers(struct MetalScopeSearch *search,NSUInteger node,NSIndexSet *frontier) {
    if(!frontier.count)return YES;
    MetalWork *work=search->history[node];
    NSMutableIndexSet *needed=[[frontier mutableCopy] autorelease];
    [needed removeIndex:node];
    if(work->dependencies.count)[needed removeIndexes:work->dependencies];
    if(!needed.count)return YES; // The common chain never walks its history.
    const NSUInteger *positions=search->positions.bytes;
    for(NSUInteger at=needed.firstIndex;at!=NSNotFound;at=[needed indexGreaterThanIndex:at])
        if(positions[at]&&positions[at]>=positions[node])return NO;
    if(++search->generation==0) {
        memset(search->visited.mutableBytes,0,search->visited.length);
        search->generation=1;
    }
    NSUInteger *visited=search->visited.mutableBytes;
    NSMutableIndexSet *pending=[[work->dependencies mutableCopy] autorelease];
    while(pending.count) {
        const NSUInteger at=pending.lastIndex;[pending removeIndex:at];
        if(visited[at]==search->generation)continue;
        visited[at]=search->generation;
        if([frontier containsIndex:at]) {
            [needed removeIndex:at];
            if(!needed.count)return YES;
            // Frontier members are an antichain: no other required member
            // can hide among this matched member's predecessors.
            continue;
        }
        MetalWork *prior=search->history[at];
        if(prior->dependencies.count)[pending addIndexes:prior->dependencies];
    }
    return NO;
}

static BOOL MetalScopePrecedes(struct MetalScopeSearch *search,NSUInteger before,NSUInteger after) {
    return MetalScopeCovers(search,after,[NSIndexSet indexSetWithIndex:before]);
}

static void MetalScopeAdd(struct MetalScopeSearch *search,NSMutableIndexSet *frontier,NSUInteger node) {
    NSMutableIndexSet *removed=[NSMutableIndexSet indexSet];
    for(NSUInteger at=frontier.firstIndex;at!=NSNotFound;at=[frontier indexGreaterThanIndex:at]) {
        if(MetalScopePrecedes(search,node,at))return;
        if(MetalScopePrecedes(search,at,node))[removed addIndex:at];
    }
    [frontier removeIndexes:removed];[frontier addIndex:node];
}

static NSMutableIndexSet *MetalScopeStage(NSMutableDictionary *stages,MTLStages stage) {
    NSNumber *key=@(stage);
    NSMutableIndexSet *frontier=stages[key];
    if(!frontier)stages[key]=frontier=[NSMutableIndexSet indexSet];
    return frontier;
}

static void MetalScopeMerge(struct MetalScopeSearch *search,NSMutableIndexSet *into,NSIndexSet *from) {
    if(!from.count)return;
    for(NSUInteger at=from.firstIndex;at!=NSNotFound;at=[from indexGreaterThanIndex:at])
        MetalScopeAdd(search,into,at);
}

static NSMutableIndexSet *MetalScopeFrontier(struct MetalScopeSearch *search,NSDictionary *stages,MTLStages mask) {
    NSMutableIndexSet *frontier=[NSMutableIndexSet indexSet];
    while(mask) {
        const MTLStages stage=mask&(~mask+1);mask&=~stage;
        MetalScopeMerge(search,frontier,stages[@(stage)]);
    }
    return frontier;
}

static BOOL MetalScopeIncludes(struct MetalScopeSearch *search,NSIndexSet *frontier,NSUInteger producer) {
    for(NSUInteger at=frontier.firstIndex;at!=NSNotFound;at=[frontier indexGreaterThanIndex:at])
        if(MetalScopePrecedes(search,producer,at))return YES;
    return NO;
}

// Cohorts contain independent GPU commands with identical input/output edges.
// Exported values and GPU-to-host publications are singleton cohorts. Host and
// external entries remain individual MetalWork objects in this source order.
static void MetalScheduleScopes(struct MetalSchedule *root,NSArray *sequence,NSUInteger first) {
    struct MetalScopeSearch search={root->history,
        [NSMutableData dataWithLength:root->history.count*sizeof(NSUInteger)],
        [NSMutableData dataWithLength:root->history.count*sizeof(NSUInteger)],0};
    NSUInteger *positions=search.positions.mutableBytes,position=1;
    for(id entry in sequence) {
        if([entry isKindOfClass:[NSArray class]]) {
            for(MetalWork *work in entry)positions[work->ordinal]=position;
        } else positions[((MetalWork *)entry)->ordinal]=position;
        position++;
    }
    NSMutableDictionary *prefix=[NSMutableDictionary dictionary],*imposed=[NSMutableDictionary dictionary];
    NSIndexSet *hosts=nil;
    NSUInteger scope=0;
    for(id entry in sequence) {
        if(![entry isKindOfClass:[NSArray class]]) {
            scope=0;hosts=nil;[prefix removeAllObjects];[imposed removeAllObjects];continue;
        }
        NSArray *cohort=entry;MetalWork *work=cohort[0];
        BOOL cut=!scope||![hosts isEqual:work->hostDependencies];
        MTLStages after=0;
        if(!cut) {
            NSIndexSet *required=MetalScopeFrontier(&search,imposed,work->stages);
            cut=!MetalScopeCovers(&search,work->ordinal,required);
            // A future fence at this producer covers every earlier command of
            // its stage. Check its consumers now; a cut at the later consumer
            // could no longer remove an unrelated command from that prefix.
            NSIndexSet *prior=MetalScopeFrontier(&search,prefix,work->stages);
            for(MetalWork *member in cohort) {
                if(member->exported&&!MetalScopeCovers(&search,member->ordinal,prior))cut=YES;
                NSIndexSet *uses=member->successors;
                for(NSUInteger next=uses.firstIndex;!cut&&next!=NSNotFound;next=[uses indexGreaterThanIndex:next])
                    cut=!MetalScopeCovers(&search,first+next,prior);
            }
            for(NSUInteger p=work->gpuDependencies.firstIndex;!cut&&p!=NSNotFound;p=[work->gpuDependencies indexGreaterThanIndex:p]) {
                MetalWork *producer=root->history[p];
                if(producer->scope==scope&&!MetalScopeIncludes(&search,required,p))after|=producer->stages;
            }
            if(after)cut=!MetalScopeCovers(&search,work->ordinal,MetalScopeFrontier(&search,prefix,after));
        }
        if(cut) {
            scope=++root->scopeCount;after=0;
            [prefix removeAllObjects];[imposed removeAllObjects];
            hosts=work->hostDependencies;
        }
        for(MetalWork *member in cohort) {
            member->scope=scope;member->barrierAfter=member->barrierBefore=0;
            member->needsFence=member->exported;
        }
        work->barrierAfter=after;work->barrierBefore=after?work->stages:0;
        NSMutableIndexSet *required=MetalScopeFrontier(&search,prefix,after);
        for(NSUInteger p=work->gpuDependencies.firstIndex;p!=NSNotFound;p=[work->gpuDependencies indexGreaterThanIndex:p]) {
            MetalWork *producer=root->history[p];
            if(producer->scope!=scope) {
                MetalScopeAdd(&search,required,p);
                if(!producer->external)producer->needsFence=YES;
            }
        }
        MTLStages stages=work->stages;
        while(stages) {
            const MTLStages stage=stages&(~stages+1);stages&=~stage;
            MetalScopeMerge(&search,MetalScopeStage(imposed,stage),required);
            for(MetalWork *member in cohort)MetalScopeAdd(&search,MetalScopeStage(prefix,stage),member->ordinal);
        }
        if(work->hostConsumer) {
            scope=0;hosts=nil;[prefix removeAllObjects];[imposed removeAllObjects];
        }
    }
}
