"""The historical tiled recurrence, declared to the shared page compiler.

Numerical phases retain their source and exact group/page relations. The same
Execution that consumes FX and provider graphs composes their transfer effects.
"""
import torch

from mesh_relation import Algebra
from . import _stream, empty
from .backend import Call, ALLGATHER, BROADCAST, WRITE, bind_source
from .declarations import Execution
from .streamed import instance


def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


def _floor2(n):
    return 1 << (n.bit_length() - 1)


def _form(K, S, rows, ranks, resident_tiles, capabilities):
    """Compile geometry from the numerical recipe and declared resources."""
    threads, memory = (capabilities[k] for k in ('max_threads_per_threadgroup', 'threadgroup_memory_bytes'))
    tile = capabilities['simdgroup_matrix_edge']
    simd, cores = _stream.compilation['SIMD_WIDTH'], capabilities['hardware']['gpu']['cores']
    kt, st, rt = K // tile, S // tile, rows // tile
    workers = min(cores, max(kt, st) * rt)
    most = threads // simd

    def simdgroups(rg):
        groups = rt // rg
        return min(most, _floor2(max(-(-st * groups // workers) * kt, -(-kt * groups // workers) * st)))

    def splits(rg, sg):
        groups = rt // rg
        return _ksplit(-(-st * groups // workers), kt, sg), _ksplit(-(-kt * groups // workers), st, sg)

    sg = simdgroups(1)
    rg = max(rg for rg in (1 << i for i in range(rt.bit_length()))
             if rt % rg == 0 and max(kt, st) * (rt // rg) >= workers
             and simdgroups(rg) == sg and splits(rg, sg) == splits(1, sg))
    rb = rg
    while rb and sg * rb * tile * tile * 4 + 2 * capabilities['threadgroup_memory_alignment_bytes'] > memory:
        rb //= 2
    if not rb:
        raise ValueError('declared chain geometry exceeds its threadgroup memory capability')
    ks0, ks1 = splits(rg, sg)
    return dict(K=K, S=S, R=rows, G=workers, RG=rg, RB=rb, tile=tile,
                threads=sg * simd, simd=simd, ranks=ranks, ks0=ks0, ks1=ks1,
                linear=max(-(-kt // ks0), -(-st // ks1)) > resident_tiles)


def _share(position, group, extent, workers):
    q, r = divmod(extent, workers)
    return (f'(({group} < {r} and {group}*{q+1} <= ({position}) < ({group}+1)*{q+1}) or '
            f'({group} >= {r} and {group}*{q}+{r} <= ({position}) < ({group}+1)*{q}+{r}))')


def _dispatch(form, phase, first, last, width, rows, algebra):
    """Exact group-to-scalar relations of chain.metal's retained phase body."""
    K, S, R, G, tile, rg = (form[k] for k in ('K', 'S', 'R', 'G', 'tile', 'RG'))
    uses = []
    def use(side, slot, extent, condition):
        uses.append(dict(side=side, slot=slot,
            group_relation=algebra.map((G,), (extent,), lambda g, i: [condition(g[0], i[0])])) )
    if phase == 0:
        selected = lambda g, i: _share(f'floor(({i})/{width})*{K}+({i})%{width}', g, R*K, G)
        if not first:
            use('reads', 7, form['ranks']*R*K, lambda g,i:
                f'floor((({i})%{R*K})/{K}) < {rows} and ({i})%{K} < {width} and ' + _share(f'({i})%{R*K}',g,R*K,G))
            use('writes', 3, rows*width, selected)
        if not last:
            use('reads', 2, rows*width, selected)
            use('writes', 4, R*K, lambda g,i:_share(i,g,R*K,G))
        return tuple(uses)
    inner, columns, source, weight, target = (K,S,4,0,5) if phase == 1 else (S,K,5,1,6)
    row_groups = R // (tile*rg)
    tasks = columns // tile * row_groups
    use('writes',target,R*columns,lambda g,i:_share(
        f'floor((({i})%{columns})/{tile})*{row_groups}+floor(({i})/{columns*tile*rg})',g,tasks,G))
    use('reads',source,R*inner,lambda g,i:
        f'(exists c: 0 <= c < {columns//tile} and '+_share(f'c*{row_groups}+floor(({i})/{inner*tile*rg})',g,tasks,G)+')')
    use('reads',weight,inner*columns,lambda g,i:
        f'(exists r: 0 <= r < {row_groups} and '+_share(f'floor(({i})/{inner*tile})*{row_groups}+r',g,tasks,G)+')')
    return tuple(uses)


def _phase(form, phase, buffers, width, rows, algebra, sources, *, first=False, last=False):
    key=phase,first,last
    if key not in sources:
        arguments = tuple(form[k] for k in ('tile','K','S','R','G','RG','RB','threads','simd','ranks'))
        name, specialization = instance('chain_step', *arguments, phase, first, last)
        sources[key]=dict(source='chain.metal', name=name, instance=specialization,
            words=[form['ks0'],form['ks1'],width,rows], constant_slot=8,
            grid=(form['G'],1,1), threads=(form['threads'],1,1),
            dispatch=_dispatch(form,phase,first,last,width,rows,algebra),
            prefix='',indices={},coherent_accesses=True)
    parameters=sources[key]
    dispatch = parameters['dispatch']
    needed = {use['slot'] for use in dispatch}
    command = dict(kind='kernel', operands={str(slot):value for slot,value in enumerate(buffers) if slot in needed},
        parameters=parameters,before=(),after=())
    return dict(commands=(command,), output=None, pages=())


def operands(a, b, inputs, outputs, group, batch, resident_tiles, input_frames, capabilities, *, source_scope):
    """Assign page generations and retain the whole recurrence's declarations.

    Frame grouping only changes AOT page assignments. Source/destination masks
    exclude padded lanes from DMA; padded SIMD arithmetic stays in its existing
    numerical body. There are no runtime form lookups or publication wrappers.
    """
    steps, rows, width = inputs.shape
    tile = capabilities['simdgroup_matrix_edge']
    round_tile = lambda n:max(tile,-(-n//tile)*tile)
    K,S,R = round_tile(width),round_tile(a.shape[0]),round_tile(rows)
    form = _form(K,S,R,group.size(),resident_tiles,capabilities)
    algebra = Algebra()

    def weight_pages(weight,shape,tiled):
        padded=empty(shape,dtype=weight.dtype).zero_()
        padded[:weight.shape[0],:weight.shape[1]].copy_(weight)
        if not tiled:
            return padded
        result=empty(shape[1]//tile,shape[0]//tile,tile,tile,dtype=weight.dtype)
        result.copy_(padded.reshape(shape[0]//tile,tile,shape[1]//tile,tile).permute(2,0,1,3))
        return result

    if form['linear']:
        At,Bt=weight_pages(a,(S,K),False),weight_pages(b.T,(K,S),False)
    else:
        At,Bt=weight_pages(a.T,(K,S),True),weight_pages(b,(S,K),True)
    X,Y=empty(R,K,dtype=torch.half),empty(R,S,dtype=torch.half)
    # Two declared generations are enough for the same recurrent dataflow used
    # before this source migration. Their consumers and sends remain ordinary
    # page relations; the common compiler handles any physical alias hazards.
    generations=empty(2,batch,group.size(),R,K,dtype=torch.half)
    program,groups,order=[],[],[]
    windows=[]
    first=0
    while first < steps:
        count=min(batch,steps-first,input_frames-first%input_frames)
        windows.append((first,count,len(windows)%2))
        first+=count
    recurrent={}
    for first,count,generation in windows:
        recurrent.setdefault((count,generation),[]).append(first)
    input_starts={}
    for first in range(0,steps,input_frames):
        input_starts.setdefault(min(input_frames,steps-first),[]).append(first)
    definitions,invocations,phase_sources,products={},{},{},{}

    def crossing(kind,source,destination,statement,firsts,frames,frame_stride,slots):
        deliveries,calls=[],[]
        for producer in (range(group.size()) if kind == ALLGATHER else (0,)):
            identity=dict(scope=source_scope,statement=statement,producer=producer)
            for peer in range(group.size()):
                if producer==peer:
                    continue
                if group.rank()==producer:
                    value=source
                elif group.rank()==peer:
                    value=destination[producer] if kind==ALLGATHER else destination
                else:
                    value=torch.empty(source.shape,dtype=source.dtype,device='meta')
                deliveries.append((value,producer,peer))
                call=Call(WRITE,source.numel()*source.element_size(),torch.uint8,root=producer,destination=peer,
                          frames=frames,bindings=(slots,slots))
                calls.append(bind_source(call,source=identity,
                    occurrences=_stream.TensorLayout((len(firsts),frames),(1,1),0,{0:firsts}),
                    work=_stream.TensorLayout((source.numel()*source.element_size(),),(1,)),route_scope=identity,
                    occurrence_extent=steps))
        if not calls:
            return None
        command,calls=group.prepare_writes(deliveries,declarations=calls,frame_strides=[frame_stride]*len(calls))
        index=len(groups);groups.append(calls)
        return index,command

    def issue(definition):
        if definition is None:
            return None
        index,source=definition
        order.append(dict(first=index,count=1,repeat=source['calls'][0].frames))
        invocation=invocations.get(index,0)
        invocations[index]=invocation+1
        command=dict(source,invocation=invocation)
        at=len(program);program.append(command)
        return at,command

    def input_block(first):
        count=min(input_frames,steps-first)
        key=('input',count)
        if key not in definitions:
            starts=input_starts[count]
            value=inputs[starts[0]]
            slots=dict(instances=0,stride=input_frames*rows*width*inputs.element_size())
            definitions[key]=crossing(BROADCAST,value,value,'input',starts,count,rows*width,slots)
        issue(definitions[key])

    if steps:
        input_block(0)
    previous=None
    from .compilation import bind
    from .matrix import product
    for first,count,generation in windows:
        if first%input_frames==0 and first+input_frames<steps:
            input_block(first+input_frames)
        key=('recurrent',count,generation)
        if key not in definitions:
            values=generations[generation,0]
            definitions[key]=crossing(ALLGATHER,values[group.rank(),:rows,:width],values[:,:rows,:width],
                'recurrent',recurrent[count,generation],count,group.size()*R*K,dict(instances=1,stride=0))
        framed=issue(definitions[key])
        frame_orders=[]
        for j in range(count):
            t=first+j
            current=generations[generation,j]
            buffers=(At,Bt,inputs[t],outputs[max(0,t-1)],X,Y,current[group.rank()],previous)
            program.append(_phase(form,0,buffers,width,rows,algebra,phase_sources,first=t==0))
            if form['linear']:
                if (generation,j) not in products:
                    with bind(None,algebra=algebra):
                        commands=(*product(X,At.T,Y,act=True),*product(Y,Bt.T,current[group.rank()]))
                    products[generation,j]=dict(commands=commands,output=None,pages=())
                program.append(products[generation,j])
            else:
                program.extend(_phase(form,phase,buffers,width,rows,algebra,phase_sources) for phase in (1,2))
            if framed is not None:
                frame_orders.append(len(program)-1-framed[0])
            previous=current
        if framed is not None:
            framed[1]['frame_orders']=tuple(frame_orders)
    if steps:
        program.append(_phase(form,0,(At,Bt,None,outputs[-1],X,Y,None,previous),width,rows,algebra,phase_sources,last=True))
    execution=Execution(program,outputs,pages=(At,Bt,X,Y,generations),
        inputs=(At,Bt,*((inputs,) if group.rank()==0 else ())),algebra=algebra)
    return groups,order,execution


def prepare(execution):
    execution.realize()
    return dict(layouts=[],choices=[])
