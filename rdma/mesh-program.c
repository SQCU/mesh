#include "mesh-bridge.h"

static void busy_tell(void){
  const uint64_t busy=flow_count;
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
static uint32_t map_page,map_pages,map_room;
/* design/bridge-and-ipc.md#ledger: the ledger's clock, nanoseconds (CLOCK_MONOTONIC's are whole microseconds here), as
   Q5P's trace kept (mesh git 5702ea5, M27) */
/* design/bridge-and-ipc.md#records: the bytes the kernel can give without pressure (kern.memorystatus_level percent of
   the node's memory), read now */
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}

/* design/recovery.md#diagnosis: this node's health (mesh-plan.h struct mesh_health), sampled on the event loop from
   cheap reads only: sysctl, the host's VM statistics and the notify states of thermal pressure and of the power
   sources, whose description is read only when their notify state moved */
static struct mesh_health health;
static int power_token=-1,thermal_token=-1;
static uint64_t health_sent;
static mach_port_t host_port;
void sysctl_text(const char *name,char *out,size_t room){
  size_t size=room;
  if(sysctlbyname(name,out,&size,NULL,0))out[0]=0;
  out[room-1]=0;
}
static void power_read(struct mesh_health *h){
  h->power=MESH_POWER_UNKNOWN;h->charge=-1;h->minutes=-1;
  CFTypeRef info=IOPSCopyPowerSourcesInfo();
  if(!info)return;
  CFStringRef providing=IOPSGetProvidingPowerSourceType(info);
  if(providing && CFEqual(providing,CFSTR(kIOPMACPowerKey)))h->power=MESH_POWER_AC;
  else if(providing && CFEqual(providing,CFSTR(kIOPMBatteryPowerKey)))h->power=MESH_POWER_BATTERY;
  else if(providing && CFEqual(providing,CFSTR(kIOPMUPSPowerKey)))h->power=MESH_POWER_UPS;
  CFArrayRef list=IOPSCopyPowerSourcesList(info);
  for(CFIndex i=0;list && i<CFArrayGetCount(list);i++){
    CFDictionaryRef source=IOPSGetPowerSourceDescription(info,CFArrayGetValueAtIndex(list,i));
    if(!source)continue;
    int current=-1,most=-1,empty=-1;CFNumberRef value;
    if((value=CFDictionaryGetValue(source,CFSTR(kIOPSCurrentCapacityKey))))CFNumberGetValue(value,kCFNumberIntType,&current);
    if((value=CFDictionaryGetValue(source,CFSTR(kIOPSMaxCapacityKey))))CFNumberGetValue(value,kCFNumberIntType,&most);
    if((value=CFDictionaryGetValue(source,CFSTR(kIOPSTimeToEmptyKey))))CFNumberGetValue(value,kCFNumberIntType,&empty);
    if(current>=0 && most>0){h->charge=current*100/most;h->minutes=empty;break;}
  }
  if(list)CFRelease(list);
  CFRelease(info);
}
static void health_sample(struct mesh_health *h){
  h->sampled=clock_gettime_nsec_np(CLOCK_REALTIME);
  vm_statistics64_data_t vm;mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
  if(host_statistics64(host_port,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS){
    h->free=(uint64_t)vm.free_count*vm_kernel_page_size;h->wired=(uint64_t)vm.wire_count*vm_kernel_page_size;
  }
  h->available=available_now();
  int level=0;size_t size=sizeof level;
  if(!sysctlbyname("kern.memorystatus_vm_pressure_level",&level,&size,NULL,0))h->pressure=(uint32_t)level;
  uint64_t state=0;
  if(thermal_token>=0 && notify_get_state(thermal_token,&state)==NOTIFY_STATUS_OK)h->thermal=(uint32_t)state;
  int changed=0;
  if(power_token>=0 && notify_check(power_token,&changed)==NOTIFY_STATUS_OK && changed)power_read(h);
}
/* design/recovery.md#diagnosis: whether a power controller's record (`length` bytes of the ring kern.shutdownreason
   holds) is of `class`, the text before its comma */
static int record_is(const char *record,size_t length,const char *class){
  const size_t n=strlen(class);
  return length>=n && !strncmp(record,class,n) && (length==n || record[n]==',');
}
/* design/recovery.md#diagnosis: why the node last restarted, from the power controller's ring (newest first): the
   reset's own records, the ring's from its newest up to the next that opens an earlier boot's (a `por,por_vddrtc` or
   `wdog` record after the first), written whole into `out` (as many as its room holds, none cut); the watchdog where
   the newest is `wdog`, a sag where any of the reset's own records is `uv` (the controller watched its rails fall: an
   unplug or a supply under lockout), the supply cut where the newest is `por`, else other (none: unknown) */
static uint32_t reset_read(const char *ring,char *out,size_t room){
  uint32_t reset=MESH_RESET_UNKNOWN,records=0,sagged=0;
  size_t used=0;
  int whole=1;
  out[0]=0;
  for(const char *at=ring;;at+=strcspn(at," ")){
    at+=strspn(at," ");
    const size_t length=strcspn(at," ");
    if(!length || (records && (record_is(at,length,"wdog") || (length>=14 && !strncmp(at,"por,por_vddrtc",14)))))break;
    if(!records++)reset=record_is(at,length,"wdog")?MESH_RESET_WATCHDOG:record_is(at,length,"por")?MESH_RESET_SUPPLY:MESH_RESET_OTHER;
    sagged|=record_is(at,length,"uv");
    if(whole && used+(used?1:0)+length<room){if(used)out[used++]=' ';memcpy(out+used,at,length);used+=length;out[used]=0;}
    else whole=0;
  }
  return sagged && reset!=MESH_RESET_WATCHDOG?MESH_RESET_SAG:reset;
}
/* the boot's facts, read once: when the node booted, its boot session, and why it last restarted (reset_read) */
void health_init(void){
  host_port=mach_host_self();
  struct timeval booted;size_t size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  sysctl_text("kern.bootsessionuuid",health.session,sizeof health.session);
  char reason[512];sysctl_text("kern.shutdownreason",reason,sizeof reason);
  health.reset=reset_read(reason,health.shutdown,sizeof health.shutdown);
  notify_register_check(kIOPSNotifyAnyPowerSource,&power_token);
  notify_register_check("com.apple.system.thermalpressurelevel",&thermal_token);
  power_read(&health);
  health_sample(&health);
}
static void health_text(const struct mesh_health *h,char *out,size_t room){
  static const char *resets[]={"unknown","supply cut","watchdog","other","supply sagged"};
  static const char *thermals[]={"nominal","moderate","heavy","trapping","sleeping"};
  char power[64],thermal[16],when[32];
  if(h->power==MESH_POWER_AC && h->charge>=0)snprintf(power,sizeof power,"AC, battery %d%%",h->charge);
  else if(h->power==MESH_POWER_AC)snprintf(power,sizeof power,"AC");
  else if(h->power==MESH_POWER_BATTERY || h->power==MESH_POWER_UPS)
    snprintf(power,sizeof power,"%s %d%%, %d min to empty",h->power==MESH_POWER_UPS?"UPS battery":"battery",h->charge,h->minutes);
  else snprintf(power,sizeof power,"unknown");
  if(h->thermal<5)snprintf(thermal,sizeof thermal,"%s",thermals[h->thermal]);else snprintf(thermal,sizeof thermal,"%u",h->thermal);
  time_t booted=(time_t)(h->booted/1000000000u);struct tm utc;gmtime_r(&booted,&utc);strftime(when,sizeof when,"%Y-%m-%dT%H:%M:%SZ",&utc);
  snprintf(out,room,"power %s; memory available %.2f GiB, free %.2f GiB, wired %.2f GiB, pressure %s; thermal %s; booted %s "
           "(session %.36s), last restart %s (%.79s)",power,(double)h->available/1073741824.0,(double)h->free/1073741824.0,
           (double)h->wired/1073741824.0,
           h->pressure==4?"critical":h->pressure==2?"warning":h->pressure==1?"normal":"unread",thermal,when,h->session,
           h->reset<5?resets[h->reset]:"unknown",h->shutdown);
}
static uint32_t known_count;
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_held(known[i].node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
}
/* design/bridge-and-ipc.md#flows: a flow's program toward its destination: its slot's program, and its lists of queue k
   each way */
static struct mesh_tx *flow_tx(const struct flow *f){ return &mesh_client_at(M,f->owner)->tx[f->destination]; }
static uint32_t flow_length(const struct flow *f,int d,uint32_t k){ return atomic_load(mesh_order_length(M,f->owner,f->destination*MESH_QUEUES+k,d)); }
static struct mesh_transfer *flow_transfers(const struct flow *f,int d,uint32_t k){ return mesh_transfers(M,f->owner,f->destination*MESH_QUEUES+k,d); }
/* design/bridge-and-ipc.md#clients: a status told, and its client's start woken */
static void status_prepared(struct mesh_port_info *status,uint64_t owner){
  atomic_store_explicit(&status->prepared,owner,memory_order_release);
  os_sync_wake_by_address_all(&status->prepared,sizeof status->prepared,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static void status_failed(struct mesh_port_info *status,uint64_t owner,int64_t code,uint32_t domain,uint32_t where){
  atomic_store_explicit(&status->code,code,memory_order_relaxed);atomic_store_explicit(&status->domain,domain,memory_order_relaxed);
  atomic_store_explicit(&status->failed,where,memory_order_relaxed);atomic_store_explicit(&status->phase,MESH_STOPPED,memory_order_relaxed);
  status_prepared(status,owner);
}

/* The ring slot a transfer's invocation t addresses: (t - begin) mod its depth, at its slot pages. */
static uint32_t transfer_row(struct hdr *m,uint32_t fallback,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:fallback;
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/m->block)+chunk;
}

/* design/prepared-machine.md#M09: a chunk's span in the window its port's device registered when it came up; 0, or
   EINVAL for a page outside the window */
static int chunk_span(struct flow *link,uint32_t address,uint64_t remaining,struct ibv_sge *span){
  struct hdr *m=link->M;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
  if(offset>=wire.length)return EINVAL;
  *span=wire_span(&link->port->device,(uint64_t)m->block*m->pgsz,offset,remaining);
  return 0;
}

/* design/prepared-machine.md#M07, #M08, #M12: the word a landed chunk of invocation u of the transfer's slot `index` (0
   where each slot has its own rows) stores: its slot's completion word (mesh_host_inputs, the Metal transport), where
   the row's delivery names a word among the receiving program's own toward the flow's node (`words`, its cancellation
   range, [offset, offset + 8 count): a delivery another program wrote on the row, which outlives that program, is
   none of this one's); or, on a row the receiving program (`owner`) also publishes (a member that sends on what it
   receives: a relay rank's rows), that slot's publication's SEND cell once the record's last chunk landed, so the
   landing releases it to the next hop (the index hand-off, *release set); else the record's own (`sink`), which nothing
   reads.  NULL where the word is not in the window (#M09) */
static _Atomic uint64_t *receive_input(struct hdr *m,uint64_t owner,const struct mesh_cancel_range *words,uint32_t row,uint32_t index,
                                       uint32_t chunk,uint32_t chunks,uint32_t u,_Atomic uint64_t *sink,uint32_t *release){
  const struct mesh_delivery *delivery=mesh_delivery_at(m,row+chunk);
  uint64_t at=0;
  const uint64_t word=delivery->device_input+delivery->device_stride*u+sizeof(uint64_t)*(uint64_t)index*chunks;
  *release=0;
  if(delivery->device_input && words && word>=words->offset && (word-words->offset)/sizeof(uint64_t)<words->count)at=word;
  else if(chunk+1==chunks){
    uint32_t sends;
    const struct mesh_target *target=mesh_row_targets(m,owner,row,&sends);
    if(index<sends){at=target[index].cells+sizeof(struct mesh_send)*(uint64_t)u;*release=1;}
  }
  if(!at)return sink;
  return at>=m->data_off && at<=m->length-sizeof(uint64_t)?(void *)((char *)m+at):NULL;
}

/* design/bridge-and-ipc.md#cables: whether the flow carries the program's invocation t, and how many of a transfer's
   active invocations [begin, begin + active) it carries */
static int stripe_of(const struct flow *link,uint32_t t){ return t%link->stripes==link->stripe; }
static uint32_t stripe_count(const struct flow *link,uint32_t begin,uint32_t active){
  uint32_t first=(link->stripe+link->stripes-begin%link->stripes)%link->stripes;
  return first<active?(active-first+link->stripes-1)/link->stripes:0;
}

static _Atomic uint32_t ledger_ids;
/* a flow's ledger, made on its port's queue before its rings and cells are prepared or the flow listed, where the
   configuration keeps them, its send events kept a lane each (the lanes' gates retire them); a flow whose ledger cannot
   be allocated runs without one, logged */
static void ledger_open(struct flow *f){
  if(!ledger_kept || f->ledger)return;
  struct ledger *l=calloc(1,sizeof *l);
  uint32_t *words=l?malloc(3*(size_t)MESH_LINK_LANES*sizeof *words):NULL;
  if(!words){
    free(l);
    say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger: %s\n",f->port->index,(unsigned long long)f->key,
        f->far,f->stripe,f->stripes,strerror(ENOMEM));
    return;
  }
  for(size_t i=0;i<3*(size_t)MESH_LINK_LANES;i++)words[i]=LEDGER_NONE;
  l->id=atomic_fetch_add(&ledger_ids,1)+1;l->queues=MESH_LINK_LANES;l->started=ledger_ns();
  l->current=words;l->pending=words+MESH_LINK_LANES;l->last=words+2*(size_t)MESH_LINK_LANES;
  f->ledger=l;
}
/* the events a side's ledger holds: one pass of the flow's `once` cells or records, every pass of a cyclic program's
   the bound's latest (a ring: event e at e % room, its count published as each is stored, so a cut (ledger_cut) reads
   them while the flow runs) */
static uint32_t ledger_room(uint64_t once,int cyclic){ return cyclic || once>MESH_LEDGER_EVENTS?MESH_LEDGER_EVENTS:(uint32_t)once; }
/* design/bridge-and-ipc.md#ledger: taken by a cut and by a ledger's release, so a cut never reads a ledger freed */
static pthread_mutex_t ledger_lock=PTHREAD_MUTEX_INITIALIZER;
static int ledger_cell_order(const void *a,const void *b){
  const uint64_t x=((const struct ledger_cell *)a)->cell,y=((const struct ledger_cell *)b)->cell;
  return (x>y)-(x<y);
}
/* the ledger's header line, `cut` the events past the latest cut */
static void ledger_header(const struct flow *f,const struct ledger *l,uint32_t sends,uint32_t landings,const char *file,char *line,size_t room){
  snprintf(line,room,"{\"ledger\":%u,\"key\":\"%016llx\",\"node\":%u,\"peer\":%u,\"port\":%u,\"near\":%u,\"far\":%u,\"stripe\":%u,"
           "\"stripes\":%u,\"exclusive\":1,\"queues\":%u,\"qps\":%u,\"lanes\":[%u,%u],\"invocations\":%u,\"cyclic\":%u,\"started\":%llu,\"sends\":%u,"
           "\"send_room\":%u,\"landings\":%u,\"landing_room\":%u,\"cut\":[%u,%u],\"file\":%s%s%s}\n",l->id,(unsigned long long)f->key,M->node,
           f->port->peer,f->port->index,M->node,f->far,f->stripe,f->stripes,f->queues,f->queues,f->lane[0],f->lane[1],l->invocations,l->cyclic,
           (unsigned long long)l->started,sends,l->send_room,landings,l->landing_room,l->cut_sends,l->cut_landings,file?"\"":"",file?file:"null",
           file?"\"":"");
}
/* the event lines of the ledger's cells seen and records landed from the latest cut through `sends` and `landings` (the
   ring's latest where more passed), each naming its cell or record by queue, invocation, position, binding and cycle as
   the peer's ledger of the flow names the record or cell (rdma/ledger.py pairs them), to `out`, else the log */
static void ledger_events(const struct flow *f,struct ledger *l,uint32_t sends,uint32_t landings,FILE *out){
  char line[512];
  if(l->cells && !l->sorted){qsort(l->cell,l->cells,sizeof *l->cell,ledger_cell_order);l->sorted=1;}
  for(uint32_t e=sends-l->cut_sends>l->send_room?sends-l->send_room:l->cut_sends;e!=sends;e++){
    const struct ledger_send *s=l->send+e%l->send_room;
    const struct ledger_cell key={.cell=s->cell},*c=l->cells?bsearch(&key,l->cell,l->cells,sizeof *l->cell,ledger_cell_order):NULL;
    snprintf(line,sizeof line,"{\"s\":[%u,%lld,%lld,%lld,%lld,%u,%lld,%llu,%llu,%llu,%llu,%lld]}\n",l->id,c?(long long)c->queue:-1ll,
             c?(long long)c->invocation:-1ll,c?(long long)c->position:-1ll,c?(long long)c->binding:-1ll,s->cycle,c?(long long)c->chunks:-1ll,
             (unsigned long long)s->seen,(unsigned long long)s->first,(unsigned long long)s->posted,(unsigned long long)s->completed,
             c?(long long)c->bytes:-1ll);
    if(out)fputs(line,out);else say("%s",line);
  }
  for(uint32_t e=landings-l->cut_landings>l->landing_room?landings-l->landing_room:l->cut_landings;e!=landings;e++){
    const struct ledger_landing *r=l->landing+e%l->landing_room;
    const struct prepared_receive *record=f->receive && r->record<l->records?f->receive+r->record:NULL;
    const struct ledger_record *n=record?l->record+r->record:NULL;
    snprintf(line,sizeof line,"{\"r\":[%u,%lld,%lld,%lld,%lld,%lld,%lld,%u,%llu]}\n",l->id,record?(long long)record->queue:-1ll,
             record?(long long)record->invocation:-1ll,n?(long long)n->position:-1ll,n?(long long)n->binding:-1ll,n?(long long)n->chunk:-1ll,
             n?(long long)n->chunks:-1ll,r->cycle,(unsigned long long)r->landed);
    if(out)fputs(line,out);else say("%s",line);
  }
  l->cut_sends=sends;l->cut_landings=landings;
}
/* design/bridge-and-ipc.md#ledger: a flow's ledger written at its stop, on its port's queue once neither progress thread
   reads the flow: its header into the bridge's log, then its events past its latest cut: into the log where there are
   MESH_LEDGER_LINES or fewer, else into a file beside the log named by the program's key, the node, its peer, the
   stripe, the ledger and the time, which the header names; then freed */
static void ledger_write(struct flow *f){
  pthread_mutex_lock(&ledger_lock);
  struct ledger *l=f->ledger;
  f->ledger=NULL;
  pthread_mutex_unlock(&ledger_lock);
  if(!l)return;
  char path[PATH_MAX]={0},line[640];
  const char *file=NULL;
  FILE *out=NULL;
  struct stat log;
  if((uint64_t)(l->sends-l->cut_sends)+(l->landings-l->cut_landings)>MESH_LEDGER_LINES && !fstat(STDERR_FILENO,&log) && S_ISREG(log.st_mode) &&
     fcntl(STDERR_FILENO,F_GETPATH,path)!=-1){
    char *slash=strrchr(path,'/'),when[24];
    const time_t wall=time(NULL);struct tm utc;
    gmtime_r(&wall,&utc);strftime(when,sizeof when,"%Y%m%dT%H%M%SZ",&utc);
    if(slash && (size_t)snprintf(slash+1,sizeof path-(size_t)(slash+1-path),"ledger-%016llx-%u-%u-%u-%u-%s.jsonl",
                                 (unsigned long long)f->key,M->node,f->port->peer,f->stripe,l->id,when)<sizeof path-(size_t)(slash+1-path) &&
       (out=fopen(path,"w")))file=slash+1;
  }
  ledger_header(f,l,l->sends,l->landings,file,line,sizeof line);
  say("%s",line);
  if(out)fputs(line,out);
  ledger_events(f,l,l->sends,l->landings,out);
  if(out && fclose(out))say("link %u: ledger %u: %s: %s\n",f->port->index,l->id,file,strerror(errno));
  free(l->current);free(l->send);free(l->cell);free(l->landing);free(l->record);free(l);
}
/* design/bridge-and-ipc.md#ledger: a client's request for its program's ledger (MESH_REQUEST_LEDGER, the program's key):
   each running flow of the key's events since its latest cut appended to `path`, each after its header, while the flow
   runs (its progress threads store on; a cut reads the counts they published); ENOENT where no flow of the key keeps a
   ledger, ENOTSUP where the configuration keeps none */
static int ledger_cut(uint64_t key,const char *path){
  if(!ledger_kept)return ENOTSUP;
  FILE *out=NULL;
  int found=0;
  char line[640];
  pthread_mutex_lock(&ledger_lock);
  for(struct flow *f=flows;f;f=f->next){
    struct ledger *l=f->ledger;
    if(f->key!=key || !l || !f->running || f->ended)continue;
    if(!out && !(out=fopen(path,"a"))){const int error=errno;pthread_mutex_unlock(&ledger_lock);return error;}
    const uint32_t sends=__atomic_load_n(&l->sends,__ATOMIC_ACQUIRE),landings=__atomic_load_n(&l->landings,__ATOMIC_ACQUIRE);
    ledger_header(f,l,sends,landings,NULL,line,sizeof line);
    fputs(line,out);
    ledger_events(f,l,sends,landings,out);
    found=1;
  }
  pthread_mutex_unlock(&ledger_lock);
  if(out && fclose(out))return errno;
  return found?0:ENOENT;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M04, #M06, #M29 */
/* The SEND side: every SEND cell of the flow's invocations (#cables), its chain of chunk requests over its invocation's
   ring slot and its stream successor, a stream a queue of the program, posted on that queue's lane (the cell's queue
   pair and `queue` its lane, whose gate counts its frames: struct link `gate`; the cells' layout is
   mesh_transfers_prepare's).  A stream's cells run invocation by invocation, the publications of each in binding order,
   as the peer's ring of the same queue posts its receives on the lane's other end: TN3205's credit flow control holds a
   SEND until its matching receive is posted (metal-microbench docs/kernels.md Transport (git 7300d1e^)).  0, or the
   error (EINVAL: a transfer of more than the lane's one slot). */
static int sends_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->owner);
  struct mesh_tx *tx=flow_tx(link);
  const struct link *lanes=&link->port->link;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire);
  const int cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;
  const uint32_t queues=link->queues;
  size_t requests=0,cells=0;
  for(uint32_t k=0;k<queues;k++){
    struct mesh_transfer *out=flow_transfers(link,MESH_SEND,k);
    for(uint32_t i=0;i<flow_length(link,MESH_SEND,k);i++){
      if(out[i].count>1)return EINVAL;
      const size_t own=(size_t)out[i].count*stripe_count(link,out[i].begin,mesh_transfer_active(out+i,invocations));
      requests+=(size_t)(mesh_row_chunks(m,out[i].local_row,out[i].bytes)-1)*own;
      cells+=own;
    }
  }
  link->publications=(void *)((char *)m+tx->cells);
  /* design/prepared-machine.md#M29 */
  link->requests=aligned_alloc(128,(requests+1)*sizeof *link->requests);
  link->streams=calloc(queues,sizeof *link->streams);
  struct mesh_send **last=calloc(queues,sizeof *last),**terminal=calloc(queues,sizeof *terminal);
  if(!link->requests||!link->streams||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  link->request_count=requests;
  /* #ledger: the cells' names, each stream's position at its latest invocation (`positions`), and room for the events
     of one pass, a cyclic program's of every pass up to the bound */
  struct ledger *ledger=link->ledger;
  uint32_t positions[2]={0,0};
  if(ledger){
    ledger->invocations=invocations;ledger->cyclic=(uint32_t)cyclic;
    ledger->cell=malloc((cells+1)*sizeof *ledger->cell);
    ledger->send=malloc(((size_t)ledger_room(cells,cyclic)+1)*sizeof *ledger->send);
    if(!ledger->cell || !ledger->send){
      free(ledger->cell);free(ledger->send);ledger->cell=NULL;ledger->send=NULL;ledger=NULL;
      say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger of its SENDs: %s\n",link->port->index,
          (unsigned long long)link->key,link->far,link->stripe,link->stripes,strerror(ENOMEM));
    }else ledger->send_room=ledger_room(cells,cyclic);
  }
  size_t next=0;int error=0;
  for(uint32_t k=0;k<queues && !error;k++){
    const uint32_t count=flow_length(link,MESH_SEND,k),lane=link->lane[k];
    struct mesh_transfer *out=flow_transfers(link,MESH_SEND,k);
    struct ibv_qp *pair=lanes->provider.queues[lane].pair;
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      if(!out[i].count)continue;
      const uint32_t active=mesh_transfer_active(out+i,invocations),row=out[i].local_row,chunks=mesh_row_chunks(m,row,out[i].bytes);
      struct mesh_send *cells=link->publications+out[i].first;
      for(uint32_t u=0;u<active && !error;u++){
        if(!stripe_of(link,out[i].begin+u))continue;
        struct mesh_send *cell=cells+u;
        cell->pair=(uintptr_t)pair;cell->queue=lane;cell->chunks=chunks;
        atomic_store_explicit(&cell->port,link->port->index+1,memory_order_release);
        uint64_t remaining=out[i].bytes;
        for(uint32_t c=0;c<chunks && !error;c++){
          struct ibv_sge span;
          if((error=chunk_span(link,transfer_row(m,fallback,out+i,row,out[i].begin+u,c),remaining,&span)))break;
          remaining-=span.length;
          struct ibv_sge *entry=c?&link->requests[next+c-1].span:&cell->span;
          struct ibv_send_wr *request=c?&link->requests[next+c-1].request:&cell->request;
          *entry=span;
          /* the first request's wr_id stays the producer's nonzero argument (mesh_transfers_prepare),
             which a host producer reads at publication (mesh_host_publish) */
          *request=(struct ibv_send_wr){.wr_id=c?0:1,.next=c+1==chunks?NULL:&link->requests[next+c].request,
            .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
        }
        next+=chunks-1;
      }
      terminal[k]=cells+active;
    }
    /* the stream's cells in invocation order, its publications in binding order within one */
    positions[0]=positions[1]=0;
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && stripe_of(link,t);i++){
      const uint32_t active=mesh_transfer_active(out+i,invocations);
      if(!out[i].count || t<out[i].begin || t-out[i].begin>=active)continue;
      struct mesh_send *cell=link->publications+out[i].first+(t-out[i].begin);
      if(last[k])last[k]->successor=(uintptr_t)cell;
      else link->streams[k].cell=cell;
      last[k]=cell;
      /* #ledger: the cell's name, its position among the stream's cells of invocation t */
      if(ledger && ledger->cells<cells){
        if(positions[0]!=t+1){positions[0]=t+1;positions[1]=0;}
        ledger->cell[ledger->cells++]=(struct ledger_cell){(uintptr_t)cell,out[i].bytes,k,t,positions[1]++,out[i].binding,
          mesh_row_chunks(m,out[i].local_row,out[i].bytes)};
      }
    }
  }
  /* design/prepared-machine.md#M30: a cyclic program's last cell is followed by its first */
  for(uint32_t k=0;k<queues && !error;k++)if(last[k])last[k]->successor=cyclic?(uintptr_t)link->streams[k].cell:(uintptr_t)terminal[k];
  for(uint32_t k=0;k<queues;k++){link->streams[k].first=link->streams[k].cell;link->streams[k].cycle=0;}
  link->stream_count=0;
  for(uint32_t k=0;k<queues;k++)if(link->streams[k].cell)link->streams[link->stream_count++]=link->streams[k];
  free(last);free(terminal);
  return error;
}

/* design/prepared-machine.md#M08 */
/* The receive side: a ring a queue of the program, on that queue's lane, its records invocation by invocation (the
   flow's invocations, #cables), the transfers of each in binding order, a transfer's chunks in order, each storing its
   word (receive_input); each record posted once the record of its transfer `span` of the flow's own invocations before
   has landed (the ring's span over the program's invocations, divided among the stripes; a cyclic ring's first records
   wait on its last of the cycle before), so a ring's posted frames stay within its lane's queue pair and its share of
   the link's completion queues.  Its first records are posted by the port's receive thread once the flow is listed
   (flow_arm), never here.  0, or the error (EINVAL: a transfer of more than the lane's one slot, a word outside the
   window). */
static int receives_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->owner);
  const struct link *lanes=&link->port->link;
  /* design/prepared-machine.md#M09, #M10: the program's targets lie in the window, as each word it stores does */
  if(client->target_count && (client->targets<m->data_off || client->targets>m->length ||
     client->target_count>(m->length-client->targets)/sizeof(struct mesh_target)))return EINVAL;
  struct mesh_tx *tx=flow_tx(link);
  /* design/prepared-machine.md#M12: the program's own words toward the flow's node, where it has any */
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)link->destination+1);
  const struct mesh_cancellation *cancel=tx->cancel>=m->data_off && tx->cancel<=m->length-range?(const void *)((char *)m+tx->cancel):NULL;
  const struct mesh_cancel_range *words=cancel?cancel->ranges+link->destination:NULL;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  const uint32_t queues=link->queues;
  size_t records=0;
  for(uint32_t k=0;k<queues;k++){
    struct mesh_transfer *in=flow_transfers(link,MESH_RECEIVE,k);
    for(uint32_t i=0;i<flow_length(link,MESH_RECEIVE,k);i++){
      if(in[i].count>1)return EINVAL;
      records+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count*stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations));
    }
  }
  link->receive=aligned_alloc(128,(records+1)*sizeof *link->receive);
  link->rings=calloc(queues,sizeof *link->rings);
  if(!link->receive||!link->rings)return ENOMEM;
  link->receive_count=records;
  /* #ledger: the records' names, and room for the landings of one pass, a cyclic program's of every pass up to the
     bound */
  struct ledger *ledger=link->ledger;
  if(ledger){
    ledger->invocations=invocations;ledger->cyclic=(uint32_t)link->cyclic;
    ledger->record=malloc((records+1)*sizeof *ledger->record);
    ledger->landing=malloc(((size_t)ledger_room(records,link->cyclic)+1)*sizeof *ledger->landing);
    if(!ledger->record || !ledger->landing){
      free(ledger->record);free(ledger->landing);ledger->record=NULL;ledger->landing=NULL;ledger=NULL;
      say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger of its landings: %s\n",link->port->index,
          (unsigned long long)link->key,link->far,link->stripe,link->stripes,strerror(ENOMEM));
    }else{ledger->records=(uint32_t)records;ledger->landing_room=ledger_room(records,link->cyclic);}
  }
  /* a record is one frame at least, so a lane's share of frames also bounds the completions it can have pending in the
     receive completion queue the link's lanes share */
  for(uint32_t q=0;q<queues;q++)link->rings[q]=(struct receive_ring){.completed=-1,.span=UINT32_MAX,
    .capacity=MIN(lanes->provider.queues[link->lane[q]].receive_capacity,lanes->share>1?lanes->share:1)};
  size_t at=0;int error=0;
  for(uint32_t ring=0;ring<queues && !error;ring++){
    const uint32_t count=flow_length(link,MESH_RECEIVE,ring);
    struct mesh_transfer *in=count?flow_transfers(link,MESH_RECEIVE,ring):NULL;
    struct ibv_qp *pair=lanes->provider.queues[link->lane[ring]].pair;
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    /* firsts: the ring index of each transfer's own records, own invocation by own invocation (base[i] the first's) */
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)
      base[i+1]=base[i]+(in[i].count?stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations)):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(!in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
    }
    uint32_t position=0,latest=UINT32_MAX;
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(!in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      if(latest!=t){latest=t;position=0;}
      uint32_t u=t-in[i].begin,row=in[i].local_row,chunks=mesh_row_chunks(m,row,in[i].bytes);
      const uint32_t first=(link->stripe+link->stripes-in[i].begin%link->stripes)%link->stripes,mine=(u-first)/link->stripes;
      uint64_t remaining=in[i].bytes;
      for(uint32_t c=0;c<chunks;c++){
        struct ibv_sge span;
        if((error=chunk_span(link,transfer_row(m,fallback,in+i,row,t,c),remaining,&span)))break;
        remaining-=span.length;
        struct prepared_receive *record=link->receive+at++;
        if(!c)firsts[base[i]+mine]=(size_t)(record-r->first);
        uint32_t release;
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=receive_input(m,link->owner,words,row,0,c,chunks,u,&record->sink,&release),
          .pair=pair,.argument=1,.frames=(span.length+4095)/4096,.invocation=t,.queue=ring,.link=link};
        record->release=release;
        if(!record->input){error=EINVAL;break;}
        /* #ledger: the record's name, its transfer's position among the ring's transfers of invocation t */
        if(ledger)ledger->record[record-link->receive]=(struct ledger_record){position,in[i].binding,c,chunks};
      }
      position++;
    }
    r->count=(size_t)(link->receive+at-r->first);
    /* design/prepared-machine.md#M30 */
    if(link->cyclic)r->span=MIN(r->span,invocations>1?invocations-1:1);
    /* #cables: the span in the flow's own invocations, a stripe's share of the program's (one at least) */
    const uint32_t span=r->span==UINT32_MAX?UINT32_MAX:MAX(r->span/link->stripes,1u);
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t own=(uint32_t)(base[i+1]-base[i]);
      if(!own)continue;
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row,in[i].bytes);
      for(uint32_t j=0;j<own;j++){
        size_t here=firsts[base[i]+j],behind=0;
        if(span!=UINT32_MAX && j>=span)behind=here-firsts[base[i]+j-span];
        else if(span!=UINT32_MAX && link->cyclic && j+own>=span)behind=here+r->count-firsts[base[i]+j+own-span];
        else if(span!=UINT32_MAX && link->cyclic)behind=r->count;
        for(uint32_t c=0;c<chunks;c++)r->first[here+c].behind=(uint32_t)behind;
      }
    }
    r->span=span;
    free(base);free(firsts);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/bridge-and-ipc.md#flows: a flow's one job, on its port's queue: its ledger opened where kept, its receive
   records and its SEND chains prepared over its lanes, then the flow listed in the port's table (its rings' first
   records posted by the receive thread's first look at it, flow_arm).  No verbs call is made: its lanes' queue pairs
   are the link's, made at its pairing (design/bridge-and-ipc.md#lanes) */
static void flow_job(struct flow *f){
  f->job=1;atomic_store_explicit(&f->done,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    ledger_open(f);
    int error=receives_configure(f);
    if(!error)error=sends_configure(f);
    if(!error)error=port_list(f);
    f->result=error;
    atomic_store_explicit(&f->done,1,memory_order_release);
    wake();
  });
}
/* design/algorithm-sources.md#meshresult: a flow stopped on its port's queue, after its job: its lanes let go
   (lanes_release: no completion of them is taken as its own after it, and a lane it left with positions outstanding
   is DIRTY), then, where it failed, its program's pending words toward the far end cancelled (M12) and its status the
   failure naming the node `where`; the census printed (each of its queues' lane: the SEND requests and frames posted
   and retired through its gate, and its ring's records posted and landed and their bytes), its ledger written
   (#ledger), and its storage freed.  No verbs call is made */
static void flow_stop(struct flow *f,int64_t code,uint32_t domain,uint32_t where){
  const uint64_t owner=f->owner;const uint32_t destination=f->destination;struct mesh_cancellation *cancel=f->cancel;
  f->stopping=1;atomic_store_explicit(&f->stopped,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    const uint32_t dirty=lanes_release(f);
    if(code && owned(owner)){
      if(cancel)mesh_cancel(M,cancel,destination);
      status_failed(&mesh_client_at(M,owner)->status[destination],owner,code,domain,where);
    }
    const struct link *l=&f->port->link;
    for(uint32_t q=0;f->rings && q<f->queues;q++){
      const struct send_gate *g=l->gate+f->lane[q];
      say("link %u census queue=%u requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu "
          "lane=%u%s\n",f->port->index,q,(unsigned long long)g->requests,(unsigned long long)g->posted,(unsigned long long)g->retired,
          (unsigned long long)g->completions,f->rings[q].count,f->rings[q].posted,(unsigned long long)f->rings[q].landed,
          (unsigned long long)f->rings[q].bytes,f->lane[q],dirty>>f->lane[q]&1?" dirty":"");
    }
    ledger_write(f);
    free(f->receive);free(f->streams);free(f->rings);free(f->requests);
    f->receive=NULL;f->streams=NULL;f->rings=NULL;f->requests=NULL;
    atomic_store_explicit(&f->stopped,1,memory_order_release);
    wake();
  });
}
/* a flow freed, its index in its port's table given back */
static void flow_free(struct flow *f){
  for(struct flow **at=&flows;*at;at=&(*at)->next)if(*at==f){*at=f->next;break;}
  f->port->flows--;
  f->port->reserved[f->slot/64]&=~(UINT64_C(1)<<(f->slot%64));
  free(f);
  flow_count--;busy_tell();
}
/* design/algorithm-sources.md#meshresult: a program on a port failed as a value, its status the failure naming the node
   `where`, and every flow of the same client program toward the node (its other cables) with it.  No message tells
   the peer: its programs end by what ends them at that end (the link's loss there, its caller's revoke as its words,
   design/bridge-and-ipc.md#protocol) */
void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t where){
  if(f->ended)return;
  f->ended=1;
  say("link %u: program %016llx toward node %u (stripe %u of %u) failed at node %u: code %lld domain %u\n",f->port->index,
    (unsigned long long)f->key,f->far,f->stripe,f->stripes,where,(long long)code,domain);
  flow_stop(f,code,domain,where);
  for(struct flow *g=flows;g;g=g->next)if(g->owner==f->owner && g->destination==f->destination)flow_fail(g,code,domain,where);
}
/* a client's program toward its destination `d` failed as a value naming the node `where`: its flows failed, and its
   status the failure */
static void program_fail(uint64_t owner,uint32_t d,int64_t code,uint32_t where){
  for(struct flow *g=flows;g;g=g->next)if(g->owner==owner && g->destination==d)flow_fail(g,code,4,where);
  if(owned(owner))status_failed(&mesh_client_at(M,owner)->status[d],owner,code,4,where);
}
/* a program that ended, its client gone or its slot given back: its flows stopped, nothing told */
static void flow_end(struct flow *f){
  if(f->ended)return;
  f->ended=1;
  flow_stop(f,0,0,0);
}
/* design/bridge-and-ipc.md#clients: a client program's status once every flow of it toward the node runs */
static void program_paired(struct flow *f){
  for(struct flow *g=flows;g;g=g->next)
    if(g->owner==f->owner && g->destination==f->destination && (!g->running || g->job))return;
  if(!owned(f->owner))return;
  struct mesh_port_info *status=&mesh_client_at(M,f->owner)->status[f->destination];
  atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
  status_prepared(status,f->owner);
}
/* a flow's job's end taken, on the event loop: a failure the flow's, else it runs */
static void flow_done(struct flow *f){
  f->job=0;
  if(f->ended)return;
  if(f->result){flow_fail(f,f->result,1,M->node);return;}
  f->running=1;
  program_paired(f);
}
/* design/bridge-and-ipc.md#flows: a flow made on a port toward `far` for the client's program `owner`'s destination `d`
   (its execution `key`), its `queues` queues on the lanes `lane` (held by it from now: lanes_leased found them free),
   the program's invocations t with t mod stripes = stripe (#cables), an index of the port's table its own (MESH_FLOWS
   a port); NULL where the table is full */
static struct flow *flow_make(struct port *port,uint64_t owner,uint32_t d,uint64_t key,uint32_t far,uint32_t queues,const uint32_t *lane,
                              uint32_t stripe,uint32_t stripes){
  uint32_t slot=0;
  while(slot<MESH_FLOWS && port->reserved[slot/64]>>(slot%64)&1)slot++;
  if(slot==MESH_FLOWS)return NULL;
  struct flow *f=calloc(1,sizeof *f);
  if(!f)return NULL;
  const struct mesh_queue *queue=port->link.provider.queues+lane[0];
  const uint64_t cancel=mesh_client_at(M,owner)->tx[d].cancel;
  *f=(struct flow){.next=flows,.far=far,.queues=queues,.stripe=stripe,.stripes=stripes,.destination=d,.lane={lane[0],queues>1?lane[1]:0},
    .slot=slot,.generation=++port->generation[slot],.lanes=UINT32_C(1)<<lane[0]|(queues>1?UINT32_C(1)<<lane[1]:0),.key=key,.owner=owner,
    .M=M,.port=port,.post_send=queue->send,.post_receive=queue->receive,.cancel=cancel?(void *)((char *)M+cancel):NULL};
  port->reserved[slot/64]|=UINT64_C(1)<<(slot%64);
  if(slot>=atomic_load(&port->high))atomic_store(&port->high,slot+1);
  for(uint32_t q=0;q<queues;q++)atomic_store_explicit(&port->link.holder[lane[q]],f,memory_order_release);
  flows=f;port->flows++;flow_count++;busy_tell();
  return f;
}

/* the queues of a client's program toward its destination `d`: its highest queue with a list, plus one */
static uint32_t program_queues(uint64_t owner,uint32_t d){
  uint32_t queues=0;
  for(uint32_t q=0;q<MESH_QUEUES;q++)for(int w=0;w<2;w++)if(atomic_load(mesh_order_length(M,owner,d*MESH_QUEUES+q,w)))queues=q+1;
  return queues;
}
/* design/bridge-and-ipc.md#ports: the ports paired with `node`, in the cable order both ends read alike, by the index
   the lower node gave each cable */
static uint32_t cable_rank(const struct port *port){ return M->node<port->peer?port->index:port->peer_port; }
static uint32_t cables(uint32_t node,struct port **out){
  uint32_t n=0;
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted && ports[p].peer==node)out[n++]=ports+p;
  for(uint32_t i=1;i<n;i++)for(uint32_t j=i;j>0 && cable_rank(out[j])<cable_rank(out[j-1]);j--){struct port *t=out[j];out[j]=out[j-1];out[j-1]=t;}
  return n;
}
/* design/bridge-and-ipc.md#clients, #lanes, #cables: a started program's destination taken onto its lanes.  Its queues
   (at most two, of one slot each: its communicator's lane pair, queue k on the lane its slot names, mesh.h struct
   mesh_tx `lane`, which its client leased at the communicator's init, mesh.h struct hdr `lanes`) ride each cable to its
   node on which the client holds that lease, in the cable order both ends read alike, flow j of n carrying the
   program's invocations t with t mod n = j; no message is sent, the peer's program of the communicator riding the same
   lanes from the other end, each end posting its receives in its program's order and TN3205's credit holding a SEND
   until its receive is posted.  A program reaches only a node on a cable (a relay is a member rank of the communicator
   with no contribution: DESIGN.md decision 1).  Refused as a value naming the node where it is this node (EINVAL) or no
   cable reaches it (EHOSTUNREACH); naming this node where its shape is not a lane pair's (EINVAL), no cable carries its
   lease (ENOLCK), or a port's table is full (E2BIG) */
static void program_take(struct mesh_client *client,uint64_t owner,uint32_t d){
  const uint32_t node=client->tx[d].node,queues=program_queues(owner,d);
  const uint32_t lane[2]={client->tx[d].lane[0],client->tx[d].lane[1]};
  if(!node || node==M->node){status_failed(&client->status[d],owner,EINVAL,4,node);return;}
  if(client->tx[d].slots!=1 || !queues || queues>2 || (queues==2 && lane[0]==lane[1])){
    status_failed(&client->status[d],owner,EINVAL,4,M->node);return;
  }
  struct port *cable[MESH_PORTS];
  const uint32_t n=cables(node,cable);
  if(!n){status_failed(&client->status[d],owner,EHOSTUNREACH,4,node);return;}
  uint32_t leased=0;
  for(uint32_t j=0;j<n;j++)if(lanes_leased(cable[j],owner,lane,queues))cable[leased++]=cable[j];
  if(!leased){status_failed(&client->status[d],owner,ENOLCK,4,M->node);return;}
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
  for(uint32_t j=0;j<leased;j++){
    struct flow *f=flow_make(cable[j],owner,d,client->key,node,queues,lane,j,leased);
    if(!f){program_fail(owner,d,E2BIG,M->node);return;}
    atomic_fetch_or_explicit(&client->ports,UINT32_C(1)<<cable[j]->index,memory_order_release);
    flow_job(f);
  }
}

/* design/bridge-and-ipc.md#ports, #region: the map, a record a node: this node's own, rebuilt from its port table when
   that changes (its sequence one more), and every other node's newest heard; each change written between two
   notifications of the map serial (odd while it writes), the records in arena pages the bridge leases and leases again
   larger as the mesh grows; nodes no paired path reaches dropped */
static int map_grow(uint64_t records){
  if(records<=map_room)return 0;
  uint32_t pages=(uint32_t)((records*2*sizeof(struct mesh_record)+M->pgsz-1)/M->pgsz),page=mesh_arena_alloc(&bridge,pages,1);
  if(page==MESH_ABSENT)return ENOMEM;
  memcpy(mesh_at(M,page),mesh_records(M),(size_t)atomic_load(&M->map_nodes)*sizeof(struct mesh_record));
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,page)-(unsigned char *)M));
  if(map_pages)mesh_bits_clear(mesh_arena_bits(M),map_page,map_pages);
  map_page=page;map_pages=pages;map_room=(uint32_t)((uint64_t)pages*M->pgsz/sizeof(struct mesh_record));
  return 0;
}
/* RFC 2328 s14: the map with `record` put in (its node's held record replaced, else added) and every record no paired
   path from this node reaches left out, written only where it differs from the map held (so a record that changes
   nothing, or that the prune drops at once, moves no serial) */
static void map_put(const struct mesh_record *record){
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  uint64_t at=0;
  while(at<n && r[at].node!=record->node)at++;
  const uint64_t count=at==n?n+1:n;
  struct mesh_record *next=malloc(count*sizeof *next);
  unsigned char *reached=calloc(count,1);
  uint32_t *order=malloc(count*sizeof *order),head=0,tail=0;
  if(!next || !reached || !order){free(next);free(reached);free(order);say("map: no memory to put node %u's record\n",record->node);return;}
  memcpy(next,r,(size_t)n*sizeof *r);
  memcpy(next+at,record,sizeof *record);
  for(uint64_t k=0;k<count;k++)if(next[k].node==M->node){reached[k]=1;order[tail++]=(uint32_t)k;}
  while(head<tail){
    uint32_t a=order[head++];double beta;
    for(uint64_t b=0;b<count;b++)if(!reached[b] && mesh_joined(next+a,next+b,&beta)){reached[b]=1;order[tail++]=(uint32_t)b;}
  }
  /* design/recovery.md#diagnosis: a node the prune drops is logged with its last health first (one put now and never
     reached yet is not a loss) */
  for(uint64_t k=0;k<count;k++)if(!reached[k] && !(at==n && k==at)){
    struct known *gone=known_at(next[k].node);char text[768];health_text(&next[k].health,text,sizeof text);
    say("node %u (%s) unreachable; its last health, heard %.1f s before: %s\n",next[k].node,next[k].name,
        gone && gone->heard?(double)(now-gone->heard)/1e9:-1.0,text);
    if(gone)gone->gone=now;
  }
  uint64_t kept=0;
  for(uint64_t k=0;k<count;k++)if(reached[k])memmove(next+kept++,next+k,sizeof *next);
  if(kept!=n || memcmp(next,r,(size_t)n*sizeof *r)){
    mesh_notify(&M->map);
    if(map_grow(kept))say("map: no lease for %llu records; %llu kept\n",(unsigned long long)kept,(unsigned long long)map_room);
    if(kept>map_room)kept=map_room;
    memcpy(mesh_records(M),next,(size_t)kept*sizeof *next);
    atomic_store(&M->map_nodes,kept);
    mesh_notify(&M->map);
  }
  free(next);free(reached);free(order);
}
/* design/recovery.md#diagnosis: a record whose links (each port's peer, phase and bandwidth), name, boot and facts are
   the held one's changes only its health, sequence and ports' counts (their codes, pairings and queue pairs), written
   under the region's health serial and never the map's: the map serial moves only where the mesh changes (a
   communicator's session reads the map again where it moved, and reopens only where its own routes or their links'
   costs changed: nccl-mesh.c map_digest, session_ensure), and neither a health sample nor a queue pair is a change of
   the mesh */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->ports==b->ports && a->boot==b->boot && !memcmp(a->name,b->name,sizeof a->name) &&
    a->window==b->window && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t p=0;same && p<MESH_PORTS;p++)
    same=atomic_load(&a->port[p].peer)==atomic_load(&b->port[p].peer) && atomic_load(&a->port[p].phase)==atomic_load(&b->port[p].phase) &&
      atomic_load(&a->port[p].bandwidth)==atomic_load(&b->port[p].bandwidth);
  return same;
}
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  memcpy(held->port,record->port,sizeof held->port);
  mesh_notify(&M->health);
}
struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
void record_changed(void){
  struct mesh_record own={.node=M->node,.ports=port_count,.boot=M->boot,.sequence=++sequence,.window=wire.length,.memory=memory};
  memcpy(own.name,host,sizeof own.name);
  memcpy(own.model,model,sizeof own.model);
  memcpy(own.port,mesh_links(M),sizeof own.port);
  own.health=health;
  struct mesh_record *held=record_held(M->node);
  if(held && record_links_same(held,&own))health_put(held,&own);
  else map_put(&own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* design/recovery.md#five-seconds: this node's record told again at its current sequence on every greeted port, each
   liveness turn (MESH_PAUSE_NS), so each control socket carries a message well within its peer's MESH_PLATFORM_NS
   (mesh-link.c port_silent); a receiver drops a record no newer than its newest (record_heard, RFC 2328 s13), its map and
   serials unmoved.  Told from a copy: a port lost meanwhile may write the map */
void record_retell(void){
  const struct mesh_record *held=record_held(M->node);
  if(!held)return;
  struct mesh_record own;memcpy(&own,held,sizeof own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
static uint64_t low_told;
static void memory_watch(void){
  if(health.pressure<2 && health.available>=memory/MESH_LOW_SHARE){low_told=0;return;}
  if(low_told && now-low_told<MESH_LOW_NS)return;
  low_told=now;
  int count=proc_listallpids(NULL,0);
  pid_t *pids=count>0?calloc((size_t)count+16,sizeof *pids):NULL;
  struct { pid_t pid; uint64_t bytes; } top[MESH_LOW_LISTED]={{0,0}};
  count=pids?proc_listallpids(pids,(int)(((size_t)count+16)*sizeof *pids)):0;
  for(int i=0;i<count;i++){
    struct rusage_info_v0 use;
    if(proc_pid_rusage(pids[i],RUSAGE_INFO_V0,(rusage_info_t *)&use))continue;
    for(int k=0;k<MESH_LOW_LISTED;k++)if(use.ri_phys_footprint>top[k].bytes){
      memmove(top+k+1,top+k,(MESH_LOW_LISTED-1-(size_t)k)*sizeof *top);
      top[k].pid=pids[i];top[k].bytes=use.ri_phys_footprint;break;
    }
  }
  free(pids);
  char listed[512]="";size_t used=0;
  for(int k=0;k<MESH_LOW_LISTED && top[k].bytes;k++){
    char name[64]="?";proc_name(top[k].pid,name,sizeof name);
    used+=(size_t)snprintf(listed+used,used<sizeof listed?sizeof listed-used:0,"%s%s (%d) %.2f GiB",k?", ":"",name,top[k].pid,
                           (double)top[k].bytes/1073741824.0);
    if(used>=sizeof listed)break;
  }
  say("memory low: %.2f GiB available of %.2f GiB, pressure %s, the window %.2f GiB; largest: %s\n",
      (double)health.available/1073741824.0,(double)memory/1073741824.0,health.pressure==4?"critical":health.pressure==2?"warning":"normal",
      (double)((uint64_t)mesh_arena_pages(M)*M->pgsz)/1073741824.0,listed);
}
void health_turn(void){
  struct mesh_health sampled=health;
  health_sample(&sampled);
  const int urgent=sampled.power!=health.power || (sampled.power!=MESH_POWER_AC && sampled.charge!=health.charge) ||
    sampled.pressure!=health.pressure || sampled.thermal!=health.thermal;
  health=sampled;
  if(urgent){char text[768];health_text(&health,text,sizeof text);say("health: %s\n",text);}
  if(urgent || now-health_sent>=MESH_HEALTH_NS){health_sent=now;record_changed();}
  memory_watch();
}
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard;
  if(bytes!=sizeof heard)return;
  memcpy(&heard,payload,sizeof heard);
  heard.name[sizeof heard.name-1]=0;
  heard.health.session[sizeof heard.health.session-1]=0;heard.health.shutdown[sizeof heard.health.shutdown-1]=0;
  if(!heard.node || heard.node==M->node || heard.ports>MESH_PORTS)return;
  heard.model[sizeof heard.model-1]=0;
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  /* design/bridge-and-ipc.md#records, RFC 2328 s13: a record no newer than the newest heard of its node, the map holding
     it or not, is dropped and not sent on, so a record the map pruned never circulates (a node's later boot is always
     newer) */
  if(k && k->boot && (heard.boot<k->boot || (heard.boot==k->boot && heard.sequence<=k->sequence))){
    /* the newest known of a node the map dropped while no paired path reached it: put again where a path now does,
       and sent on only then (a pruned record that no path reaches stops here) */
    if(!held && heard.boot==k->boot && heard.sequence==k->sequence){
      map_put(&heard);
      for(uint32_t p=0;record_held(heard.node) && p<port_count;p++)
        if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
    }
    return;
  }
  if(held && (heard.boot<held->boot || (heard.boot==held->boot && heard.sequence<=held->sequence)))return;
  /* design/recovery.md#diagnosis: a node that joins or comes back, and a peer's urgent change, are logged here */
  char text[768];
  if(!held && k && (k->gone || !k->heard)){
    health_text(&heard.health,text,sizeof text);
    if(k->gone)say("node %u (%s) back after %.1f s, %s: %s\n",heard.node,heard.name,(double)(now-k->gone)/1e9,
                   strncmp(k->session,heard.health.session,sizeof k->session)?"a new boot":"the same boot",text);
    else say("node %u (%s) joined: %s\n",heard.node,heard.name,text);
  }
  if(held && (held->health.power!=heard.health.power || held->health.pressure!=heard.health.pressure ||
              held->health.thermal!=heard.health.thermal ||
              (heard.health.power!=MESH_POWER_AC && held->health.charge!=heard.health.charge))){
    health_text(&heard.health,text,sizeof text);
    say("node %u (%s) health: %s\n",heard.node,heard.name,text);
  }
  if(k){k->heard=now;k->gone=0;k->boot=heard.boot;k->sequence=heard.sequence;memcpy(k->session,heard.health.session,sizeof k->session);}
  if(held && record_links_same(held,&heard))health_put(held,&heard);
  else map_put(&heard);
  for(uint32_t p=0;p<port_count;p++)if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
}
/* the ports paired with `node` whose links have their stripe lanes served */
static uint32_t stripe_cables(uint32_t node){
  uint32_t n=0;
  for(uint32_t p=0;p<port_count;p++)
    n+=ports[p].greeted && ports[p].peer==node && (atomic_load_explicit(&ports[p].link.served,memory_order_acquire)>>MESH_LANE_STRIPES&1);
  return n;
}
/* design/bridge-dataflow.md#stripes: every registered ring's next request read on the event loop: a ring whose client
   is gone deregistered; a request naming a node toward which a cable of this node's has its stripe lane served is left
   to the send threads of those ports, which take it; any other fails as a value at once (its writer's word UINT64_MAX,
   its ring's code): its node this node (EINVAL), or no cable's with its stripe lane served (EHOSTUNREACH: a stripe
   goes over one link, its node a neighbour whose link is up).  Nothing waits for a link to pair or its stripe lane to
   be made again (the wait had no event that ends it: output_data/structural-20261018/WAITS.md).  At the stop
   (`stopping` its code, design/bridge-and-ipc.md#stop) every request not yet taken fails with it */
void stripes_requests_turn(int64_t stopping){
  _Atomic uint64_t *rings=mesh_rings(M);
  for(uint32_t r=0;r<MESH_RINGS;r++){
    uint64_t at=atomic_load_explicit(rings+r,memory_order_acquire);
    if(!at)continue;
    struct mesh_requests *q=at>=M->data_off && at<=M->length-sizeof(struct mesh_requests)?(void *)((char *)M+at):NULL;
    if(!q || !owned(q->owner) || !q->entries || (q->entries&(q->entries-1)) || q->entries>(M->length-at-sizeof *q)/sizeof(struct mesh_stripe)){
      atomic_compare_exchange_strong(rings+r,&at,0);
      continue;
    }
    for(uint64_t looked=0;looked<q->entries;looked++){
      uint64_t k=atomic_load_explicit(&q->taken,memory_order_acquire);
      const struct mesh_stripe *e=q->request+(k&(q->entries-1));
      if(atomic_load_explicit(&e->ready,memory_order_acquire)!=k+1)break;
      const uint32_t node=e->node;
      const int64_t code=stopping?stopping:node==M->node?EINVAL:EHOSTUNREACH;
      if(!stopping && node!=M->node && stripe_cables(node))break;
      const uint64_t word=e->word,owner=q->owner;
      if(!atomic_compare_exchange_strong_explicit(&q->taken,&k,k+1,memory_order_acq_rel,memory_order_relaxed))continue;
      if(atomic_load_explicit(&q->code,memory_order_relaxed)!=code)
        say("stripes: a request of client %llx toward node %u refused: %s\n",(unsigned long long)owner,node,strerror((int)code));
      stripe_failed(at,owner,word,code);
      if(!stopping)break;
    }
  }
}
static struct channel channels[MESH_CHANNELS];
static struct asking askings[MESH_CHANNELS];
static struct serving servings[MESH_SERVICES];
static uint32_t channel_sequence;
static pid_t orphans[MESH_CHANNELS];
static void fd_watch(int fd,int filter,int on,uint64_t udata){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)fd,filter,EV_ADD|(on?EV_ENABLE:EV_DISABLE),0,0,udata,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
static void fd_reset(int fd){
  struct linger now_={1,0};
  setsockopt(fd,SOL_SOCKET,SO_LINGER,&now_,sizeof now_);
  close(fd);
}
void fd_answer(int fd,int32_t code){
  if(send(fd,&code,sizeof code,0)!=(ssize_t)sizeof code)return;
}
static uint64_t channel_udata(const struct channel *c,int s){ return TAG_CHANNEL|(uint64_t)(c-channels)<<1|(uint64_t)s; }
static void channel_say(const struct channel *c,int s,uint32_t kind,uint64_t count,int64_t code,const void *payload,uint32_t bytes){
  if(c->end[s].kind==END_PORT)port_say(c->end[s].port,(struct message){.kind=kind,.key=c->key,.from=c->origin,.to=c->destination,
                                        .count=(uint32_t)count,.code=code},payload,bytes);
}
/* a channel ended: where `code`, each of its ends but `from` told so (a port end CLOSE, a socket end reset; a client
   still awaiting its answer answered with the code); the slot freed first, so a port lost meanwhile finds it free */
static void channel_end(struct channel *c,int64_t code,int from){
  struct channel was=*c;
  memset(c,0,sizeof *c);
  if(code)say("channel %u.%u to node %u's %s: %s\n",was.origin,(uint32_t)was.key,was.destination,was.service,strerror((int)code));
  if(was.pid && !was.exited){
    kill(-was.pid,SIGTERM);
    for(uint32_t i=0;i<MESH_CHANNELS;i++)if(!orphans[i]){orphans[i]=was.pid;break;}
  }
  for(int s=0;s<2;s++){
    struct end *e=was.end+s;
    if(e->kind==END_SOCKET && e->fd>=0){
      if(code && !was.open && s==0 && was.origin==M->node){fd_answer(e->fd,(int32_t)code);close(e->fd);}
      else if(code)fd_reset(e->fd);
      else close(e->fd);
    }
    if(e->kind==END_PORT && code && s!=from)channel_say(&was,s,CHANNEL_CLOSE,0,code,NULL,0);
    free(e->queue);
  }
  free(was.held);
}
static struct channel *channel_new(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)if(!channels[i].used){
    struct channel *c=channels+i;
    memset(c,0,sizeof *c);c->used=1;c->end[0].fd=c->end[1].fd=-1;
    return c;
  }
  return NULL;
}
static struct port *channel_hop(uint32_t node){
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  struct port *port=hops>=2 && cables(route[1],next)>0?next[0]:NULL;
  free(route);
  return port;
}
static void channel_pump(struct channel *c);
static void hold_done(struct channel *c);
static int64_t child_spawn(struct channel *c,int s);
static void child_ended(struct channel *c);
/* the bytes queued for end `t` given to it as far as it takes them, each granted back to the other end where it is a
   port; once none is left and the other end has ended, end `t` told so (a socket's write side shut, a port CHANNEL_SHUT, the
   bridge's own service answering) */
static void end_drain(struct channel *c,int t){
  struct end *e=c->end+t,*o=c->end+(1-t);
  size_t given=0;
  while(given<e->queued){
    size_t n=e->queued-given;
    if(e->kind==END_SOCKET){
      ssize_t sent=send(e->fd,e->queue+given,n,0);
      if(sent<0 && (errno==EAGAIN || errno==EINTR))break;
      if(sent<=0){channel_end(c,sent?errno:EPIPE,-1);return;}
      n=(size_t)sent;
    }else if(e->kind==END_EXEC){
      const unsigned char *line=memchr(e->queue+given,'\n',n);
      if(line)n=(size_t)(line-(e->queue+given))+1;
      if(c->held_bytes+n>MESH_HOLD_BYTES){channel_end(c,EMSGSIZE,-1);return;}
      unsigned char *grown=realloc(c->held,c->held_bytes+n+1);
      if(!grown){channel_end(c,ENOMEM,-1);return;}
      c->held=grown;memcpy(c->held+c->held_bytes,e->queue+given,n);c->held_bytes+=n;
      if(o->kind==END_PORT)channel_say(c,1-t,CHANNEL_WINDOW,n,0,NULL,0);
      given+=n;
      if(!line)continue;
      memmove(e->queue,e->queue+given,e->queued-given);e->queued-=given;given=0;
      const int64_t error=child_spawn(c,t);
      if(error){channel_end(c,error,-1);return;}
      continue;
    }else if(e->kind==END_PORT){
      n=MIN(n,MIN(e->credit,MESH_PACKET));
      if(!n)break;
      channel_say(c,t,CHANNEL_DATA,0,0,e->queue+given,(uint32_t)n);
      e->credit-=n;
      if(!c->used)return;
    }else{
      if(c->held_bytes+n>MESH_HOLD_BYTES){channel_end(c,EMSGSIZE,-1);return;}
      unsigned char *grown=realloc(c->held,c->held_bytes+n);
      if(!grown){channel_end(c,ENOMEM,-1);return;}
      c->held=grown;memcpy(c->held+c->held_bytes,e->queue+given,n);c->held_bytes+=n;
    }
    given+=n;
    if(o->kind==END_PORT)channel_say(c,1-t,CHANNEL_WINDOW,n,0,NULL,0);
  }
  memmove(e->queue,e->queue+given,e->queued-given);e->queued-=given;
  if(e->kind==END_SOCKET){
    const int want=e->queued>0;
    if(want!=e->writing){e->writing=want;fd_watch(e->fd,EVFILT_WRITE,want,channel_udata(c,t));}
  }
  if(!e->queued && o->ended && !e->told && e->kind==END_EXEC){
    const int64_t error=child_spawn(c,t);
    if(error){channel_end(c,error,-1);return;}
  }
  if(!e->queued && o->ended && !e->told){
    e->told=1;
    if(e->kind==END_SOCKET)shutdown(e->fd,SHUT_WR);
    else if(e->kind==END_PORT)channel_say(c,t,CHANNEL_SHUT,0,0,NULL,0);
    else hold_done(c);
    if(!c->used)return;
  }
  if(c->end[0].told && c->end[1].told){channel_end(c,0,-1);return;}
  channel_pump(c);
}
static void end_put(struct channel *c,int t,const void *bytes,size_t n){
  struct end *e=c->end+t;
  if(e->queued+n>e->room){
    unsigned char *grown=realloc(e->queue,e->queued+n);
    if(!grown){channel_end(c,ENOMEM,-1);return;}
    e->queue=grown;e->room=e->queued+n;
  }
  memcpy(e->queue+e->queued,bytes,n);e->queued+=n;
  end_drain(c,t);
}
/* each socket end read while the channel is open, it has not ended, and the other end takes what it gives at once */
static void channel_pump(struct channel *c){
  for(int s=0;s<2;s++){
    struct end *e=c->end+s,*o=c->end+(1-s);
    if(e->kind!=END_SOCKET || e->ended || (c->pid && s==c->child && c->drained))continue;
    const int want=c->open && !o->queued && (o->kind!=END_PORT || o->credit>0);
    if(want!=e->reading){e->reading=want;fd_watch(e->fd,EVFILT_READ,want,channel_udata(c,s));}
  }
}
static void end_read(struct channel *c,int s){
  struct end *e=c->end+s,*o=c->end+(1-s);
  if(e->ended)return;
  size_t room=o->kind==END_PORT?MIN(o->credit,MESH_PACKET):MESH_PACKET;
  if(!room || o->queued){channel_pump(c);return;}
  unsigned char block[MESH_PACKET];
  ssize_t got=recv(e->fd,block,room,0);
  if(got<0 && (errno==EAGAIN || errno==EINTR))return;
  if(got<0){channel_end(c,errno,-1);return;}
  if(!got){
    e->reading=0;fd_watch(e->fd,EVFILT_READ,0,channel_udata(c,s));
    if(c->pid && s==c->child){c->drained=1;child_ended(c);return;}
    e->ended=1;end_drain(c,1-s);return;
  }
  end_put(c,1-s,block,(size_t)got);
}
/* #holds: the bridge's own service: one line, `<name> <the unix second it ends> <why>`, written to
   ~/.cache/mesh/hold/<name> (which the node's convergence reads: README.md Convergence), or `<name>` alone, that file
   removed; answered with a line saying which */
static void hold_done(struct channel *c){
  char line[MESH_HOLD_BYTES+1],answer[160],dir[PATH_MAX],path[PATH_MAX],staged[PATH_MAX];
  size_t n=c->held_bytes;
  memcpy(line,c->held,n);line[n]=0;
  while(n && (line[n-1]=='\n' || line[n-1]=='\r'))line[--n]=0;
  char *rest=strchr(line,' ');
  if(rest)*rest++=0;
  const struct passwd *user=getpwuid(getuid());
  size_t name=strlen(line);
  int valid=name>0 && name<=64 && user && line[0]!='.';
  for(size_t i=0;valid && i<name;i++)valid=isalnum((unsigned char)line[i]) || line[i]=='-' || line[i]=='_' || line[i]=='.';
  int error=0;
  if(!valid)error=EINVAL;
  else{
    snprintf(dir,sizeof dir,"%s/.cache/mesh",user->pw_dir);mkdir(dir,0755);
    snprintf(dir,sizeof dir,"%s/.cache/mesh/hold",user->pw_dir);mkdir(dir,0755);
    snprintf(path,sizeof path,"%s/%s",dir,line);
    if(!rest || !*rest){if(unlink(path) && errno!=ENOENT)error=errno;}
    else{
      snprintf(staged,sizeof staged,"%s/.%s.%u",dir,line,(unsigned)getpid());
      FILE *f=fopen(staged,"w");
      if(!f)error=errno;
      else{
        const int wrote=fprintf(f,"%s\n",rest)>=0;
        if(fclose(f) || !wrote)error=errno?errno:EIO;
        else if(rename(staged,path))error=errno;
        if(error)unlink(staged);
      }
    }
  }
  if(error)snprintf(answer,sizeof answer,"hold %s: %s\n",valid?line:"(a name of 1 to 64 letters, digits, '-', '_' or '.')",strerror(error));
  else snprintf(answer,sizeof answer,"%s %s\n",rest && *rest?"held":"released",line);
  say("hold %s from node %u: %s",valid?line:"?",c->origin,answer);
  struct end *own=c->end[0].kind==END_OWN?c->end:c->end+1;
  own->ended=1;
  const int t=own==c->end?1:0;
  end_put(c,t,answer,strlen(answer));
}
/* #admission: a request for `bytes` outside the window admitted while the node's available memory, read now, holds
   them; a refusal logged */
static int64_t admitted(const struct channel *c,uint64_t bytes){
  const uint64_t room=bytes?available_now():0;
  if(!bytes || room>=bytes)return 0;
  say("admission: node %u's request to %s for %.2f GiB refused: %.2f GiB available, pressure %u\n",c->origin,c->service,
      (double)bytes/1073741824.0,(double)room/1073741824.0,health.pressure);
  return ENOMEM;
}
/* a channel's far end opened here: the service's socket connected, or the bridge's own */
static int64_t service_open(struct channel *c,int s,uint64_t bytes){
  struct end *e=c->end+s;
  const int64_t refused=admitted(c,bytes);
  if(refused)return refused;
  if(!strcmp(c->service,"hold")){e->kind=END_OWN;return 0;}
  if(!strcmp(c->service,"exec")){
    if(!exec_served)return EACCES;
    e->kind=END_EXEC;return 0;
  }
  if(!strcmp(c->service,"nodeinfo")){
    static const uint16_t served[]={8100,8099};
    int64_t error=ECONNREFUSED;
    for(size_t i=0;i<sizeof served/sizeof *served;i++){
      struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons(served[i]),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
      int fd=socket(AF_INET6,SOCK_STREAM,0),on=1,bound=1;
      if(fd<0)return errno;
      if(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) || setsockopt(fd,IPPROTO_TCP,TCP_CONNECTIONTIMEOUT,&bound,sizeof bound) ||
         connect(fd,(struct sockaddr *)&to,sizeof to) || fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK)){error=errno;close(fd);continue;}
      e->kind=END_SOCKET;e->fd=fd;
      return 0;
    }
    return error;
  }
  for(uint32_t i=0;i<MESH_SERVICES;i++)if(servings[i].fd>=0 && !strcmp(servings[i].service,c->service)){
    struct sockaddr_un to={.sun_family=AF_UNIX};
    snprintf(to.sun_path,sizeof to.sun_path,"%s",servings[i].path);
    int fd=socket(AF_UNIX,SOCK_STREAM,0),on=1;
    if(fd<0)return errno;
    if(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) || connect(fd,(struct sockaddr *)&to,sizeof to) ||
       fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK)){const int error=errno;close(fd);return error;}
    e->kind=END_SOCKET;e->fd=fd;
    return 0;
  }
  return ENOENT;
}
/* #exec: the bridge's own service `exec`, where its configuration serves it: the request's first line a command run by
   the user's shell (-c) in its home, its process group its own and every descriptor but its standard ones closed
   (POSIX_SPAWN_CLOEXEC_DEFAULT: no device or socket of the bridge's), the rest of the request its standard input, and
   the answer its standard output and error, then its wait status (int32), once both have ended; a channel that fails
   while it runs sends its group SIGTERM */
static int64_t child_spawn(struct channel *c,int s){
  struct end *e=c->end+s;
  int pair[2],on=1;
  const struct passwd *user=getpwuid(getuid());
  if(!user)return ENOENT;
  if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))return errno;
  c->held=c->held?c->held:malloc(1);
  size_t n=c->held_bytes;
  while(n && (c->held[n-1]=='\n' || c->held[n-1]=='\r'))n--;
  c->held[n]=0;
  char *shell=user->pw_shell && *user->pw_shell?user->pw_shell:"/bin/zsh",temp[PATH_MAX]="/tmp";
  confstr(_CS_DARWIN_USER_TEMP_DIR,temp,sizeof temp);
  char home[PATH_MAX+8],name[256],login[256],shelled[PATH_MAX+8],tmpdir[PATH_MAX+8];
  snprintf(home,sizeof home,"HOME=%s",user->pw_dir);snprintf(name,sizeof name,"USER=%s",user->pw_name);
  snprintf(login,sizeof login,"LOGNAME=%s",user->pw_name);snprintf(shelled,sizeof shelled,"SHELL=%s",shell);
  snprintf(tmpdir,sizeof tmpdir,"TMPDIR=%s",temp);
  char *argv[]={shell,"-c",(char *)c->held,NULL};
  char *envp[]={home,name,login,shelled,tmpdir,"PATH=/usr/bin:/bin:/usr/sbin:/sbin","LANG=en_US.UTF-8",NULL};
  posix_spawn_file_actions_t actions;posix_spawnattr_t attributes;
  sigset_t every,none;sigfillset(&every);sigemptyset(&none);
  posix_spawn_file_actions_init(&actions);posix_spawnattr_init(&attributes);
  for(int k=0;k<3;k++)posix_spawn_file_actions_adddup2(&actions,pair[1],k);
  posix_spawn_file_actions_addchdir(&actions,user->pw_dir);
  posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP|POSIX_SPAWN_SETSIGDEF|POSIX_SPAWN_SETSIGMASK);
  posix_spawnattr_setpgroup(&attributes,0);posix_spawnattr_setsigdefault(&attributes,&every);posix_spawnattr_setsigmask(&attributes,&none);
  pid_t pid=0;
  const int error=posix_spawn(&pid,shell,&actions,&attributes,argv,envp);
  posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
  close(pair[1]);
  if(error){close(pair[0]);return error;}
  if(fcntl(pair[0],F_SETFL,fcntl(pair[0],F_GETFL)|O_NONBLOCK) || setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    kill(-pid,SIGTERM);close(pair[0]);c->pid=pid;return errno?errno:EIO;
  }
  e->kind=END_SOCKET;e->fd=pair[0];c->pid=pid;c->child=s;
  struct kevent64_s watch;EV_SET64(&watch,(uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,UINT64_MAX,0,0);
  kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  say("exec from node %u, pid %d: %.200s\n",c->origin,pid,c->held);
  channel_pump(c);
  return 0;
}
/* a child's output ended and its process exited: its wait status sent after its output, and its end ended */
static void child_ended(struct channel *c){
  if(!c->exited){
    int status=0;
    const pid_t done=waitpid(c->pid,&status,WNOHANG);
    if(done==c->pid){c->exited=1;c->status=status;}
  }
  if(!c->exited || !c->drained || c->reported)return;
  int32_t status=(int32_t)c->status;
  c->reported=1;c->end[c->child].ended=1;
  end_put(c,1-c->child,&status,sizeof status);
}
/* a process exited: an exec's child whose channel awaits it, or one whose channel ended, reaped */
void child_heard(pid_t pid){
  for(uint32_t i=0;i<MESH_CHANNELS;i++){
    if(orphans[i]==pid && waitpid(pid,NULL,WNOHANG)==pid)orphans[i]=0;
    if(channels[i].used && channels[i].pid==pid && !channels[i].exited)child_ended(channels+i);
  }
}
/* a client's request read whole: a channel opened toward its node's service (here, or its CHANNEL_OPEN sent to the next hop),
   a service registered, or a ledger cut; any other kind refused (EPROTO) */
static void asked(struct asking *a){
  struct mesh_request *r=&a->request;
  const int fd=a->fd;a->fd=-1;
  fd_watch(fd,EVFILT_READ,0,0);
  r->service[sizeof r->service-1]=0;r->path[sizeof r->path-1]=0;
  if(r->magic==MESH_REQUEST_MAGIC && r->kind==MESH_REQUEST_LEDGER){
    fd_answer(fd,r->path[0]=='/'?ledger_cut(r->bytes,r->path):EINVAL);close(fd);
    return;
  }
  if(r->magic!=MESH_REQUEST_MAGIC || !r->service[0] || (r->kind!=MESH_REQUEST_OPEN && r->kind!=MESH_REQUEST_SERVE)){fd_answer(fd,EPROTO);close(fd);return;}
  if(r->kind==MESH_REQUEST_SERVE){
    struct serving *at=NULL;
    for(uint32_t i=0;i<MESH_SERVICES;i++)if(servings[i].fd>=0 && !strcmp(servings[i].service,r->service)){close(servings[i].fd);servings[i].fd=-1;}
    for(uint32_t i=0;!at && i<MESH_SERVICES;i++)if(servings[i].fd<0)at=servings+i;
    if(r->path[0]!='/' || !strcmp(r->service,"hold") || !at){fd_answer(fd,at?EINVAL:ENOSPC);close(fd);return;}
    at->fd=fd;memcpy(at->service,r->service,sizeof at->service);memcpy(at->path,r->path,sizeof at->path);
    fd_answer(fd,0);
    fd_watch(fd,EVFILT_READ,1,TAG_SERVING|(uint64_t)(at-servings));
    say("service %s: served at %s\n",at->service,at->path);
    return;
  }
  struct channel *c=channel_new();
  if(!c){fd_answer(fd,ENOSPC);close(fd);return;}
  c->key=(uint64_t)M->node<<32|++channel_sequence;c->origin=M->node;c->destination=r->node;
  memcpy(c->service,r->service,sizeof c->service);
  c->end[0].kind=END_SOCKET;c->end[0].fd=fd;
  if(r->node==M->node){
    const int64_t error=service_open(c,1,r->bytes);
    if(error){channel_end(c,error,-1);return;}
    c->open=1;fd_answer(fd,0);channel_pump(c);
    return;
  }
  struct port *hop=channel_hop(r->node);
  if(!hop){channel_end(c,EHOSTUNREACH,-1);return;}
  c->end[1].kind=END_PORT;c->end[1].port=hop;c->deadline=now+MESH_PAIR_NS;
  channel_say(c,1,CHANNEL_OPEN,MESH_WINDOW,(int64_t)r->bytes,c->service,(uint32_t)sizeof c->service);
}
static void asking_heard(struct asking *a){
  ssize_t got=recv(a->fd,(char *)&a->request+a->got,sizeof a->request-a->got,0);
  if(got<0 && (errno==EAGAIN || errno==EINTR))return;
  if(got<=0){close(a->fd);a->fd=-1;return;}
  a->got+=(size_t)got;
  if(a->got==sizeof a->request)asked(a);
}
void asking_accepted(int fd){
  int on=1;
  for(uint32_t i=0;i<MESH_CHANNELS;i++)if(askings[i].fd<0){
    if(fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK) || setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) ||
       setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on))break;
    askings[i]=(struct asking){.fd=fd,.deadline=now+MESH_ASKING_NS};
    fd_watch(fd,EVFILT_READ,1,TAG_ASKING|i);
    return;
  }
  close(fd);
}
/* a channel's message from a port: an CHANNEL_OPEN toward this node's service or, through it, the next hop's; then its answer
   (CHANNEL_CONFIRM or CHANNEL_REFUSE), its CHANNEL_DATA, the credit its peer granted (CHANNEL_WINDOW), its end (CHANNEL_SHUT) or its failure (CHANNEL_CLOSE) */
void channel_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==CHANNEL_OPEN){
    struct channel *c=channel_new();
    struct message refuse={.kind=CHANNEL_REFUSE,.key=heard->key,.from=heard->from,.to=heard->to};
    if(!c){refuse.code=ENOSPC;port_say(port,refuse,NULL,0);return;}
    c->key=heard->key;c->origin=heard->from;c->destination=heard->to;
    memcpy(c->service,payload,MIN(heard->bytes,sizeof c->service));c->service[sizeof c->service-1]=0;
    c->end[0].kind=END_PORT;c->end[0].port=port;c->end[0].credit=heard->count;
    if(heard->to==M->node){
      const int64_t error=service_open(c,1,(uint64_t)heard->code);
      if(error){refuse.code=error;port_say(port,refuse,NULL,0);channel_end(c,0,-1);return;}
      c->open=1;channel_say(c,0,CHANNEL_CONFIRM,MESH_WINDOW,0,NULL,0);channel_pump(c);
      return;
    }
    struct port *hop=channel_hop(heard->to);
    if(!hop || hop==port){refuse.code=EHOSTUNREACH;port_say(port,refuse,NULL,0);channel_end(c,0,-1);return;}
    c->end[1].kind=END_PORT;c->end[1].port=hop;c->deadline=now+MESH_PAIR_NS;
    channel_say(c,1,CHANNEL_OPEN,MESH_WINDOW,heard->code,c->service,(uint32_t)sizeof c->service);
    return;
  }
  struct channel *c=NULL;int s=0;
  for(uint32_t i=0;!c && i<MESH_CHANNELS;i++)if(channels[i].used && channels[i].key==heard->key)
    for(int k=0;k<2;k++)if(channels[i].end[k].kind==END_PORT && channels[i].end[k].port==port){c=channels+i;s=k;}
  if(!c){
    if(heard->kind!=CHANNEL_CLOSE && heard->kind!=CHANNEL_REFUSE)
      port_say(port,(struct message){.kind=CHANNEL_CLOSE,.key=heard->key,.from=heard->from,.to=heard->to,.code=ECONNRESET},NULL,0);
    return;
  }
  struct end *e=c->end+s;
  const int t=1-s;
  if(heard->kind==CHANNEL_CONFIRM && !c->open){
    e->credit=heard->count;c->open=1;c->deadline=0;
    if(c->end[t].kind==END_PORT)channel_say(c,t,CHANNEL_CONFIRM,MESH_WINDOW,0,NULL,0);
    else fd_answer(c->end[t].fd,0);
    channel_pump(c);
  }
  else if(heard->kind==CHANNEL_REFUSE && !c->open){
    if(c->end[t].kind==END_PORT)channel_say(c,t,CHANNEL_REFUSE,0,heard->code,NULL,0);
    c->end[t].kind==END_PORT?channel_end(c,0,-1):channel_end(c,heard->code?heard->code:ECONNREFUSED,s);
  }
  else if(heard->kind==CHANNEL_DATA && heard->bytes)end_put(c,t,payload,heard->bytes);
  else if(heard->kind==CHANNEL_WINDOW){e->credit+=heard->count;end_drain(c,s);}
  else if(heard->kind==CHANNEL_SHUT){e->ended=1;end_drain(c,t);}
  else if(heard->kind==CHANNEL_CLOSE)channel_end(c,heard->code?heard->code:ECONNRESET,s);
}
/* a request socket's event: a client's request read, a channel's socket end read or written, a registration's
   connection closed */
void request_event(const struct kevent64_s *e){
  if(e->udata&TAG_ASKING){
    struct asking *a=askings+(e->udata&(MESH_CHANNELS-1));
    if(a->fd==(int)e->ident)asking_heard(a);
  }else if(e->udata&TAG_SERVING){
    struct serving *v=servings+(e->udata&(MESH_SERVICES-1));
    char drop[256];
    if(v->fd!=(int)e->ident)return;
    ssize_t got=recv(v->fd,drop,sizeof drop,0);
    if(got==0 || (got<0 && errno!=EAGAIN && errno!=EINTR)){say("service %s: no longer served\n",v->service);close(v->fd);v->fd=-1;}
  }else{
    struct channel *c=channels+((e->udata&~TAG_CHANNEL)>>1);
    const int s=(int)(e->udata&1);
    if(!c->used || c->end[s].kind!=END_SOCKET || c->end[s].fd!=(int)e->ident)return;
    if(e->filter==EVFILT_WRITE)end_drain(c,s);
    else end_read(c,s);
  }
}
/* the requests' turn: a channel unanswered past the pairing window fails (ETIMEDOUT), and a client that has not sent its
   request within MESH_ASKING_NS is closed */
void requests_turn(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++){
    struct channel *c=channels+i;
    if(c->used && !c->open && c->deadline && now>c->deadline){
      const int up=c->end[0].kind==END_PORT;
      if(up)channel_say(c,0,CHANNEL_REFUSE,0,ETIMEDOUT,NULL,0);
      channel_end(c,ETIMEDOUT,up?0:-1);
    }
    if(askings[i].fd>=0 && now>askings[i].deadline){close(askings[i].fd);askings[i].fd=-1;}
    if(orphans[i] && waitpid(orphans[i],NULL,WNOHANG)==orphans[i])orphans[i]=0;
    if(c->used && c->pid && !c->exited)child_ended(c);
    /* a socket end that has ended its input and since been reset (a client's abort: its close with SO_LINGER 0)
       fails its channel */
    for(int k=0;c->used && k<2;k++)if(c->end[k].kind==END_SOCKET && c->end[k].ended && c->end[k].fd>=0){
      int error=0;socklen_t size=sizeof error;
      if(!getsockopt(c->end[k].fd,SOL_SOCKET,SO_ERROR,&error,&size) && error){channel_end(c,error,-1);break;}
    }
  }
}
/* a port lost: every channel with an end on it fails, each other end told (#requests) */
void requests_lost(struct port *port,int64_t code){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)for(int s=0;s<2;s++)
    if(channels[i].used && channels[i].end[s].kind==END_PORT && channels[i].end[s].port==port){channel_end(channels+i,code?code:ENETRESET,s);break;}
}
void requests_init(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)askings[i].fd=-1;
  for(uint32_t i=0;i<MESH_SERVICES;i++)servings[i].fd=-1;
}
void requests_stop(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++){
    if(channels[i].used)channel_end(channels+i,ESHUTDOWN,-1);
    if(askings[i].fd>=0){close(askings[i].fd);askings[i].fd=-1;}
  }
  for(uint32_t i=0;i<MESH_SERVICES;i++)if(servings[i].fd>=0){close(servings[i].fd);servings[i].fd=-1;}
}

/* a port's link closed to be paired afresh would end every channel with an end on it: whether one has (link_idle) */
int requests_held(const struct port *port){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)for(int s=0;s<2;s++)
    if(channels[i].used && channels[i].end[s].kind==END_PORT && channels[i].end[s].port==port)return 1;
  return 0;
}

/* design/bridge-and-ipc.md#stop: the bridge stopping: every program's words toward its destinations cancelled (~0) and
   its status the stop's code, stored from the event loop's thread as plain memory stores, no queue, thread or verbs
   call waited on */
void programs_stop(int64_t code){
  for(struct flow *f=flows;f;f=f->next){
    if(!owned(f->owner))continue;
    if(f->cancel)mesh_cancel(M,f->cancel,f->destination);
    status_failed(&mesh_client_at(M,f->owner)->status[f->destination],f->owner,code,4,M->node);
  }
}
/* design/bridge-and-ipc.md#clients: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once, and on a
   liveness turn any client whose process exited retired; a program that ended (its slot given back or retired) has its
   flows stopped; a program started is taken toward each destination it binds, once no flow of the slot's earlier client
   remains.  Rows of a retired client are released once no flow of one remains and no stripe is in flight */
void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0,zombie=0;
  for(struct flow *f=flows;f;f=f->next)
    if(!owned(f->owner) || atomic_load(&mesh_client_at(M,f->owner)->configured)!=f->owner)flow_end(f);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(liveness && owner && !(owner&MESH_RETIRING) && mesh_gone((pid_t)(uint32_t)owner)){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING)){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    }
    changed|=owner!=owners[s];owners[s]=owner;
    if(!owner || (owner&MESH_RETIRING) || atomic_load_explicit(&client->configured,memory_order_acquire)!=owner || accepted[s]==owner)continue;
    int earlier=0;
    for(struct flow *f=flows;f;f=f->next)earlier|=f->owner!=owner && ((f->owner>>32)&(MESH_CLIENTS-1))==s;
    if(earlier)continue;
    accepted[s]=owner;
    atomic_store_explicit(&client->ports,0,memory_order_relaxed);
    for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)if(mesh_destination_transfers(M,owner,d))program_take(client,owner,d);
  }
  for(struct flow *f=flows;f;f=f->next)zombie|=!owned(f->owner);
  /* design/bridge-dataflow.md#stripes: a stripe in flight may name a retired client's bytes and word */
  for(uint32_t p=0;p<port_count;p++)zombie|=atomic_load_explicit(&ports[p].stripes.flight,memory_order_acquire)!=0;
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}
/* design/bridge-and-ipc.md#flows: every flow's turn: its job's end taken, a stopped flow freed, and a progress thread's
   finding (a post or a completion that failed) its failure; nothing here waits on a clock */
void flows_turn(void){
  for(struct flow *f=flows,*next;f;f=next){
    next=f->next;
    if(f->job && atomic_load_explicit(&f->done,memory_order_acquire))flow_done(f);
    if(f->stopping){
      if(!f->job && atomic_load_explicit(&f->stopped,memory_order_acquire))flow_free(f);
      continue;
    }
    if(f->ended)continue;
    const int64_t failed=atomic_load(&f->failed);
    if(failed)flow_fail(f,failed,atomic_load(&f->domain),M->node);
  }
}
