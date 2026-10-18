#include "mesh-bridge.h"

static void busy_tell(void){
  const uint64_t busy=(uint64_t)flow_count+await_count;
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
static struct mesh_tx *side_tx(struct flow *f,int d){ return &mesh_client_at(M,f->side[d].owner)->tx[f->side[d].destination]; }
static uint32_t side_channel(struct flow *f,int d,uint32_t k){ return f->side[d].destination*MESH_QUEUES+k; }
static uint32_t side_length(struct flow *f,int d,uint32_t k){ return f->side[d].owner?atomic_load(mesh_order_length(M,f->side[d].owner,side_channel(f,d,k),d)):0; }
static struct mesh_transfer *side_transfers(struct flow *f,int d,uint32_t k){ return mesh_transfers(M,f->side[d].owner,side_channel(f,d,k),d); }
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
/* the program whose status and cancellation words a flow tells */
static void flow_program(struct flow *f,uint64_t owner,uint32_t destination){
  uint64_t cancel=mesh_client_at(M,owner)->tx[destination].cancel;
  f->owner=owner;f->destination=destination;f->cancel=cancel?(void *)((char *)M+cancel):NULL;
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
  *span=wire_span(link->provider.device,(uint64_t)m->block*m->pgsz,offset,remaining);
  return 0;
}

/* design/prepared-machine.md#M07, #M08, #M12: the word a landed chunk of invocation u of the transfer's slot `index` (0
   where each slot has its own rows) stores: its slot's completion word (mesh_host_inputs, the Metal transport), where
   the row's delivery names a word among the receiving program's own toward the flow's node (`words`, its cancellation
   range, [offset, offset + 8 count): a delivery another program wrote on the row, which outlives that program, is
   none of this one's); or, on a row the receiving program (`owner`) also publishes (a relay's rows), that slot's
   publication's SEND cell once the record's last chunk landed, so the landing releases it to the next hop (the index
   hand-off, *release set); else the record's own (`sink`), which nothing reads.  NULL where the word is not in the
   window (#M09) */
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
/* a flow's ledger, made on its port's queue before either side is prepared or the flow listed, where the configuration
   keeps them; a flow whose ledger cannot be allocated runs without one, logged */
static void ledger_open(struct flow *f){
  if(!ledger_kept || f->ledger)return;
  struct ledger *l=calloc(1,sizeof *l);
  uint32_t *words=l?malloc(3*(size_t)f->qps*sizeof *words):NULL;
  if(!words){
    free(l);
    say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger: %s\n",f->port->index,(unsigned long long)f->key,
        f->far,f->stripe,f->stripes,strerror(ENOMEM));
    return;
  }
  for(size_t i=0;i<3*(size_t)f->qps;i++)words[i]=LEDGER_NONE;
  l->id=atomic_fetch_add(&ledger_ids,1)+1;l->queues=(uint32_t)f->qps;l->started=ledger_ns();
  l->current=words;l->pending=words+f->qps;l->last=words+2*(size_t)f->qps;
  f->ledger=l;
}
/* the events a side's ledger holds: one pass of the flow's `once` cells or records, every pass of a cyclic program's
   the bound's latest (a ring: event e at e % room, its count published as each is stored, so a cut (ledger_cut) reads
   them while the flow runs) */
static uint32_t ledger_room(uint64_t once,int cyclic){ return cyclic || once>MESH_LEDGER_EVENTS?MESH_LEDGER_EVENTS:(uint32_t)once; }
/* design/bridge-and-ipc.md#ledger: taken by a cut and by a ledger's release, so a cut never reads a ledger freed */
static pthread_mutex_t ledger_lock=PTHREAD_MUTEX_INITIALIZER;
/* design/bridge-and-ipc.md#ledger: a flow's ledger written at its release, on its port's queue once neither progress
   thread reads the flow: its header into the bridge's log, then a line for each cell seen and each record landed, which
   name the cell or record by queue, invocation, position, binding and cycle, as the peer's ledger of the same flow
   names the record or cell (rdma/ledger.py pairs them): into the log where there are MESH_LEDGER_LINES or fewer, else
   into a file beside the log named by the program's key, the node, its peer, the stripe, the ledger and the time, which
   the header names; then freed */
static int ledger_cell_order(const void *a,const void *b){
  const uint64_t x=((const struct ledger_cell *)a)->cell,y=((const struct ledger_cell *)b)->cell;
  return (x>y)-(x<y);
}
/* the ledger's header line, `cut` the events past the latest cut */
static void ledger_header(const struct flow *f,const struct ledger *l,uint32_t sends,uint32_t landings,const char *file,char *line,size_t room){
  snprintf(line,room,"{\"ledger\":%u,\"key\":\"%016llx\",\"node\":%u,\"peer\":%u,\"port\":%u,\"near\":%u,\"far\":%u,\"stripe\":%u,"
           "\"stripes\":%u,\"exclusive\":1,\"queues\":%u,\"qps\":%d,\"invocations\":%u,\"cyclic\":%u,\"started\":%llu,\"sends\":%u,\"send_room\":%u,"
           "\"landings\":%u,\"landing_room\":%u,\"cut\":[%u,%u],\"file\":%s%s%s}\n",l->id,(unsigned long long)f->key,M->node,f->port->peer,
           f->port->index,f->near,f->far,f->stripe,f->stripes,f->queues,f->qps,l->invocations,l->cyclic,(unsigned long long)l->started,
           sends,l->send_room,landings,l->landing_room,l->cut_sends,l->cut_landings,file?"\"":"",file?file:"null",file?"\"":"");
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
/* design/bridge-and-ipc.md#ledger: a flow's ledger written at its release, on its port's queue once neither progress
   thread reads the flow: its header into the bridge's log, then its events past its latest cut: into the log where
   there are MESH_LEDGER_LINES or fewer, else into a file beside the log named by the program's key, the node, its peer,
   the stripe, the ledger and the time, which the header names; then freed */
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
    if(f->key!=key || !l || f->state!=RUNNING)continue;
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
   ring slot and its stream successor, a stream a ring (the cells' layout is mesh_transfers_prepare's), and its gate a
   queue pair.  A stream's cells run
   invocation by invocation, the publications of each in binding order, as the peer's ring posts its receives: TN3205's
   credit flow control holds a SEND until its matching receive is posted (metal-microbench docs/kernels.md Transport
   (git 7300d1e^)).  0, or the error (EINVAL: a transfer of more slots than the program's). */
static int sends_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_SEND].owner);
  struct mesh_tx *tx=side_tx(link,MESH_SEND);
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire);
  int cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0,qps=link->qps;
  size_t requests=0,cells=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    for(uint32_t i=0;i<side_length(link,MESH_SEND,k);i++){
      if(out[i].count>tx->slots || (uint64_t)k*tx->slots+out[i].count>(uint64_t)qps)return EINVAL;
      const size_t own=(size_t)out[i].count*stripe_count(link,out[i].begin,mesh_transfer_active(out+i,invocations));
      requests+=(size_t)(mesh_row_chunks(m,out[i].local_row,out[i].bytes)-1)*own;
      cells+=own;
    }
  }
  link->publications=(void *)((char *)m+tx->cells);
  /* design/prepared-machine.md#M29 */
  link->requests=aligned_alloc(128,(requests+1)*sizeof *link->requests);
  link->streams=calloc((size_t)qps,sizeof *link->streams);
  link->gates=aligned_alloc(64,(size_t)qps*sizeof *link->gates);
  struct mesh_send **last=calloc((size_t)qps,sizeof *last),**terminal=calloc((size_t)qps,sizeof *terminal);
  if(!link->requests||!link->streams||!link->gates||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  link->request_count=requests;
  /* #ledger: the cells' names, each stream's position at its latest invocation (`positions`, a slot's pair), and room for
     the events of one pass, a cyclic program's of every pass up to the bound */
  struct ledger *ledger=link->ledger;
  uint32_t *positions=NULL;
  if(ledger){
    ledger->invocations=invocations;ledger->cyclic=(uint32_t)cyclic;
    ledger->cell=malloc((cells+1)*sizeof *ledger->cell);
    ledger->send=malloc(((size_t)ledger_room(cells,cyclic)+1)*sizeof *ledger->send);
    positions=malloc((2*(size_t)tx->slots+2)*sizeof *positions);
    if(!ledger->cell || !ledger->send || !positions){
      free(ledger->cell);free(ledger->send);free(positions);ledger->cell=NULL;ledger->send=NULL;positions=NULL;ledger=NULL;
      say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger of its SENDs: %s\n",link->port->index,
          (unsigned long long)link->key,link->far,link->stripe,link->stripes,strerror(ENOMEM));
    }else ledger->send_room=ledger_room(cells,cyclic);
  }
  /* a request is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     SEND completion queue its pairs share */
  const uint32_t sends=(uint32_t)link->provider.sent->cqe/(uint32_t)qps;
  for(int q=0;q<qps;q++)link->gates[q]=(struct send_gate){.capacity=MIN(link->provider.queues[q].send_capacity,sends)};
  size_t next=0;int error=0;
  for(uint32_t k=0;k<link->queues && !error;k++){
    uint32_t count=side_length(link,MESH_SEND,k);
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count && !error;slot++){
        uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes),stream=k*tx->slots+slot;
        struct ibv_qp *pair=link->provider.queues[stream].pair;
        struct mesh_send *cells=link->publications+out[i].first+(size_t)slot*column;
        for(uint32_t u=0;u<active && !error;u++){
          if(!stripe_of(link,out[i].begin+u))continue;
          struct mesh_send *cell=cells+u;
          cell->pair=(uintptr_t)pair;cell->queue=stream;cell->chunks=chunks;
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
        terminal[stream]=cells+active;
      }
    }
    /* each stream's cells in invocation order, its publications in binding order within one */
    if(positions)memset(positions,0,(2*(size_t)tx->slots+2)*sizeof *positions);
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(out+i,invocations);
      if(t<out[i].begin || t-out[i].begin>=active)continue;
      for(uint32_t slot=0;slot<out[i].count;slot++){
        uint32_t stream=k*tx->slots+slot;
        struct mesh_send *cell=link->publications+out[i].first+(size_t)slot*(active+1)+(t-out[i].begin);
        if(last[stream])last[stream]->successor=(uintptr_t)cell;
        else link->streams[stream].cell=cell;
        last[stream]=cell;
        /* #ledger: the cell's name, its position among the stream's cells of invocation t */
        if(positions && ledger->cells<cells){
          if(positions[2*slot]!=t+1){positions[2*slot]=t+1;positions[2*slot+1]=0;}
          ledger->cell[ledger->cells++]=(struct ledger_cell){(uintptr_t)cell,out[i].bytes,stream,t,positions[2*slot+1]++,out[i].binding,
            mesh_row_chunks(m,out[i].local_row+slot*out[i].stride,out[i].bytes)};
        }
      }
    }
  }
  /* design/prepared-machine.md#M30: a cyclic program's last cell is followed by its first */
  for(int q=0;q<qps && !error;q++)if(last[q])last[q]->successor=cyclic?(uintptr_t)link->streams[q].cell:(uintptr_t)terminal[q];
  for(int q=0;q<qps;q++){link->streams[q].first=link->streams[q].cell;link->streams[q].cycle=0;}
  link->stream_count=0;
  for(int q=0;q<qps;q++)if(link->streams[q].cell)link->streams[link->stream_count++]=link->streams[q];
  free(last);free(terminal);free(positions);
  return error;
}

/* design/prepared-machine.md#M08 */
/* The receive side: each ring's records invocation by invocation (the flow's invocations, #cables), the transfers of
   each in binding order, a transfer's chunks in order, each storing its word (receive_input).  A ring is its queue
   pair's, its first records posted at once and each record posted once the record of its transfer `span` of the flow's
   own invocations before has landed (the ring's span over the program's invocations, divided among the stripes; a
   cyclic ring's first records wait on its last of the cycle before), so a ring's posted frames stay within its queue
   pair's.  0, or the error (EINVAL: a transfer of more slots than the program's, a word outside the window). */
static int receives_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_RECEIVE].owner);
  /* design/prepared-machine.md#M09, #M10: the program's targets lie in the window, as each word it stores does */
  if(client->target_count && (client->targets<m->data_off || client->targets>m->length ||
     client->target_count>(m->length-client->targets)/sizeof(struct mesh_target)))return EINVAL;
  struct mesh_tx *tx=side_tx(link,MESH_RECEIVE);
  /* design/prepared-machine.md#M12: the program's own words toward the flow's node, where it has any */
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)link->side[MESH_RECEIVE].destination+1);
  const struct mesh_cancellation *cancel=tx->cancel>=m->data_off && tx->cancel<=m->length-range?(const void *)((char *)m+tx->cancel):NULL;
  const struct mesh_cancel_range *words=cancel?cancel->ranges+link->side[MESH_RECEIVE].destination:NULL;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  int qps=link->qps;
  size_t records=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *in=side_transfers(link,MESH_RECEIVE,k);
    for(uint32_t i=0;i<side_length(link,MESH_RECEIVE,k);i++){
      if(in[i].count>tx->slots || (uint64_t)k*tx->slots+in[i].count>(uint64_t)qps)return EINVAL;
      records+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count*stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations));
    }
  }
  link->receive=aligned_alloc(128,(records+1)*sizeof *link->receive);
  link->rings=calloc((size_t)qps,sizeof *link->rings);
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
  /* a record is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     receive completion queue its pairs share */
  const uint32_t receives=(uint32_t)link->provider.completion->cqe/(uint32_t)qps;
  for(int q=0;q<qps;q++)link->rings[q]=(struct receive_ring){.completed=-1,.span=UINT32_MAX,
    .capacity=MIN(link->provider.queues[q].receive_capacity,receives)};
  size_t at=0;int error=0;
  for(int ring=0;ring<qps && !error;ring++){
    uint32_t k=(uint32_t)ring/tx->slots,slot=(uint32_t)ring%tx->slots,count=k<link->queues?side_length(link,MESH_RECEIVE,k):0;
    struct mesh_transfer *in=count?side_transfers(link,MESH_RECEIVE,k):NULL;
    struct ibv_qp *pair=link->provider.queues[ring].pair;
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    /* firsts: the ring index of each transfer's own records, own invocation by own invocation (base[i] the first's) */
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)
      base[i+1]=base[i]+(slot<in[i].count?stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations)):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
    }
    uint32_t position=0,latest=UINT32_MAX;
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      if(latest!=t){latest=t;position=0;}
      uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(m,row,in[i].bytes);
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
          .span=span,.input=receive_input(m,link->side[MESH_RECEIVE].owner,words,row,in[i].stride?0:slot,c,chunks,u,&record->sink,&release),
          .pair=pair,.argument=1,.frames=(span.length+4095)/4096,.invocation=t,.queue=(uint32_t)ring,.link=link};
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
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row+slot*in[i].stride,in[i].bytes);
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
    if(!error)error=ring_advance(link,r);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/bridge-and-ipc.md#flows: a flow's verbs work, on its port's queue: its queue pairs made; ready to receive, its
   receives prepared and posted; its SENDs prepared, its queue pairs ready to send, and served by its port's progress
   threads; or a side whose lists arrived, the flow's SENDs unserved while it is prepared and served again where they
   were.  0, or the error */
static int flow_work(struct flow *f,int job){
  int error=0;
  if(job==JOB_PAIR)return pair_up(&f->provider,f->qps);
  ledger_open(f);
  const int ran=atomic_load_explicit(&f->progressing,memory_order_acquire);
  if(job==JOB_CHANGE)port_quiet(f);
  if(job==JOB_RECEIVE)error=pair_receive(&f->provider,(const uint32_t (*)[2])f->pairs,f->port->heard.lid,f->port->heard.gid);
  if(!error && job!=JOB_RUN && f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE) && !(error=receives_configure(f)))
    f->configured|=1u<<MESH_RECEIVE;
  if(!error && job!=JOB_RECEIVE && f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && !(error=sends_configure(f)))
    f->configured|=1u<<MESH_SEND;
  if(!error && job==JOB_RUN)error=pair_send(&f->provider);
  if(!error && (job==JOB_RUN || (job==JOB_CHANGE && ran)))error=port_list(f);
  return error;
}
static void flow_job(struct flow *f,int job){
  f->job=job;atomic_store_explicit(&f->done,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    /* design/bridge-and-ipc.md#lanes: guarded (port_guard) */
    const int raised=guard_raise(f->port);
    f->result=flow_work(f,job);
    guard_lower(f->port,raised);
    port_pairs(f->port);
    atomic_store_explicit(&f->done,1,memory_order_release);
    wake();
  });
}

/* design/algorithm-sources.md#meshresult: a flow out of its port's table and its queue pairs destroyed, on its port's
   queue, the link's census printed: each ring's SEND requests and frames posted and retired by the send completions
   its queue pair took, and its records landed and their bytes, its ledger then written (#ledger).  1 once nothing is
   held (a failed destruction keeps it for the next turn) */
static int flow_release(struct flow *flow){
  port_unlist(flow);
  for(int q=0;flow->rings && q<flow->qps;q++)
    say("link %u census queue=%d requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu\n",
      flow->port->index,q,flow->gates?(unsigned long long)flow->gates[q].requests:0ull,flow->gates?(unsigned long long)flow->gates[q].posted:0ull,
      flow->gates?(unsigned long long)flow->gates[q].retired:0ull,flow->gates?(unsigned long long)flow->gates[q].completions:0ull,
      flow->rings[q].count,flow->rings[q].posted,(unsigned long long)flow->rings[q].landed,(unsigned long long)flow->rings[q].bytes);
  /* design/bridge-and-ipc.md#lanes: guarded (port_guard) */
  const int raised=guard_raise(flow->port),down=down_pair(&flow->provider);
  guard_lower(flow->port,raised);
  if(!down)return 0;
  ledger_write(flow);
  free(flow->receive);free(flow->streams);free(flow->gates);free(flow->rings);free(flow->requests);
  flow->receive=NULL;flow->streams=NULL;flow->gates=NULL;flow->rings=NULL;flow->requests=NULL;
  return 1;
}
/* A flow stopped on its port's queue, after its job: where it failed, its program's pending words toward the far end
   cancelled (M12) once its port's progress threads no longer read it, and its status the failure naming the node
   `where` */
static void flow_stop(struct flow *f,int64_t code,uint32_t domain,uint32_t where){
  const uint64_t owner=f->owner;const uint32_t destination=f->destination;struct mesh_cancellation *cancel=f->cancel;
  f->stopping=1;atomic_store_explicit(&f->stopped,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    port_unlist(f);
    if(code && owned(owner)){
      if(cancel)mesh_cancel(M,cancel,destination);
      status_failed(&mesh_client_at(M,owner)->status[destination],owner,code,domain,where);
    }
    atomic_store_explicit(&f->stopped,flow_release(f)?1:2,memory_order_release);
    port_pairs(f->port);
    wake();
  });
}
static void flow_say(struct flow *f,uint32_t kind,uint32_t count,int64_t code,const uint32_t (*pair)[2],uint32_t origin){
  struct message said={.kind=kind,.count=count,.key=f->key,.code=code,.from=f->near,.to=f->far,.failed=origin,.stripe=(uint16_t)f->stripe};
  for(uint32_t q=0;pair && q<count && q<MESH_PAIRS;q++){said.pair[q][0]=pair[q][0];said.pair[q][1]=pair[q][1];}
  port_say(f->port,said,NULL,0);
}
/* a READY: the flow's queue pairs, then its route's `hops` nodes and the transfers its SEND side sends, with its
   program's digest */
static void ready_say(struct flow *f,const uint32_t *route,uint32_t hops){
  int d=f->side[MESH_SEND].owner?MESH_SEND:MESH_RECEIVE;
  struct mesh_client *client=mesh_client_at(M,f->side[d].owner);
  struct mesh_tx *tx=side_tx(f,d);
  struct ready ready={.hops=hops,.slots=tx->slots,.invocations=tx->invocations,.depth=atomic_load(&client->depth),
    .cyclic=atomic_load(&client->cyclic),.stripe=f->stripe,.stripes=f->stripes,.exclusive=1,
    .digest=atomic_load_explicit(&client->digest,memory_order_relaxed)};
  size_t transfers=0;
  for(uint32_t k=0;k<f->queues;k++)transfers+=ready.lengths[k]=side_length(f,MESH_SEND,k);
  size_t bytes=sizeof ready+(size_t)hops*sizeof *route+transfers*sizeof(struct mesh_transfer);
  unsigned char *payload=malloc(bytes),*at=payload;
  if(!payload){port_lost(f->port,ENOMEM);return;}
  memcpy(at,&ready,sizeof ready);at+=sizeof ready;
  memcpy(at,route,(size_t)hops*sizeof *route);at+=(size_t)hops*sizeof *route;
  for(uint32_t k=0;k<f->queues;k++){memcpy(at,side_transfers(f,MESH_SEND,k),ready.lengths[k]*sizeof(struct mesh_transfer));at+=ready.lengths[k]*sizeof(struct mesh_transfer);}
  port_say(f->port,(struct message){.kind=READY,.count=(uint32_t)f->qps,.key=f->key,.from=f->near,.to=f->far,.stripe=(uint16_t)f->stripe},
           payload,(uint32_t)bytes);
  free(payload);
}
static struct await *awaits;
static struct await *await_find(uint64_t key,uint32_t far){
  for(struct await *a=awaits;a;a=a->next)if(a->key==key && a->far==far)return a;
  return NULL;
}
static struct await *await_of(uint64_t owner,uint32_t d){
  for(struct await *a=awaits;a;a=a->next)if(a->owner==owner && a->destination==d)return a;
  return NULL;
}
static void await_drop(struct await *a){
  for(struct await **at=&awaits;*at;at=&(*at)->next)if(*at==a){*at=a->next;await_count--;break;}
  free(a);
  busy_tell();
}
/* an await's stripe taken: dropped once every stripe is (1 then) */
static int await_took(struct await *a,uint32_t stripe,uint32_t stripes){
  a->stripes=stripes;a->taken|=UINT64_C(1)<<stripe;
  if((uint32_t)__builtin_popcountll(a->taken)!=a->stripes)return 0;
  await_drop(a);
  return 1;
}

/* design/algorithm-sources.md#meshresult: a program on a port stopped, its peer told (`tell`, END or FAIL naming the node
   `origin` where it happened), its status the failure naming `where`; a relay's other flow fails with it, telling its
   peer, and so does every flow of the same client program on its other cables and stripes, and its await for the
   stripes that have not come */
void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t tell,uint32_t where,uint32_t origin){
  if(f->state==FAILED)return;
  f->state=FAILED;
  if(tell)flow_say(f,tell,0,code,NULL,origin);
  say("link %u: program %016llx toward node %u (stripe %u of %u) failed at node %u: code %lld domain %u\n",f->port->index,
    (unsigned long long)f->key,f->far,f->stripe,f->stripes,where,(long long)code,domain);
  flow_stop(f,code,domain,where);
  if(f->partner)flow_fail(f->partner,code,domain,tell?tell:(uint32_t)(code==ECONNRESET?END:FAIL),origin,origin);
  for(struct flow *g=flows;f->owner && g;g=g->next)if(g->owner==f->owner && g->destination==f->destination)flow_fail(g,code,domain,FAIL,where,origin);
  struct await *a=f->owner?await_of(f->owner,f->destination):NULL;
  if(a)await_drop(a);
}
/* a client's program toward its destination `d` failed as a value naming the node `where`: its flows failed, telling
   their peers (their FAILs naming `origin`), its await dropped, and its status the failure */
static void program_end(uint64_t owner,uint32_t d,int64_t code,uint32_t where,uint32_t origin){
  for(struct flow *g=flows;g;g=g->next)if(g->owner==owner && g->destination==d)flow_fail(g,code,4,FAIL,where,origin);
  struct await *a=await_of(owner,d);
  if(a)await_drop(a);
  if(owned(owner))status_failed(&mesh_client_at(M,owner)->status[d],owner,code,4,where);
}
static void program_fail(uint64_t owner,uint32_t d,int64_t code,uint32_t where){ program_end(owner,d,code,where,M->node); }
/* design/bridge-and-ipc.md#protocol: a client's program of this node that runs: started (its slot's `configured` its
   owner, acquired before its digest is read) and not yet failed toward every destination it binds; and its digest
   (mesh.h mesh_client `digest`, 0 where it names none, or where the program does not run: a READY that meets a program
   that ended or failed is held for the execution's next program), and whether two named ones differ */
static int program_runs(uint64_t owner){
  if(!owned(owner) || atomic_load_explicit(&mesh_client_at(M,owner)->configured,memory_order_acquire)!=owner)return 0;
  const struct mesh_client *client=mesh_client_at(M,owner);
  for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)
    if(mesh_destination_transfers(M,owner,d) && atomic_load(&client->status[d].phase)!=MESH_STOPPED)return 1;
  return 0;
}
static uint64_t program_digest(uint64_t owner){
  return program_runs(owner)?atomic_load_explicit(&mesh_client_at(M,owner)->digest,memory_order_relaxed):0;
}
static int digest_apart(uint64_t mine,uint64_t theirs){ return mine && theirs && mine!=theirs; }
/* a client's program whose peer at `node` runs a program of its execution made from another digest: failed toward every
   destination it binds as a value (ESTALE naming that node), each flow telling its peer FAIL with ESTALE naming it too
   (that node's own told this one), so the peer's program fails alike (port_heard), and so on along the execution's
   programs: no member waits for a peer whose program differs */
void program_stale(uint64_t owner,uint32_t node){
  if(!program_runs(owner))return;
  struct mesh_client *client=mesh_client_at(M,owner);
  say("program %016llx: node %u's program of it is made from another digest than this node's (%016llx): failed\n",
      (unsigned long long)client->key,node,(unsigned long long)atomic_load_explicit(&client->digest,memory_order_relaxed));
  for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)
    if(mesh_destination_transfers(M,owner,d))program_end(owner,d,ESTALE,node,client->tx[d].node==node?M->node:node);
}
/* a program that ended, its client gone: its peer told (END) */
static void flow_end(struct flow *f){
  if(f->state==FAILED)return;
  f->state=FAILED;
  flow_say(f,END,0,0,NULL,M->node);
  flow_stop(f,0,0,0);
}
static void lease_free(struct lease *lease){
  mesh_retire(M,lease->context.client);
  unreleased=1;free(lease);
}
static void lease_drop(struct lease *lease){ if(lease && !--lease->users)lease_free(lease); }
static struct lease *lease_take(uint64_t key){
  struct lease *lease=calloc(1,sizeof *lease);
  if(lease){lease->context.fd=-1;if(mesh_slot_take(&lease->context,M,key)){free(lease);lease=NULL;}}
  return lease;
}
/* a flow freed, its index in its port's table and the queue pairs it held of the device's given back */
static void flow_free(struct flow *f){
  for(struct flow **at=&flows;*at;at=&(*at)->next)if(*at==f){*at=f->next;break;}
  if(f->partner)f->partner->partner=NULL;
  f->port->flows--;
  f->port->reserved[f->slot/64]&=~(UINT64_C(1)<<(f->slot%64));
  for(int d=0;d<2;d++){lease_drop(f->side[d].lease);lease_drop(f->pending[d].lease);}
  free(f);
  flow_count--;busy_tell();
}
/* design/bridge-and-ipc.md#clients: a client program's status once every flow of it toward the node runs, every stripe
   of a routed one among them (no await of it left) */
static void program_paired(struct flow *f){
  for(struct flow *g=flows;g;g=g->next)
    if(g->owner==f->owner && g->destination==f->destination && (g->state!=RUNNING || g->job))return;
  if(!owned(f->owner) || await_of(f->owner,f->destination))return;
  struct mesh_port_info *status=&mesh_client_at(M,f->owner)->status[f->destination];
  atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
  status_prepared(status,f->owner);
}

/* design/bridge-and-ipc.md#protocol: POSTED told the peer once the flow's receives are posted; a relay's once its own
   receives are and its partner heard the next hop's POSTED, so a POSTED reaches a route's end only once the far end's
   program has started and every relay between has posted (end to end, design/recovery.md#first).  Tried at each of
   the flow's job ends and as its partner hears POSTED (port_heard), whichever comes last */
void flow_tell(struct flow *f){
  if(f->told || f->job || f->state==FAILED || f->state<POSTING)return;
  if(f->partner && (!(f->partner->heard&HEARD_POSTED) || !f->side[MESH_RECEIVE].owner || !(f->configured&1u<<MESH_RECEIVE)))return;
  f->told=1;
  flow_say(f,POSTED,0,0,NULL,M->node);
}
/* design/bridge-and-ipc.md#protocol: a flow's next step, its job's end or a message heard: paired once the peer's READY
   or PAIRS is heard (its queue pairs made; PAIRS sent), ready to receive once its PAIRS are and, held,
   once its client's program took it (POSTED told, flow_tell), running once its POSTED is; a relay whose side's lists
   arrived after its receives were posted changes */
void flow_step(struct flow *f){
  if(f->state==FAILED || f->job || f->stopping)return;
  for(int d=0;d<2;d++)if(f->pending[d].owner && !f->side[d].owner){f->side[d]=f->pending[d];f->pending[d]=(struct side){0};}
  if(f->state==READIED && (f->heard&(HEARD_READY|HEARD_PAIRS)))flow_job(f,JOB_PAIR);
  else if(f->state==PAIRING && (f->heard&HEARD_PAIRS) && !f->hold)flow_job(f,JOB_RECEIVE);
  else if(f->state==POSTING && (f->heard&HEARD_POSTED))flow_job(f,JOB_RUN);
  else if((f->state==POSTING || f->state==RUNNING) && ((f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && f->state==RUNNING) ||
                                (f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE))))flow_job(f,JOB_CHANGE);
}
static void flow_done(struct flow *f){
  const int job=f->job;
  f->job=0;
  if(f->state==FAILED)return;
  if(f->result){
    if(job==JOB_PAIR && f->result==ENOSPC)
      say("link %u (%s): program %016llx toward node %u: %d queue pairs refused, the device holding %u of its max_qp %u\n",
        f->port->index,f->port->name,(unsigned long long)f->key,f->far,f->qps,f->port->device.pairs,f->port->device.max_pairs);
    flow_fail(f,f->result,1,FAIL,M->node,M->node);
    return;
  }
  if(job==JOB_PAIR){
    uint32_t pairs[MESH_PAIRS][2];
    for(int q=0;q<f->qps;q++){pairs[q][0]=f->provider.queues[q].pair->qp_num;pairs[q][1]=(f->provider.psn+(uint32_t)q)&0xffffff;}
    flow_say(f,PAIRS,(uint32_t)f->qps,0,(const uint32_t (*)[2])pairs,M->node);
    f->state=PAIRING;
    if(!f->hold)f->deadline=now+MESH_PAIR_NS;
  }else if(job==JOB_RECEIVE){
    /* design/recovery.md#first: the peer's POSTED waits on its program's start where no READY told of it (a peer that
       holds posts once its client's program takes it), and a relay's on the far end's: the peer window, and the
       pairing window after it */
    f->state=POSTING;f->deadline=(f->heard&HEARD_READY) && !f->partner?now+MESH_PAIR_NS:f->created+MESH_PEER_NS+MESH_PAIR_NS;
  }else{
    if(job==JOB_RUN)f->state=RUNNING;
    if(f->state==RUNNING)program_paired(f);
  }
  flow_tell(f);
  flow_step(f);
}

/* design/bridge-and-ipc.md#flows: a flow made on a port toward `far` for the execution `key`, the program's invocations
   t with t mod stripes = stripe (#cables), an index of the port's table its own (MESH_FLOWS a port), its queue pairs
   fitting a PAIRS message (the device refuses those past its usable ones as its job makes them, ENOSPC as a value);
   NULL where it is not one or the table is full */
static struct flow *flow_make(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t queues,uint32_t slots,uint32_t stripe,uint32_t stripes){
  if(!queues || !slots || queues>MESH_QUEUES || (uint64_t)queues*slots>MESH_PAIRS || !stripes || stripes>MESH_STRIPES || stripe>=stripes)return NULL;
  uint32_t slot=0;
  while(slot<MESH_FLOWS && port->reserved[slot/64]>>(slot%64)&1)slot++;
  if(slot==MESH_FLOWS)return NULL;
  struct flow *f=calloc(1,sizeof *f);
  if(!f)return NULL;
  const uint32_t qps=queues*slots;
  *f=(struct flow){.next=flows,.near=near,.far=far,.queues=queues,.stripe=stripe,.stripes=stripes,.qps=(int)qps,
    .slot=slot,.generation=++port->generation[slot],.key=key,.created=now,.deadline=now+MESH_PEER_NS,.M=M,.port=port,
    .provider={.device=&port->device}};
  port->reserved[slot/64]|=UINT64_C(1)<<(slot%64);
  if(slot>=atomic_load(&port->high))atomic_store(&port->high,slot+1);
  flows=f;port->flows++;flow_count++;busy_tell();
  return f;
}
struct flow *flow_on(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t stripe){
  for(struct flow *f=flows;f;f=f->next)
    if(f->port==port && f->key==key && f->near==near && f->far==far && f->stripe==stripe && f->state!=FAILED)return f;
  return NULL;
}

/* design/bridge-and-ipc.md#flows, #routes: a relayed program a READY carried, mirrored into a lease: rows the bridge
   leases for each transfer the sender sends (mesh.h mesh_relay_slot a slot of it), bound as receives from `from` and as
   SENDs toward `to` over the same rows, so a landed record releases the SEND that carries it on (M08), and the
   sender's digest named, which the relay's READY onward carries (#protocol).  0, or the refusal */
static int mirror(struct lease *lease,const struct ready *r,const struct mesh_transfer *t,uint32_t from,uint32_t to){
  struct mesh_ctx *c=&lease->context;
  struct mesh_client *client=mesh_client_at(M,c->client);
  int error=0;
  client->tx[mesh_destination(client,from)].node=from;
  client->tx[mesh_destination(client,to)].node=to;
  for(uint32_t k=0,i=0;k<MESH_QUEUES && !error;k++)for(uint32_t j=0;j<r->lengths[k] && !error;j++,i++){
    const struct mesh_transfer *x=t+i;
    struct mesh_section s;
    if(!x->count)error=EPROTO;
    else if(!(error=mesh_section_create(c,mesh_relay_slot(x,r->depth,M->pgsz),x->stride?x->count:1,&s))){
      s.bytes=x->bytes;if(!x->stride){s.stride=0;s.count=x->count;}
      error=mesh_transfer_bind(c,from,k,MESH_RECEIVE,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
      if(!error)error=mesh_transfer_bind(c,to,k,MESH_SEND,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
    }
  }
  if(!error)error=mesh_transfers_prepare(c,r->slots,r->invocations,r->depth);
  if(!error && r->cyclic)error=mesh_transfers_cyclic(c);
  if(!error)error=mesh_transfers_digest(c,r->digest);
  return error;
}
/* a READY refused as a value, naming this node */
static void ready_refuse(struct port *port,const struct message *heard,int64_t code){
  port_say(port,(struct message){.kind=FAIL,.key=heard->key,.from=heard->to,.to=heard->from,.code=code,.failed=M->node,.stripe=heard->stripe},
           NULL,0);
}
/* a READY's program apart from its stripe and route, the lease of a relayed direction its stripes share (FNV-1a) */
static uint64_t ready_shape(const struct ready *ready,const struct mesh_transfer *list,uint64_t transfers){
  struct ready shape=*ready;
  shape.hops=shape.stripe=shape.exclusive=0;
  uint64_t x=UINT64_C(0xcbf29ce484222325);
  for(size_t i=0;i<sizeof shape;i++)x=(x^((const unsigned char *)&shape)[i])*UINT64_C(0x100000001b3);
  for(size_t i=0;i<transfers*sizeof *list;i++)x=(x^((const unsigned char *)list)[i])*UINT64_C(0x100000001b3);
  return x;
}
/* the lease a relayed direction's other stripes through this node took: a relay flow of the same execution receiving
   from the same far end, whose partner sends toward the same far end; NULL where none has */
static struct lease *relay_lease(const struct flow *f,const struct flow *g){
  for(struct flow *h=flows;h;h=h->next){
    if(h==f || h->key!=f->key || h->far!=f->far || h->stripes!=f->stripes || !h->partner || h->partner->far!=g->far || h->state==FAILED ||
       h->stopping)continue;
    struct lease *lease=h->side[MESH_RECEIVE].lease?h->side[MESH_RECEIVE].lease:h->pending[MESH_RECEIVE].lease;
    if(lease)return lease;
  }
  return NULL;
}
/* a relay's direction a READY told: received by the flow it arrived on and sent on by its partner, over a lease that
   mirrors its lists, the one its program's other stripes through this node share (each stripe's flows prepare their
   own invocations of its rows, #cables), else a new one; EPROTO where a stripe's program is not its lease's */
static int relay_direction(struct flow *f,const struct ready *ready,const struct mesh_transfer *list,uint64_t shape){
  struct flow *g=f->partner;
  struct lease *lease=relay_lease(f,g);
  if(lease && lease->shape!=shape)return EPROTO;
  if(!lease){
    if(!(lease=lease_take(f->key)))return EBUSY;
    int error=mirror(lease,ready,list,f->far,g->far);
    if(error){lease_free(lease);return error;}
    lease->shape=shape;
  }
  struct mesh_client *client=mesh_client_at(M,lease->context.client);
  f->pending[MESH_RECEIVE]=(struct side){lease->context.client,mesh_destination(client,f->far),lease};
  g->pending[MESH_SEND]=(struct side){lease->context.client,mesh_destination(client,g->far),lease};
  lease->users+=2;
  if(!f->owner)flow_program(f,f->pending[MESH_RECEIVE].owner,f->pending[MESH_RECEIVE].destination);
  if(!g->owner)flow_program(g,g->pending[MESH_SEND].owner,g->pending[MESH_SEND].destination);
  flow_step(g);
  return 0;
}

static uint32_t program_queues(uint64_t owner,uint32_t d);
/* a flow taken by its client's program toward destination `d` (its one job before, JOB_PAIR, reads no side), its port
   among the program's (mesh_client `ports`, the bells mesh_transfers_wake rings) */
static void flow_take(struct flow *f,uint64_t owner,uint32_t d){
  f->side[MESH_SEND]=f->side[MESH_RECEIVE]=(struct side){owner,d,NULL};
  flow_program(f,owner,d);
  if(f->hold)f->deadline=now+MESH_PAIR_NS;
  f->hold=0;
  struct mesh_client *client=mesh_client_at(M,owner);
  atomic_fetch_or_explicit(&client->ports,UINT32_C(1)<<f->port->index,memory_order_release);
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
}
static int program_live(uint64_t key,uint32_t node,uint64_t *owner,uint32_t *d){
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    const uint64_t o=accepted[s];
    if(!o || !owned(o) || client->key!=key)continue;
    const uint32_t at=mesh_destination(client,node);
    if(at==MESH_DESTINATIONS || client->tx[at].node!=node)continue;
    for(struct flow *g=flows;g;g=g->next)if(g->owner==o && g->destination==at && g->state!=FAILED){*owner=o;*d=at;return 1;}
  }
  return 0;
}
/* the program of execution `key` that runs here (program_runs: its slot's owner), 0 where none */
static uint64_t program_keyed(uint64_t key){
  for(uint32_t s=0;s<MESH_CLIENTS;s++)
    if(accepted[s] && program_runs(accepted[s]) && mesh_client_at(M,accepted[s])->key==key)return accepted[s];
  return 0;
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
/* design/bridge-and-ipc.md#cables: a program's pair of end nodes and its execution mixed (murmur3's finalizer), alike at
   either end and on every relay, which with a stripe picks the cable a routed hop takes */
static uint64_t pair_hash(uint64_t key,uint32_t a,uint32_t b){
  uint64_t x=key^((uint64_t)(a<b?a:b)<<32|(a<b?b:a));
  x^=x>>33;x*=UINT64_C(0xff51afd7ed558ccd);x^=x>>33;x*=UINT64_C(0xc4ceb9fe1a85ec53);x^=x>>33;
  return x;
}
/* the cable of a routed hop toward the node of `next` (its n cables) that stripe `stripe` of the program between `a`
   and `b` rides: stripe j on cable (hash + j) mod n, so a hop of c cables carries stripes/c of a program whose stripes
   c divides */
static struct port *stripe_cable(struct port **next,uint32_t n,uint64_t key,uint32_t a,uint32_t b,uint32_t stripe){
  return next[(pair_hash(key,a,b)+stripe)%n];
}

/* design/bridge-and-ipc.md#protocol, #routes: a READY.  Its flow here pairs; one heard by a relay's flow (the far end's program telling its direction) is mirrored and sent on by the
   relay's other flow toward the route's start, so every relay of the route carries both directions; a program whose
   route goes on past this node is relayed toward the route's next node, over rows the bridge leases (no client of this
   node takes part, its stripes sharing them), each stripe on the cable of the next hop it rides (stripe_cable); one
   that ends here is taken by a program awaiting it, else held, its queue pairs made and no rows, for the client whose
   program takes it (#flows).  One that ends here naming another digest than
   this node's program of its key (#protocol) is refused, and that program failed toward every destination, both as
   values (ESTALE: program_stale).  One that ends on a port where this node's program of its key toward its sender has
   no flow, while that program runs on others, finds the two ends' maps apart: refused, and the program failed, both as
   values (EPROTO) */
void ready_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct ready ready;
  if(heard->bytes<sizeof ready){port_lost(port,EPROTO);return;}
  memcpy(&ready,payload,sizeof ready);
  uint64_t transfers=0;
  for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=ready.lengths[k];
  const uint32_t queues=ready.slots?heard->count/ready.slots:0;
  if(!ready.hops || !queues || queues>MESH_QUEUES || heard->count%ready.slots || !ready.stripes || ready.stripes>MESH_STRIPES ||
     ready.stripe>=ready.stripes || ready.stripe!=heard->stripe ||
     heard->bytes!=sizeof ready+(uint64_t)ready.hops*sizeof(uint32_t)+transfers*sizeof(struct mesh_transfer)){port_lost(port,EPROTO);return;}
  uint32_t *route=malloc((size_t)ready.hops*sizeof *route);
  struct mesh_transfer *list=malloc((transfers?transfers:1)*sizeof *list);
  if(!route || !list){free(route);free(list);ready_refuse(port,heard,ENOMEM);return;}
  memcpy(route,payload+sizeof ready,(size_t)ready.hops*sizeof *route);
  memcpy(list,payload+sizeof ready+(size_t)ready.hops*sizeof *route,transfers*sizeof *list);
  const uint64_t shape=ready_shape(&ready,list,transfers);
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,ready.stripe);
  uint64_t owner=0;uint32_t d=0;
  if(f && (f->heard&HEARD_READY)){}
  else if(f && !f->partner && digest_apart(program_digest(f->owner),ready.digest)){
    flow_fail(f,ESTALE,4,FAIL,heard->from,M->node);
    program_stale(f->owner,heard->from);
  }
  else if(f && !f->partner && (f->stripes!=ready.stripes || f->queues!=queues || (uint32_t)f->qps!=heard->count))
    flow_fail(f,EPROTO,4,FAIL,M->node,M->node);
  else if(f){
    int error=f->partner && !f->side[MESH_RECEIVE].owner && !f->pending[MESH_RECEIVE].owner?relay_direction(f,&ready,list,shape):0;
    if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
    else{
      if(f->partner)port_say(f->partner->port,(struct message){.kind=READY,.count=heard->count,.key=heard->key,.from=heard->from,
                                                               .to=heard->to,.stripe=heard->stripe},payload,heard->bytes);
      f->heard|=HEARD_READY;flow_step(f);
    }
  }else if(route[0]!=M->node || route[ready.hops-1]!=heard->to || heard->from==M->node)ready_refuse(port,heard,EPROTO);
  else if(ready.hops==1){
    struct await *a=await_find(heard->key,heard->from);
    const uint64_t keyed=a?a->owner:program_keyed(heard->key);
    if(digest_apart(program_digest(keyed),ready.digest)){
      ready_refuse(port,heard,ESTALE);
      program_stale(keyed,heard->from);
    }else if(a && (program_queues(a->owner,a->destination)!=queues || mesh_client_at(M,a->owner)->tx[a->destination].slots!=ready.slots ||
             (a->stripes && a->stripes!=ready.stripes) || (a->taken>>ready.stripe&1))){
      say("link %u (%s): program %016llx: node %u's READY (stripe %u of %u) is not the shape of the program awaiting it here: refused\n",
          port->index,port->name,(unsigned long long)heard->key,heard->from,ready.stripe,ready.stripes);
      ready_refuse(port,heard,EPROTO);
      program_fail(a->owner,a->destination,EPROTO,M->node);
    }else if(!a && program_live(heard->key,heard->from,&owner,&d)){
      say("link %u (%s): program %016llx: node %u's READY came on a port its program here does not use (the two ends' maps "
          "differ): refused\n",port->index,port->name,(unsigned long long)heard->key,heard->from);
      ready_refuse(port,heard,EPROTO);
      program_fail(owner,d,EPROTO,M->node);
    }else if(!(f=flow_make(port,heard->key,M->node,heard->from,queues,ready.slots,ready.stripe,ready.stripes)))
      ready_refuse(port,heard,E2BIG);
    else{
      f->hold=1;f->heard=HEARD_READY;f->digest=ready.digest;
      if(a){flow_take(f,a->owner,a->destination);await_took(a,ready.stripe,ready.stripes);ready_say(f,&f->far,1);}
      flow_step(f);
    }
  }else{
    struct port *next[MESH_PORTS];
    struct flow *g=NULL;
    const uint32_t n=cables(route[1],next);
    f=n?flow_make(port,heard->key,heard->to,heard->from,queues,ready.slots,ready.stripe,ready.stripes):NULL;
    if(f)g=flow_make(stripe_cable(next,n,heard->key,heard->from,heard->to,ready.stripe),heard->key,heard->from,heard->to,queues,
                     ready.slots,ready.stripe,ready.stripes);
    if(!g){
      if(f){f->state=FAILED;flow_free(f);}
      ready_refuse(port,heard,n?E2BIG:EHOSTUNREACH);
    }else{
      f->partner=g;g->partner=f;
      int error=relay_direction(f,&ready,list,shape);
      if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
      else{
        ready_say(g,route+1,ready.hops-1);
        f->heard=HEARD_READY;flow_step(f);
      }
    }
  }
  free(route);free(list);
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

/* design/bridge-dataflow.md#stripes: every registered ring's next request read on the event loop: a ring whose client
   is gone deregistered; a request naming a node a cable of this node's is paired with is left to the send threads of
   the ports toward it, which take it once their links' stripe lanes are served (made with the link, and made again
   with it); and it fails as a value (its writer's word UINT64_MAX, its ring's code) where its node is this node
   (EINVAL) or no cable's (EHOSTUNREACH: a stripe goes over one link, its node a neighbour) */
void stripes_requests_turn(void){
  _Atomic uint64_t *rings=mesh_rings(M);
  for(uint32_t r=0;r<MESH_RINGS;r++){
    uint64_t at=atomic_load_explicit(rings+r,memory_order_acquire);
    if(!at)continue;
    struct mesh_requests *q=at>=M->data_off && at<=M->length-sizeof(struct mesh_requests)?(void *)((char *)M+at):NULL;
    if(!q || !owned(q->owner) || !q->entries || (q->entries&(q->entries-1)) || q->entries>(M->length-at-sizeof *q)/sizeof(struct mesh_stripe)){
      atomic_compare_exchange_strong(rings+r,&at,0);
      continue;
    }
    uint64_t k=atomic_load_explicit(&q->taken,memory_order_acquire);
    const struct mesh_stripe *e=q->request+(k&(q->entries-1));
    if(atomic_load_explicit(&e->ready,memory_order_acquire)!=k+1)continue;
    const uint32_t node=e->node;
    struct port *cable[MESH_PORTS];
    const uint32_t n=node==M->node?0:cables(node,cable);
    const int64_t code=node==M->node?EINVAL:EHOSTUNREACH;
    if(n)continue;
    const uint64_t word=e->word,owner=q->owner;
    if(!atomic_compare_exchange_strong_explicit(&q->taken,&k,k+1,memory_order_acq_rel,memory_order_relaxed))continue;
    if(atomic_load_explicit(&q->code,memory_order_relaxed)!=code)
      say("stripes: a request of client %llx toward node %u refused: %s\n",(unsigned long long)owner,node,strerror((int)code));
    stripe_failed(at,owner,word,code);
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
static void stream_asked(int fd,uint32_t node,uint64_t bytes);
/* a client's request read whole: a channel opened toward its node's service (here, or its CHANNEL_OPEN sent to the next hop),
   a service registered, a ledger cut, or a raw stream (design/bridge-dataflow.md#streams) */
static void asked(struct asking *a){
  struct mesh_request *r=&a->request;
  const int fd=a->fd;a->fd=-1;
  fd_watch(fd,EVFILT_READ,0,0);
  r->service[sizeof r->service-1]=0;r->path[sizeof r->path-1]=0;
  if(r->magic==MESH_REQUEST_MAGIC && r->kind==MESH_REQUEST_STREAM){stream_asked(fd,r->node,r->bytes);return;}
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

/* design/bridge-dataflow.md#streams: a stream's verbs work on its port's queue, always guarded (port_guard): its queue
   pair made (STREAMS_MAKING); connected to the peer's and ready (STREAMS_CONNECTING: its chains' storage, and at the
   receiver its first `window` receives posted before it tells POSTED, so no SEND of the sender's comes before its
   receive: the 2026-09-10 benchmark's first defect, design/transport-one-gib-2026-09-10.md); or, once neither
   progress thread reads it, flushed and destroyed (STREAMS_CLOSING) */
static void streams_job(struct port *port,int job){
  struct stream *s=&port->stream;
  s->job=job;atomic_store_explicit(&s->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    if(job==STREAMS_CLOSING){atomic_store_explicit(&s->up,0,memory_order_release);port_settle(port);}
    port_guard(port,1);
    if(job==STREAMS_MAKING)result=pair_up(&s->provider,1);
    else if(job==STREAMS_CONNECTING){
      result=pair_receive(&s->provider,(const uint32_t (*)[2])s->peer,port->heard.lid,port->heard.gid);
      if(!result)result=pair_send(&s->provider);
      const uint32_t frames=(s->page_bytes+4095)/4096,cqe=(uint32_t)MIN(s->provider.sent->cqe,s->provider.completion->cqe);
      s->window=MIN(MESH_STREAM_WINDOW,MIN(s->sending?s->provider.queues[0].send_capacity/frames:s->provider.queues[0].receive_capacity/frames,
                                           cqe>1?cqe-1:1));
      if(!s->window)s->window=1;
      s->spans=calloc(s->window,sizeof *s->spans);s->sends=calloc(s->window,sizeof *s->sends);
      s->receives=calloc(s->window,sizeof *s->receives);s->completions=calloc(s->window,sizeof *s->completions);
      if(!result && (!s->spans || !s->sends || !s->receives || !s->completions))result=ENOMEM;
      s->posted=s->completed=0;
      if(!result && !s->sending)result=stream_post(port,MIN(s->window,s->count));
      if(!result)result=port_threads(port);
    }else{
      s->flushed=s->provider.queues?pair_flush(&s->provider,MESH_FLUSH_POLLS):0;
      result=down_pair(&s->provider)?0:EBUSY;
      free(s->spans);free(s->sends);free(s->receives);free(s->completions);
      s->spans=NULL;s->sends=NULL;s->receives=NULL;s->completions=NULL;
    }
    port_guard(port,0);
    s->result=result;
    port_pairs(port);
    atomic_store_explicit(&s->done,1,memory_order_release);
    wake();
  });
}
/* a stream's lease filled with its pattern (the sender: word w of page p is p << 32 ^ w ^ MESH_STREAM_MAGIC, as
   mesh-verbs-stream.c and mesh-page-stream.c wrote it), cleared (the receiver, so a page not written fails its check),
   or checked word by word (the receiver, once every page landed); off the event loop, on a global queue */
static void streams_work(struct port *port,int job){
  struct stream *s=&port->stream;
  s->job=job;atomic_store_explicit(&s->done,0,memory_order_relaxed);
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{
    const uint64_t words=s->page_bytes/sizeof(uint64_t);
    uint64_t mismatches=0,correct=0;
    for(uint64_t p=0;p<s->count;p++){
      uint64_t *page=(uint64_t *)(wire.data+stream_offset(port,p)),bad=0;
      if(job==STREAMS_FILLING && s->sending)for(uint64_t w=0;w<words;w++)page[w]=(p<<32)^w^MESH_STREAM_MAGIC;
      else if(job==STREAMS_FILLING)memset(page,0,s->page_bytes);
      else for(uint64_t w=0;w<words;w++)bad+=page[w]!=((p<<32)^w^MESH_STREAM_MAGIC);
      mismatches+=bad;correct+=!bad;
    }
    if(job==STREAMS_CHECKING){s->own.mismatches=mismatches;s->own.correct=correct;}
    s->result=0;
    atomic_store_explicit(&s->done,1,memory_order_release);
    wake();
  });
}
/* the stream's lease returned to the window and its state cleared, the requester (the sender's end) answered with
   `code` where it still waits */
static void streams_clear(struct port *port,int64_t code){
  struct stream *s=&port->stream;
  if(s->pages)mesh_bits_clear(mesh_arena_bits(M),s->page,s->pages);
  if(s->fd>=0){fd_answer(s->fd,(int32_t)code);close(s->fd);}
  if(s->committed)port->committed-=1;
  *s=(struct stream){.fd=-1};
}
/* design/recovery.md#loss: a stream that failed (a verbs call's error, a completion's, the peer's FAIL, the link lost,
   past its deadline): the peer told where the failure is this end's, the requester answered with the code, its queue
   pair closed (flushed) where it has one */
void streams_fail(struct port *port,int64_t code,int tell){
  struct stream *s=&port->stream;
  say("link %u (%s): stream %016llx toward node %u: %s\n",port->index,port->name,(unsigned long long)s->key,port->peer,strerror((int)code));
  if(tell)port_say(port,(struct message){.kind=STREAM,.count=STREAM_FAILED,.key=s->key,.code=code},NULL,0);
  atomic_store_explicit(&s->up,0,memory_order_release);
  if(s->fd>=0){fd_answer(s->fd,(int32_t)code);close(s->fd);s->fd=-1;}
  if(s->job)s->state=STREAMS_CLOSING;
  else if(s->provider.queues || s->state>=STREAMS_MAKING){s->state=STREAMS_CLOSING;streams_job(port,STREAMS_CLOSING);}
  else streams_clear(port,0);
}
/* the stream's lease, `bytes` in whole blocks from the window; 0, or ENOMEM */
static int streams_lease(struct port *port,uint64_t bytes){
  struct stream *s=&port->stream;
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  s->page_bytes=(uint32_t)block;s->count=(uint32_t)(bytes/block);s->bytes=(uint64_t)s->count*block;
  const uint64_t pages=s->bytes/M->pgsz;
  const uint32_t page=pages && pages<=UINT32_MAX?mesh_arena_alloc(&bridge,(uint32_t)pages,M->block):MESH_ABSENT;
  if(page==MESH_ABSENT)return ENOMEM;
  s->page=page;s->pages=(uint32_t)pages;
  return 0;
}
/* design/bridge-dataflow.md#streams: a client's request for a raw stream of `bytes` toward node `node` (a neighbour),
   its socket held for the answer: the first cable's port toward it, its lease filled, then its queue pair made and
   offered (streams_turn); refused at once as a value where the node has no cable (EHOSTUNREACH), that port runs a
   stream (EBUSY), the bytes are not a block at least or exceed 2^32 blocks (EINVAL), or the window has no room
   (ENOMEM) */
static void stream_asked(int fd,uint32_t node,uint64_t bytes){
  struct port *cable[MESH_PORTS];
  const uint32_t n=node==M->node?0:cables(node,cable);
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  int64_t code=!n?EHOSTUNREACH:bytes<block || bytes/block>UINT32_MAX?EINVAL:cable[0]->stream.state!=STREAMS_DOWN?EBUSY:0;
  if(!code)code=streams_lease(cable[0],bytes);
  if(code){fd_answer(fd,(int32_t)code);close(fd);if(n && code==ENOMEM)streams_clear(cable[0],0);return;}
  struct stream *s=&cable[0]->stream;
  static uint32_t streams;
  s->fd=fd;s->sending=1;s->key=(uint64_t)M->node<<32|++streams;s->deadline=now+MESH_PAIR_NS;
  s->state=STREAMS_FILLING;
  say("link %u (%s): stream %016llx of %llu bytes toward node %u asked\n",cable[0]->index,cable[0]->name,(unsigned long long)s->key,
      (unsigned long long)s->bytes,node);
  streams_work(cable[0],STREAMS_FILLING);
}
/* a peer's STREAM: its offer (this end leases, clears, makes and connects, then tells POSTED), its posted receives (this
   end connects and sends), its result (the requester answered), or its failure */
void stream_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct stream *s=&port->stream;
  if(heard->count==STREAM_OFFER){
    int64_t code=s->state!=STREAMS_DOWN?EBUSY:0;
    if(!code)code=streams_lease(port,(uint64_t)heard->code);
    if(code){
      if(s->state==STREAMS_DOWN)streams_clear(port,0);
      port_say(port,(struct message){.kind=STREAM,.count=STREAM_FAILED,.key=heard->key,.code=code},NULL,0);
      return;
    }
    s->fd=-1;s->sending=0;s->key=heard->key;s->peer[0][0]=heard->pair[0][0];s->peer[0][1]=heard->pair[0][1];
    s->deadline=now+MESH_PAIR_NS;s->state=STREAMS_FILLING;
    streams_work(port,STREAMS_FILLING);
    return;
  }
  if(heard->key!=s->key || s->state==STREAMS_DOWN)return;
  if(heard->count==STREAM_POSTED && s->sending && s->state==STREAMS_MADE){
    s->peer[0][0]=heard->pair[0][0];s->peer[0][1]=heard->pair[0][1];
    s->state=STREAMS_CONNECTING;streams_job(port,STREAMS_CONNECTING);
  }else if(heard->count==STREAM_RESULT && s->sending && heard->bytes==sizeof s->told){
    memcpy(&s->told,payload,sizeof s->told);
    const struct stream_result *r=&s->told,*o=&s->own;
    const double interior=r->high_ns>r->low_ns?(double)(r->high_ns-r->low_ns)/1e9:0;
    const uint64_t interior_bytes=(r->high_count-r->low_count)*r->page_bytes;
    char line[1024];
    snprintf(line,sizeof line,"{\"stream\":\"raw\",\"key\":\"%016llx\",\"node\":%u,\"peer\":%u,\"port\":%u,\"bytes\":%llu,\"page_bytes\":%u,"
      "\"pages\":%u,\"window\":%u,\"sender\":{\"window\":%u,\"posts\":%llu,\"submit_to_last_s\":%.9f,\"polls\":%llu,\"empty_polls\":%llu,"
      "\"misordered\":%llu},\"receiver\":{\"window\":%llu,\"completed\":%llu,\"mismatches\":%llu,\"correct_pages\":%llu,\"misordered\":%llu,"
      "\"first_to_last_s\":%.9f,\"first_to_last_bytes\":%llu,\"interior_s\":%.9f,\"interior_bytes\":%llu,\"interior_GBps\":%.6f,\"polls\":%llu,"
      "\"empty_polls\":%llu}}\n",(unsigned long long)s->key,M->node,port->peer,port->index,(unsigned long long)s->bytes,s->page_bytes,s->count,
      s->window,s->window,(unsigned long long)o->posts,o->last_ns>o->begin_ns?(double)(o->last_ns-o->begin_ns)/1e9:0,(unsigned long long)o->polls,
      (unsigned long long)o->empty_polls,(unsigned long long)o->misordered,(unsigned long long)r->window,(unsigned long long)r->completed,
      (unsigned long long)r->mismatches,(unsigned long long)r->correct,(unsigned long long)r->misordered,
      r->last_ns>r->first_ns?(double)(r->last_ns-r->first_ns)/1e9:0,(unsigned long long)((r->completed-r->first_count)*r->page_bytes),interior,
      (unsigned long long)interior_bytes,interior>0?(double)interior_bytes/interior/1e9:0,(unsigned long long)r->polls,
      (unsigned long long)r->empty_polls);
    say("link %u (%s): stream %s",port->index,port->name,line);
    if(s->fd>=0){fd_answer(s->fd,0);if(send(s->fd,line,strlen(line),0)<0){}close(s->fd);s->fd=-1;}
    s->state=STREAMS_CLOSING;
    if(!s->job)streams_job(port,STREAMS_CLOSING);
  }else if(heard->count==STREAM_FAILED)streams_fail(port,heard->code?heard->code:ECANCELED,0);
}
/* design/bridge-dataflow.md#streams: the port's stream's turn, on the event loop: a job's end taken (its lease filled
   or cleared: its queue pair made; made: the sender offers it, the receiver connects; connected: running, the receiver
   telling POSTED; checked: the receiver tells its result and closes; closed: its lease returned), its end at a
   progress thread taken (the receiver checks every word; the sender waits for the result), and its failure or its
   deadline (the pairing window from its ask or offer) ending it */
void streams_turn(struct port *port){
  struct stream *s=&port->stream;
  if(s->state==STREAMS_DOWN)return;
  if(s->job){
    if(!atomic_load_explicit(&s->done,memory_order_acquire))return;
    const int job=s->job,result=s->result;
    s->job=0;
    if(job==STREAMS_CLOSING){
      if(result){say("link %u (%s): stream teardown: %s\n",port->index,port->name,strerror(result));return;}
      say("link %u (%s): stream %016llx closed (%llu completions flushed)\n",port->index,port->name,(unsigned long long)s->key,
          (unsigned long long)s->flushed);
      streams_clear(port,ECANCELED);
      return;
    }
    if(s->state==STREAMS_CLOSING){streams_job(port,STREAMS_CLOSING);return;}
    if(result){streams_fail(port,result,1);return;}
    if(job==STREAMS_FILLING){
      if(port->committed+1>port->device.max_pairs){streams_fail(port,ENOSPC,1);return;}
      port->committed+=1;s->committed=1;s->state=STREAMS_MAKING;streams_job(port,STREAMS_MAKING);
    }else if(job==STREAMS_MAKING){
      s->state=STREAMS_MADE;
      if(s->sending){
        struct message said={.kind=STREAM,.count=STREAM_OFFER,.key=s->key,.code=(int64_t)s->bytes};
        said.pair[0][0]=s->provider.queues[0].pair->qp_num;said.pair[0][1]=s->provider.psn&0xffffff;
        port_say(port,said,NULL,0);
      }else{s->state=STREAMS_CONNECTING;streams_job(port,STREAMS_CONNECTING);}
    }else if(job==STREAMS_CONNECTING){
      s->state=STREAMS_RUNNING;
      atomic_store_explicit(&s->up,1,memory_order_release);port_bell(port);
      if(!s->sending){
        struct message said={.kind=STREAM,.count=STREAM_POSTED,.key=s->key};
        said.pair[0][0]=s->provider.queues[0].pair->qp_num;said.pair[0][1]=s->provider.psn&0xffffff;
        port_say(port,said,NULL,0);
      }
    }else if(job==STREAMS_CHECKING){
      s->own.bytes=s->bytes;s->own.pages=s->count;s->own.page_bytes=s->page_bytes;s->own.window=s->window;
      port_say(port,(struct message){.kind=STREAM,.count=STREAM_RESULT,.key=s->key},&s->own,sizeof s->own);
      s->state=STREAMS_CLOSING;streams_job(port,STREAMS_CLOSING);
    }
    return;
  }
  const int64_t failed=atomic_exchange(&s->failed,0);
  if(failed){streams_fail(port,failed,1);return;}
  if(s->state==STREAMS_RUNNING && atomic_load_explicit(&s->finished,memory_order_acquire) && !s->sending){
    s->state=STREAMS_CHECKING;streams_work(port,STREAMS_CHECKING);return;
  }
  if(now>s->deadline && s->state!=STREAMS_CLOSING)streams_fail(port,ETIMEDOUT,1);
}

/* the queues (its highest queue with a list, plus one) and slots of a client's program toward its destination `d` */
static uint32_t program_queues(uint64_t owner,uint32_t d){
  uint32_t queues=0;
  for(uint32_t q=0;q<MESH_QUEUES;q++)for(int w=0;w<2;w++)if(atomic_load(mesh_order_length(M,owner,d*MESH_QUEUES+q,w)))queues=q+1;
  return queues;
}
/* whether a READY of execution `key` from `far` left the flow held here, on any port, for the client whose program
   takes it */
static int flow_held(const struct flow *f,uint64_t key,uint32_t far){
  return f->hold && f->key==key && f->near==M->node && f->far==far && f->state!=FAILED && !f->stopping;
}
/* design/bridge-and-ipc.md#cables: the cables a record holds paired with `peer`, and the stripes a routed program takes:
   the least common multiple of its hops' cables (a hop's the fewer of its two ends' records', the first hop's this
   node's own `first`), so each cable of every hop carries an equal share, or past MESH_STRIPES the fewest cables of a
   hop, so each stripe rides a cable of its own on every hop and the route carries its slowest hop's cables */
static uint32_t record_cables(const struct mesh_record *r,uint32_t peer){
  uint32_t n=0;
  for(uint32_t p=0;r && p<r->ports && p<MESH_PORTS;p++)
    n+=atomic_load_explicit(&r->port[p].phase,memory_order_relaxed)==MESH_PAIRED && atomic_load_explicit(&r->port[p].peer,memory_order_relaxed)==peer;
  return n;
}
static uint32_t route_stripes(const uint32_t *route,uint32_t hops,uint32_t first){
  uint32_t stripes=1,fewest=MESH_STRIPES;
  for(uint32_t i=0;i+1<hops;i++){
    uint32_t c=i?MIN(record_cables(record_held(route[i]),route[i+1]),record_cables(record_held(route[i+1]),route[i])):first;
    c=MIN(MAX(c,1u),MESH_STRIPES);
    uint32_t a=stripes,b=c;
    while(b){const uint32_t t=a%b;a=b;b=t;}
    stripes=stripes/a*c>MESH_STRIPES?MESH_STRIPES+1:stripes/a*c;
    fewest=MIN(fewest,c);
  }
  return stripes<=MESH_STRIPES?stripes:fewest;
}
/* whether this end's map routes to `node` over a cable it holds paired now */
static int route_held(uint32_t node){
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  const int held=hops>=2 && cables(route[1],next)>0;
  free(route);
  return held;
}
/* the newest record this node holds or held of `node` (its entry apart from the map; design/bridge-and-ipc.md#records);
   NULL where it never heard of it */
static const struct known *known_of(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  return NULL;
}
/* design/bridge-and-ipc.md#routes: the higher end of a routed pair awaits the lower's route, whatever its own map holds:
   the stripes the lower's READYs left held taken at once, the rest as they come (ready_heard), within the peer window
   where its own map routes to the lower too, else within the pairing window from when the lower was last reachable
   (its record pruned: a link of its route pairing again), and refused at once (EHOSTUNREACH, naming the lower) where it
   has been unreachable longer or was never heard of; a revoke of the client's program (its slot retired) or the lower's
   failure as a value ends it sooner.  Each take updates the await before the READY that may fail the program goes out */
static void program_await(struct mesh_client *client,uint64_t owner,uint32_t d,uint32_t node,uint32_t queues,uint32_t slots){
  struct await *a=calloc(1,sizeof *a);
  if(!a){status_failed(&client->status[d],owner,ENOMEM,4,M->node);return;}
  *a=(struct await){.next=awaits,.key=client->key,.owner=owner,.created=now,.destination=d,.far=node,.routed=1,.deadline=now+MESH_PEER_NS};
  awaits=a;await_count++;busy_tell();
  for(struct flow *h=flows;h;h=h->next){
    if(!flow_held(h,client->key,node))continue;
    if(h->queues!=queues || (uint32_t)h->qps!=queues*slots || (a->stripes && h->stripes!=a->stripes) || (a->taken>>h->stripe&1)){
      flow_fail(h,EPROTO,4,FAIL,M->node,M->node);
      program_fail(owner,d,EPROTO,M->node);
      return;
    }
    flow_take(h,owner,d);
    const int last=await_took(a,h->stripe,h->stripes);
    ready_say(h,&h->far,1);flow_step(h);
    if(last || (a=await_of(owner,d))==NULL)return;
  }
  if(a->taken || route_held(node))return;
  const struct known *k=known_of(node);
  const uint64_t since=record_held(node)?now:k && k->gone?k->gone:0;
  a->routed=0;a->deadline=since+MESH_PAIR_NS;
  if(!since || now>=a->deadline){
    say("program %016llx: node %u, the lower end of its route, %s: refused\n",(unsigned long long)client->key,node,
        since?"has been unreachable longer than the pairing window":"was never heard of");
    program_fail(owner,d,EHOSTUNREACH,node);
  }
}
/* design/bridge-and-ipc.md#clients, #cables, #routes: a started program's destination taken onto its ports.  Over the
   cables paired with its node, a flow a cable in the cable order both ends read alike, flow j carrying the program's
   invocations t with t mod n = j, each asking for queue pairs of its own (#lanes); else along a route, which the
   pair's lower node chooses on its map and both ends take: the lower's flows, one a stripe of its route's
   (route_stripes), each on the lanes of the cable of the route's first hop it rides (stripe_cable), each READY
   carrying the route; the higher's program awaits the lower's READYs (program_await), each READY of its going back
   along the relays that READY made.  Its flows' sides the client's, a READY on each, its receives then posted
   (flow_step).  Refused as a value naming the node where it is this node (EINVAL) or no route reaches it (EHOSTUNREACH,
   the lower end's), and naming this node where its table is full (E2BIG) or a flow of its key toward the node runs for
   another program (EBUSY) */
static void program_take(struct mesh_client *client,uint64_t owner,uint32_t d){
  const uint32_t node=client->tx[d].node,slots=client->tx[d].slots,queues=program_queues(owner,d);
  if(!node || node==M->node){status_failed(&client->status[d],owner,EINVAL,4,node);return;}
  struct port *cable[MESH_PORTS];
  const uint32_t n=cables(node,cable);
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
  /* #routes: the peer's routed READY held here while this end sees cables to it: the two ends' maps differ */
  for(struct flow *h=flows;n && h;h=h->next){
    uint32_t j=0;
    if(!flow_held(h,client->key,node))continue;
    while(j<n && h->port!=cable[j])j++;
    if(j<n)continue;
    say("link %u (%s): program %016llx: node %u's READY came routed while this end has cables to it (the two ends' maps "
        "differ): refused\n",h->port->index,h->port->name,(unsigned long long)client->key,node);
    flow_fail(h,EPROTO,4,FAIL,M->node,M->node);
    program_fail(owner,d,EPROTO,M->node);
    return;
  }
  for(uint32_t j=0;j<n;j++){
    struct flow *f=flow_on(cable[j],client->key,M->node,node,j);
    if(f && !f->hold){program_fail(owner,d,EBUSY,M->node);return;}
    if(f && (f->stripes!=n || f->queues!=queues || (uint32_t)f->qps!=queues*slots)){flow_fail(f,EPROTO,4,FAIL,M->node,M->node);f=NULL;}
    if(!f && !(f=flow_make(cable[j],client->key,M->node,node,queues,slots,j,n))){program_fail(owner,d,E2BIG,M->node);return;}
    flow_take(f,owner,d);
    ready_say(f,&node,1);
    flow_step(f);
    if(f->state==FAILED)return;
  }
  if(n)return;
  if(M->node>node){program_await(client,owner,d,node,queues,slots);return;}
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  const uint32_t m=hops>=2?cables(route[1],next):0;
  if(!m){free(route);status_failed(&client->status[d],owner,EHOSTUNREACH,4,node);return;}
  const uint32_t stripes=route_stripes(route,hops,m);
  for(uint32_t j=0;j<stripes;j++){
    struct flow *f=flow_make(stripe_cable(next,m,client->key,M->node,node,j),client->key,M->node,node,queues,slots,j,stripes);
    if(!f){free(route);program_fail(owner,d,E2BIG,M->node);return;}
    flow_take(f,owner,d);
    ready_say(f,route+1,hops-1);
    flow_step(f);
    if(f->state==FAILED)break;
  }
  free(route);
}
/* design/bridge-and-ipc.md#stop: the bridge stopping: every program's words toward its destinations cancelled (~0) and
   its status the stop's code, stored from the event loop's thread as plain memory stores, no queue, thread or verbs
   call waited on; a program awaiting its routed peer's READYs told alike */
void programs_stop(int64_t code){
  for(struct flow *f=flows;f;f=f->next){
    if(!owned(f->owner))continue;
    if(f->cancel)mesh_cancel(M,f->cancel,f->destination);
    status_failed(&mesh_client_at(M,f->owner)->status[f->destination],f->owner,code,4,M->node);
  }
  for(struct await *a=awaits;a;a=a->next)
    if(owned(a->owner))status_failed(&mesh_client_at(M,a->owner)->status[a->destination],a->owner,code,4,M->node);
}
/* design/bridge-and-ipc.md#clients: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once, and on a
   liveness turn any client whose process exited retired; a program that ended (stopped, its client gone) leaves its
   ports; a program started is taken toward each destination it binds, once no flow of the slot's earlier client
   remains, or, where a READY of its execution held here names another digest than its own, failed with that READY's
   flow, both as values (ESTALE: program_stale).  Rows of a retired client or lease are released once no flow of one
   remains */
void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0,zombie=0;
  for(struct flow *f=flows;f;f=f->next)
    if(f->owner && (pid_t)(uint32_t)f->owner!=self && (!owned(f->owner) || atomic_load(&mesh_client_at(M,f->owner)->configured)!=f->owner))flow_end(f);
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
    for(struct flow *f=flows;f;f=f->next)earlier|=f->owner && f->owner!=owner && ((f->owner>>32)&(MESH_CLIENTS-1))==s && (pid_t)(uint32_t)f->owner!=self;
    if(earlier)continue;
    accepted[s]=owner;
    atomic_store_explicit(&client->ports,0,memory_order_relaxed);
    uint32_t stale=0;
    for(struct flow *h=flows;h;h=h->next)
      if(h->hold && h->key==client->key && h->near==M->node && h->state!=FAILED && !h->stopping &&
         digest_apart(atomic_load_explicit(&client->digest,memory_order_relaxed),h->digest)){
        stale=h->far;flow_fail(h,ESTALE,4,FAIL,h->far,M->node);
      }
    if(stale){program_stale(owner,stale);continue;}
    for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)if(mesh_destination_transfers(M,owner,d))program_take(client,owner,d);
  }
  for(struct flow *f=flows;f;f=f->next)
    zombie|=(f->owner && !owned(f->owner)) || (f->side[0].owner && !owned(f->side[0].owner)) || (f->side[1].owner && !owned(f->side[1].owner));
  /* design/bridge-dataflow.md#stripes: a stripe in flight may name a retired client's bytes and word */
  for(uint32_t p=0;p<port_count;p++)zombie|=atomic_load_explicit(&ports[p].stripes.flight,memory_order_acquire)!=0;
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}
/* design/bridge-and-ipc.md#flows: every flow's turn: its job's end taken, a stopped flow freed (or its queue pairs
   destroyed again a pause later, where the driver kept them), a data thread's failure or a window passed (its peer's
   program never started, its queue pairs never paired, a held program no client took) its failure */
void flows_turn(void){
  for(struct flow *f=flows,*next;f;f=next){
    next=f->next;
    if(f->job && atomic_load_explicit(&f->done,memory_order_acquire))flow_done(f);
    if(f->stopping){
      const int stopped=atomic_load_explicit(&f->stopped,memory_order_acquire);
      if(stopped==1 && !f->job)flow_free(f);
      else if(stopped==2 && !f->retry)f->retry=now+MESH_PAUSE_NS;
      else if(stopped==2 && now>=f->retry){f->retry=0;flow_stop(f,0,0,0);}
      continue;
    }
    if(f->state==FAILED)continue;
    const int64_t failed=atomic_load(&f->failed);
    if(failed)flow_fail(f,failed,atomic_load(&f->domain),FAIL,M->node,M->node);
    else if(f->state!=RUNNING && now>f->deadline)flow_fail(f,ETIMEDOUT,4,FAIL,f->far,f->far);
  }
  /* #routes: a program awaiting its lower peer's READYs: dropped with its program, failed past the peer window naming
     the peer, with the stripes it took */
  for(struct await *a=awaits,*next;a;a=next){
    next=a->next;
    if(!owned(a->owner) || atomic_load(&mesh_client_at(M,a->owner)->configured)!=a->owner)await_drop(a);
    else if(now>a->deadline && !a->routed && route_held(a->far)){a->routed=1;a->deadline=a->created+MESH_PEER_NS;}
    else if(now>a->deadline)program_fail(a->owner,a->destination,a->routed?ETIMEDOUT:EHOSTUNREACH,a->far);
  }
}
