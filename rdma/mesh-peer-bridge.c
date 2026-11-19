#include "mesh-bridge.h"
#include "mesh-peer-private.h"
#include "mesh-peer-plan.h"

enum { OFFER=1,ACCEPT,OPEN,ISSUE,ABORT };
struct signal {uint32_t kind,row;};
struct notice {struct notice *next;uint32_t count;struct signal events[];};
struct channel {uint32_t stream;struct mesh_stream_view view;uint64_t prefix[2],bytes[2];};
struct job {
  struct job *next;
  uint32_t peer;uint64_t group,owner,expires;
  int fd,pinned,prepared,accepted,started,closed,sent;
  _Atomic int failed,finished;
  void *offer,*remote;size_t local_bytes,remote_bytes;
  struct mesh_peer_transfer *records;
  struct peer_plan plan;
  struct mesh_ctx context;struct mesh_section table;uint32_t holds;
  struct channel channel[MESH_STREAMS];
  pthread_mutex_t lock;struct notice *incoming,**end_in,*outgoing,**end_out;
};
struct client {struct client *next;int fd;size_t at;struct mesh_peer_request req;};
static struct job *jobs;
static struct client *clients;
static int local=-1;
#define EVENT_PEER (UINT64_MAX-1)
static struct port *reachable(uint32_t peer){
  for(uint32_t i=0;i<port_count;i++)if(ports[i].greeted && ports[i].peer==peer && (ports[i].heard.code&PEER_EXTENTS))return ports+i;
  return NULL;
}
static int peer_send(struct job *j,uint32_t kind,const void *data,size_t bytes){
  struct port *p=reachable(j->peer);
  if(!p)return ENOLINK;
  if(bytes>64u*1024*1024-sizeof(struct message))return EOVERFLOW;
  uint64_t group=j->group;int64_t code;memcpy(&code,&group,sizeof code);
  port_say(p,(struct message){.kind=PEER_MESSAGE,.code=code,.count=kind},data,(uint32_t)bytes);
  return p->greeted?0:ENOLINK;
}
static void watch(int fd){
  struct kevent64_s e;EV_SET64(&e,(uint64_t)fd,EVFILT_READ,EV_ADD,0,0,EVENT_PEER,0,0);
  if(kevent64(events,&e,1,NULL,0,0,NULL))die("peer service events");
}
void peer_service_open(void){
  struct sockaddr_un addr=mesh_peer_address((uint64_t)getpid());
  local=socket(AF_UNIX,SOCK_STREAM,0);
  unlink(addr.sun_path);
  if(local<0 || bind(local,(struct sockaddr *)&addr,sizeof addr) || chmod(addr.sun_path,MESH_MODE) ||
     listen(local,SOMAXCONN) || fcntl(local,F_SETFL,O_NONBLOCK)<0)die("peer service socket");
  watch(local);
}
void peer_service_close(void){
  if(local>=0){struct sockaddr_un addr=mesh_peer_address((uint64_t)getpid());unlink(addr.sun_path);}
}
static struct job *lookup(uint32_t peer,uint64_t group,int create){
  struct job *j=jobs;
  while(j && (j->peer!=peer || j->group!=group))j=j->next;
  if(!j && create){
    j=calloc(1,sizeof *j);if(!j)return NULL;
    *j=(struct job){.next=jobs,.peer=peer,.group=group,.fd=-1,.expires=mesh_clock_ns()+MESH_PLATFORM_NS};
    pthread_mutex_init(&j->lock,NULL);j->end_in=&j->incoming;j->end_out=&j->outgoing;jobs=j;
  }
  return j;
}
static int enqueue(struct job *j,int outbound,const struct signal *events,uint32_t n){
  struct notice *m=malloc(sizeof *m+(size_t)n*sizeof *events);if(!m)return ENOMEM;
  m->next=NULL;m->count=n;memcpy(m->events,events,(size_t)n*sizeof *events);
  pthread_mutex_lock(&j->lock);
  if(outbound){*j->end_out=m;j->end_out=&m->next;}else{*j->end_in=m;j->end_in=&m->next;}
  pthread_mutex_unlock(&j->lock);if(outbound)wake();return 0;
}
static struct notice *dequeue(struct job *j,int out){
  struct notice *m;pthread_mutex_lock(&j->lock);
  if(out){m=j->outgoing;j->outgoing=NULL;j->end_out=&j->outgoing;}
  else{m=j->incoming;j->incoming=NULL;j->end_in=&j->incoming;}
  pthread_mutex_unlock(&j->lock);return m;
}
static int owned_range(uint64_t owner,uint64_t off,uint64_t bytes){
  if(!bytes || off<M->data_off || off>M->length || bytes>M->length-off)return 0;
  for(uint32_t i=0;i<M->rows;i++){
    struct mesh_buffer *b=mesh_buffers(M)+i;
    if(atomic_load(&b->owner)!=owner || atomic_load(&b->closed) || !b->pages)continue;
    uint64_t start=atomic_load(&mesh_page(M)[i].address),size=(uint64_t)b->pages*M->pgsz;
    if(off>=start && off-start<size && bytes<=size-(off-start))return 1;
  }
  return 0;
}
static int offer(struct job *j,struct mesh_peer_request req){
  struct mesh_client *client=mesh_client_at(M,req.owner);
  uint64_t other=0;
  if(!atomic_compare_exchange_strong(&client->peer_owner,&other,req.owner) && other!=req.owner)return ESTALE;
  j->pinned=1;
  if(atomic_load(&client->owner)!=req.owner || (req.records%8) || !req.count ||
     !owned_range(req.owner,req.records,(uint64_t)req.count*sizeof *j->records))return EINVAL;
  j->records=(void *)((char *)M+req.records);
  struct mesh_endpoint links[MESH_STREAMS];uint32_t n=mesh_endpoints(M,links,MESH_STREAMS),units=0;
  for(uint32_t i=0;i<req.count;i++){
    struct mesh_peer_transfer r=j->records[i];
    if(!r.units || r.units>UINT32_MAX-units || r.land>1 || r.spans%8 || r.completed%8 || r.ready%8 ||
      r.value==UINT64_MAX || !r.value || r.threshold==UINT64_MAX ||
      !owned_range(req.owner,r.completed,8) || (r.ready && !owned_range(req.owner,r.ready,8)) ||
      !owned_range(req.owner,r.spans,r.units*sizeof(struct mesh_dma_span)))return EINVAL;
    units+=(uint32_t)r.units;
  }
  struct peer_offer h={n,req.count,units,0};int error=peer_layout_size(h,&j->local_bytes);
  if(error || j->local_bytes>64u*1024*1024-sizeof(struct message))return EOVERFLOW;
  j->offer=malloc(j->local_bytes);if(!j->offer)return ENOMEM;
  memcpy(j->offer,&h,sizeof h);struct peer_layout l;
  if((error=peer_layout(j->offer,j->local_bytes,&l)))return error;
  memcpy(l.links,links,(size_t)n*sizeof *links);
  units=0;
  for(uint32_t i=0;i<req.count;i++){
    struct mesh_peer_transfer r=j->records[i];
    struct mesh_dma_span *spans=(void *)((char *)M+r.spans);uint64_t bytes=0;
    for(uint32_t k=0;k<r.units;k++){
      if(!owned_range(req.owner,spans[k].offset,spans[k].bytes) || spans[k].bytes>UINT64_MAX-bytes)return EINVAL;
      bytes+=spans[k].bytes;
    }
    l.row[i]=(struct peer_record){r.match,bytes,r.threshold?r.ready:0,r.threshold,units,(uint32_t)r.units,r.land?j->peer:M->node,i};
    memcpy(l.spans+units,spans,r.units*sizeof *spans);units+=r.units;
  }
  void *wire=malloc(j->local_bytes);if(!wire)return ENOMEM;memcpy(wire,j->offer,j->local_bytes);
  struct peer_layout w;peer_layout(wire,j->local_bytes,&w);
  for(uint32_t k=0;k<units;k++)w.spans[k].offset=(w.spans[k].offset-M->data_off)%MESH_DMA_EXTENT;
  error=peer_send(j,OFFER,wire,j->local_bytes);free(wire);return error;
}
static int prepare(struct job *j){
  struct peer_layout a,b;
  int error=peer_layout(j->offer,j->local_bytes,&a);
  if(!error)error=peer_layout(j->remote,j->remote_bytes,&b);
  if(!error)error=peer_plan_compile(&j->plan,M->node,j->peer,a,b,j->records,M->data_off,MESH_FRAME);
  if(error)return error;
  j->context=(struct mesh_ctx){.fd=-1,.len=(size_t)M->length};
  if((error=mesh_slot_take(&j->context,M)))return error;
  if(j->plan.count>(SIZE_MAX-8)/sizeof(struct mesh_dma_span))return EOVERFLOW;
  if((error=mesh_section_create(&j->context,8+j->plan.count*sizeof(struct mesh_dma_span),1,&j->table)))return error;
  uint64_t *start=mesh_section_address(&j->context,j->table,0);*start=UINT64_MAX-1;
  struct mesh_dma_span *table=(void *)(start+1);
  for(size_t i=0;i<j->plan.count;i++)table[i]=j->plan.piece[i].span;
  for(uint32_t e=0;e<j->plan.links;e++){
    struct channel *c=j->channel+e;
    if((error=mesh_stream_take(&j->context,j->plan.cable[e].wire,&c->stream)))return error;
    j->holds++;
    if((error=mesh_stream_view(&j->context,c->stream,&c->view)))return error;
  }
  return peer_send(j,ACCEPT,NULL,0);
}
static uint64_t read_word(uint64_t off){return atomic_load_explicit((_Atomic uint64_t *)((char *)M+off),memory_order_acquire);}
static void completed(uint64_t off,uint64_t value){
  _Atomic uint64_t *p=(void *)((char *)M+off);uint64_t old=atomic_load(p);
  while(old<value && !atomic_compare_exchange_weak_explicit(p,&old,value,memory_order_release,memory_order_relaxed)){}
}
struct commands {uint32_t *rows,count,next;};
static int reap_rows(struct job *j,uint32_t *done){
  for(uint32_t k=0;k<j->plan.rows;k++){
    struct peer_row *r=j->plan.row+k;if(r->done || r->next!=r->end)continue;
    size_t i=r->first;unsigned way=r->logical.land;
    for(;i<r->end;i++){
      struct peer_piece *p=j->plan.piece+i;struct mesh_stream_view *v=&j->channel[p->channel].view;
      uint64_t n=atomic_load_explicit(way?v->landed:v->sent,memory_order_acquire);
      if(n==UINT64_MAX)return ENOLINK;
      if(n<p->prefix)break;
    }
    if(i==r->end){r->done=1;(*done)++;completed(r->logical.completed,r->logical.value);}
  }
  return 0;
}
static int run(struct job *j){
  struct peer_plan *plan=&j->plan;uint32_t done=0;
  uint64_t table=(uint64_t)((char *)mesh_section_address(&j->context,j->table,0)-(char *)M);
  struct commands commands[2]={0};struct signal *signals=calloc(plan->rows,sizeof *signals);
  int error=0;
  for(uint32_t w=0;w<2;w++)if(!(commands[w].rows=calloc(plan->rows,sizeof(uint32_t))))error=ENOMEM;
  if(!signals)error=ENOMEM;
  while(!error && done<plan->rows && !atomic_load(&j->failed)){
    struct notice *note=dequeue(j,0);
    while(note){
      struct notice *next=note->next;
      for(uint32_t i=0;i<note->count;i++){
        struct signal s=note->events[i];
        if(s.row>=plan->rows){error=EPROTO;break;}
        struct peer_row *r=plan->row+s.row;
        if(s.kind==OPEN && !r->logical.land){r->remote_open=1;continue;}
        if(s.kind!=ISSUE || !r->logical.land || !plan->dynamic[1] || r->issued){error=EPROTO;break;}
        r->issued=1;commands[1].rows[commands[1].count++]=s.row;
      }
      free(note);note=next;
    }
    uint32_t n=0;
    for(uint32_t k=0;!error && k<plan->rows;k++){
      struct peer_row *r=plan->row+k;unsigned w=r->logical.land;
      if(r->issued || r->announced)continue;
      if(plan->dynamic[w]){
        uint64_t ready=r->logical.ready?read_word(r->logical.ready):0;
        if(ready==UINT64_MAX){error=EIO;break;}
        if(r->logical.ready && ready<r->logical.threshold)continue;
        if(w){r->announced=1;if(r->logical.ready && r->logical.threshold)signals[n++]=(struct signal){OPEN,k};continue;}
        if(!r->remote_open)continue;
        signals[n++]=(struct signal){ISSUE,k};
      }
      r->issued=1;commands[w].rows[commands[w].count++]=k;
    }
    if(n && !error)error=enqueue(j,1,signals,n);
    for(uint32_t w=0;w<2 && !error;w++){
      struct commands *q=commands+w;uint32_t blocked=0;
      for(uint32_t i=q->next;i<q->count;i++){
        struct peer_row *r=plan->row+q->rows[i];
        for(;r->next<r->end;r->next++){
          struct peer_piece *p=plan->piece+r->next;uint32_t e=p->channel;struct channel *c=j->channel+e;
          if(blocked&(1u<<e))break;
          uint64_t gate=r->logical.ready,base=r->logical.threshold;
          if(!gate || !base || plan->dynamic[w]){gate=table;base=1;}
          int status=mesh_stream_present(&j->context,c->stream,w,(struct mesh_entry){
            table+8+r->next*sizeof(struct mesh_dma_span),1,gate,w?base:base-1});
          if(status==EAGAIN){blocked|=1u<<e;break;}
          if(status){error=status;break;}
          p->prefix=++c->prefix[w];c->bytes[w]+=p->span.bytes;
        }
        if(r->next<r->end)for(size_t t=r->next;t<r->end;t++)blocked|=1u<<plan->piece[t].channel;
        if(error)break;
        if(i==q->next && r->next==r->end)q->next++;
      }
    }
    if(!error)error=reap_rows(j,&done);
    if(!error && done<plan->rows){
      for(uint32_t e=0;e<j->holds;e++)if(!mesh_stream_held(&j->context,j->channel[e].stream,plan->cable[e].wire)){error=ENOLINK;break;}
      __asm__ volatile("yield");
    }
  }
  free(commands[0].rows);free(commands[1].rows);free(signals);return error;
}
static void *work(void *arg){
  struct job *j=arg;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
  int error=run(j);if(error)atomic_store(&j->failed,error);
  atomic_store(&j->finished,1);wake();return NULL;
}
static void stop_job(struct job *j,int error){
  int prior=0;
  if(error && atomic_compare_exchange_strong(&j->failed,&prior,error) && j->fd>=0 && j->offer)
    peer_send(j,ABORT,&error,sizeof error);
}
void peer_message(uint32_t peer,uint64_t group,uint32_t kind,const void *data,uint32_t bytes){
  struct job *j=lookup(peer,group,kind==OFFER);if(!j)return;
  int error=0;
  if(kind==OFFER){
    struct peer_layout layout;
    if(j->remote || peer_layout((void *)data,bytes,&layout))error=EPROTO;
    else if(!(j->remote=malloc(bytes)))error=ENOMEM;
    else{memcpy(j->remote,data,bytes);j->remote_bytes=bytes;}
  }else if(kind==ACCEPT && !bytes)j->accepted=1;
  else if(kind==ISSUE && bytes && bytes%sizeof(struct signal)==0)
    error=enqueue(j,0,data,bytes/sizeof(struct signal));
  else if(kind==ABORT && bytes==sizeof(int)){memcpy(&error,data,bytes);if(!error)error=ECONNRESET;}
  else error=EPROTO;
  if(error)stop_job(j,error);
}
static void respond(int fd,int status){send(fd,&status,sizeof status,0);}
static void binding(int fd,struct mesh_peer_request q){
  pid_t pid=0;socklen_t size=sizeof pid;int error=0;
  if(getsockopt(fd,SOL_LOCAL,LOCAL_PEERPID,&pid,&size) || !q.owner || (uint32_t)pid!=(uint32_t)q.owner ||
     atomic_load(&mesh_client_at(M,q.owner)->owner)!=q.owner)error=ESTALE;
  else if(!reachable(q.peer))error=ENOTSUP;
  else if(q.op==MESH_PEER_INSTALL){
    struct job *j=lookup(q.peer,q.group,1);
    if(!j)error=ENOMEM;
    else if(j->fd>=0)error=EALREADY;
    else{
      j->fd=fd;j->owner=q.owner;
      error=offer(j,q);if(error)stop_job(j,error);
      return;
    }
  }else if(q.op!=MESH_PEER_QUERY)error=EINVAL;
  respond(fd,error);close(fd);
}
void peer_event(int fd){
  if(fd==local){
    while((fd=accept(local,NULL,NULL))>=0){
      struct client *c=calloc(1,sizeof *c);if(!c){close(fd);continue;}
      if(fcntl(fd,F_SETFL,O_NONBLOCK)<0){close(fd);free(c);continue;}
      c->fd=fd;c->next=clients;clients=c;watch(fd);
    }
    return;
  }
  struct client **at=&clients;
  while(*at && (*at)->fd!=fd)at=&(*at)->next;
  if(*at){
    struct client *c=*at;
    ssize_t n=recv(fd,(char *)&c->req+c->at,sizeof c->req-c->at,0);
    if(n>0){c->at+=(size_t)n;if(c->at<sizeof c->req)return;}
    else if(n<0 && (errno==EAGAIN || errno==EINTR))return;
    *at=c->next;if(n>0)binding(fd,c->req);else close(fd);free(c);return;
  }
  for(struct job *j=jobs;j;j=j->next)if(j->fd==fd){
    char ch;ssize_t n=recv(fd,&ch,1,0);
    if(n<0 && (errno==EINTR || errno==EAGAIN))return;
    j->closed=1;
    if(!atomic_load(&j->finished))stop_job(j,ECANCELED);
    struct kevent64_s e;EV_SET64(&e,(uint64_t)fd,EVFILT_READ,EV_DELETE,0,0,0,0,0);
    kevent64(events,&e,1,NULL,0,0,NULL);return;
  }
}
static void release(struct job *j){
  if(j->context.M){
    for(uint32_t e=0;e<j->holds;e++)mesh_stream_give(&j->context,j->channel[e].stream);
    if(j->table.count)mesh_lease_return(&j->context,j->table.first);
    mesh_retire(M,j->context.client);j->context.M=NULL;
  }
}
static int held(struct job *j){
  for(uint32_t e=0;e<j->holds;e++){
    uint64_t owner=atomic_load(&mesh_stream_at(M,j->channel[e].stream)->owner);
    if((owner&~MESH_STREAM_GIVEN)==j->context.client)return 1;
  }
  return 0;
}
void peer_turn(void){
  struct job **at=&jobs;
  while(*at){
    struct job *j=*at;
    if(!j->started && mesh_clock_ns()>=j->expires)stop_job(j,ETIMEDOUT);
    if(!reachable(j->peer))stop_job(j,ENOLINK);
    if(j->owner && atomic_load(&mesh_client_at(M,j->owner)->owner)!=j->owner){j->closed=1;stop_job(j,ESTALE);}
    if(j->fd>=0 && j->remote && j->offer && !j->prepared && !atomic_load(&j->failed)){
      int error=prepare(j);if(error)stop_job(j,error);else j->prepared=1;
    }
    if(j->prepared && j->accepted && !j->started && !atomic_load(&j->failed)){
      pthread_t thread;int error=pthread_create(&thread,NULL,work,j);
      if(error)stop_job(j,error);
      else{pthread_detach(thread);j->started=1;}
    }
    int failure=atomic_load(&j->failed);
    if(j->fd>=0 && !j->sent && (failure || j->started)){respond(j->fd,failure);j->sent=1;}
    struct notice *s=dequeue(j,1);
    while(s){
      struct notice *next=s->next;
      if(!failure){int error=peer_send(j,ISSUE,s->events,s->count*sizeof *s->events);if(error)stop_job(j,error);}
      free(s);s=next;
    }
    if((!j->started && failure) || atomic_load(&j->finished)){
      if(j->context.M){
        if(failure)peer_send(j,ABORT,&failure,sizeof failure);
        if(j->plan.row){
          for(uint32_t k=0;failure && k<j->plan.rows;k++)if(j->plan.row[k].logical.completed && !j->plan.row[k].done)
            completed(j->plan.row[k].logical.completed,UINT64_MAX);
          for(uint32_t e=0;e<j->holds;e++)
            say("peer %u group %llu wire %llu sent %llu received %llu bytes\n",j->peer,
                (unsigned long long)j->group,(unsigned long long)j->plan.cable[e].wire,
                (unsigned long long)j->channel[e].bytes[0],(unsigned long long)j->channel[e].bytes[1]);
        }
        release(j);
      }
      if((j->closed || j->fd<0) && !held(j)){
        if(j->fd>=0)close(j->fd);
        if(j->pinned){
          int another=0;for(struct job *i=jobs;i;i=i->next)another|=i!=j && i->pinned && i->owner==j->owner;
          if(!another){uint64_t owner=j->owner;atomic_compare_exchange_strong(&mesh_client_at(M,owner)->peer_owner,&owner,0);}
        }
        struct notice *m=dequeue(j,0);while(m){struct notice *next=m->next;free(m);m=next;}
        *at=j->next;free(j->plan.piece);free(j->plan.row);free(j->offer);free(j->remote);
        pthread_mutex_destroy(&j->lock);free(j);continue;
      }
    }
    at=&j->next;
  }
}
