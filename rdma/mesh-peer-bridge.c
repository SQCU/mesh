#include "mesh-bridge.h"
#include "mesh-peer-private.h"
#include "mesh-peer-plan.h"

enum { OFFER=1,ACCEPT,OPEN,ISSUE,ABORT,ADMITTED };
struct signal {uint32_t kind,row,channel,lane;uint64_t first,offset,count;};
struct notice {struct notice *next;uint32_t count;struct signal events[];};
struct pipe_stream {
  uint32_t stream,users;uint64_t wire,assigned[2];
  struct job *fast;
  _Atomic int failure;
  struct mesh_stream_view view;
};
static struct mesh_ctx physical;
static struct pipe_stream wires[MESH_STREAMS];
struct channel {struct pipe_stream *pipe;uint64_t bytes[2],requests[2];};
struct queue {size_t last,post,retire;};
static struct {uint32_t peer;long double bytes;} ledger[MESH_PORTS];
static pthread_mutex_t ledger_lock=PTHREAD_MUTEX_INITIALIZER;
struct job {
  struct job *next;
  uint32_t peer;uint64_t group,owner,expires;
  int fd,pinned,prepared,accepted,started,sent,reported,remote_admitted; _Atomic int closed;
  uint64_t lease_first,lease_end;
  _Atomic int failed,finished,admitted,mail;
  void *offer,*remote;size_t local_bytes,remote_bytes;
  struct mesh_peer_transfer *records;
  struct peer_plan plan;
  struct mesh_ctx context;struct mesh_section table;uint32_t holds;
  struct channel channel[MESH_STREAMS];struct queue queues[MESH_STREAMS][2][2];
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
  if(outbound){*j->end_out=m;j->end_out=&m->next;}else{*j->end_in=m;j->end_in=&m->next;atomic_store(&j->mail,1);}
  pthread_mutex_unlock(&j->lock);if(outbound)wake();return 0;
}
static struct notice *dequeue(struct job *j,int out){
  struct notice *m;pthread_mutex_lock(&j->lock);
  if(out){m=j->outgoing;j->outgoing=NULL;j->end_out=&j->outgoing;}
  else{m=j->incoming;j->incoming=NULL;j->end_in=&j->incoming;atomic_store(&j->mail,0);}
  pthread_mutex_unlock(&j->lock);return m;
}
static int owned_range(struct job *j,uint64_t off,uint64_t bytes){
  if(!bytes || off<M->data_off || off>M->length || bytes>M->length-off)return 0;
  if(off>=j->lease_first && off<j->lease_end && bytes<=j->lease_end-off)return 1;
  for(uint32_t i=0;i<M->rows;i++){
    struct mesh_buffer *b=mesh_buffers(M)+i;
    if(atomic_load(&b->owner)!=j->owner || atomic_load(&b->closed) || !b->pages)continue;
    uint64_t start=atomic_load(&mesh_page(M)[i].address),size=(uint64_t)b->pages*M->pgsz;
    if(off>=start && off-start<size && bytes<=size-(off-start)){j->lease_first=start;j->lease_end=start+size;return 1;}
  }
  return 0;
}
static int row_ready(struct mesh_peer_transfer);
static int offer(struct job *j,struct mesh_peer_request req){
  struct mesh_client *client=mesh_client_at(M,req.owner);
  uint64_t other=0;
  if(!atomic_compare_exchange_strong(&client->peer_owner,&other,req.owner) && other!=req.owner)return ESTALE;
  j->pinned=1;
  if(atomic_load(&client->owner)!=req.owner || (req.records%8) || !req.count ||
     !owned_range(j,req.records,(uint64_t)req.count*sizeof *j->records))return EINVAL;
  j->records=(void *)((char *)M+req.records);
  struct mesh_endpoint links[MESH_STREAMS];uint32_t n=mesh_endpoints(M,links,MESH_STREAMS),units=0;
  for(uint32_t i=0;i<n;i++)for(uint32_t t=0;t<MESH_STREAMS;t++)
    if(wires[t].users && wires[t].wire==links[i].wire && !atomic_load(&wires[t].failure))links[i].available=1;
  for(uint32_t i=0;i<req.count;i++){
    struct mesh_peer_transfer r=j->records[i];
    if(!r.units || r.units>UINT32_MAX-units || r.land>1 || r.spans%8 || r.completed%8 || r.ready%8 ||
      r.value==UINT64_MAX || !r.value || (!r.reserved && r.threshold==UINT64_MAX) ||
      !owned_range(j,r.completed,8) || (r.reserved && !r.ready) || (r.ready && !owned_range(j,r.ready,r.reserved?sizeof(struct mesh_peer_gate)*(uint64_t)r.reserved:8)) ||
      !owned_range(j,r.spans,r.units*sizeof(struct mesh_dma_span)))return EINVAL;
    const struct mesh_peer_gate *g=(const void *)((const char *)M+r.ready);
    for(uint32_t t=0;t<r.reserved;t++)if(g[t].word%8 || g[t].threshold==UINT64_MAX || !owned_range(j,g[t].word,8))return EINVAL;
    units+=(uint32_t)r.units;
  }
  struct peer_offer h={n,req.count,units,0};int error=peer_layout_size(h,&j->local_bytes);
  if(error || j->local_bytes>64u*1024*1024-sizeof(struct message))return EOVERFLOW;
  j->offer=calloc(1,j->local_bytes);if(!j->offer)return ENOMEM;
  memcpy(j->offer,&h,sizeof h);struct peer_layout l;
  if((error=peer_layout(j->offer,j->local_bytes,&l)))return error;
  memcpy(l.links,links,(size_t)n*sizeof *links);
  for(uint32_t i=0;i<n;i++)for(uint32_t index=0;index<M->streams;index++){
    struct mesh_stream *x=mesh_stream_at(M,index);
    if(atomic_load(&x->wire)!=links[i].wire)continue;
    for(unsigned w=0;w<2;w++)l.limit[i].bytes[w]=atomic_load(&x->peer_packet[w]);
    break;
  }
  units=0;
  for(uint32_t i=0;i<req.count;i++){
    struct mesh_peer_transfer r=j->records[i];
    struct mesh_dma_span *spans=(void *)((char *)M+r.spans);uint64_t bytes=0;
    for(uint32_t k=0;k<r.units;k++){
      if(!owned_range(j,spans[k].offset,spans[k].bytes) || spans[k].bytes>UINT64_MAX-bytes)return EINVAL;
      bytes+=spans[k].bytes;
    }
    int ready=row_ready(r);if(ready<0)return -ready;
    l.row[i]=(struct peer_record){r.match,bytes,!ready,0,units,(uint32_t)r.units,r.land?j->peer:M->node,i};
    memcpy(l.spans+units,spans,r.units*sizeof *spans);units+=r.units;
  }
  void *wire=malloc(j->local_bytes);if(!wire)return ENOMEM;memcpy(wire,j->offer,j->local_bytes);
  struct peer_layout w;peer_layout(wire,j->local_bytes,&w);
  for(uint32_t k=0;k<units;k++)w.spans[k].offset=(w.spans[k].offset-M->data_off)%MESH_DMA_EXTENT;
  error=peer_send(j,OFFER,wire,j->local_bytes);free(wire);return error;
}
struct completion { uint64_t word,value;uint32_t row; };
static int completion_order(const void *a,const void *b){
  const struct completion *x=a,*y=b;
  if(x->word!=y->word)return (x->word>y->word)-(x->word<y->word);
  return (x->value>y->value)-(x->value<y->value);
}
static int hold(uint64_t wire,struct pipe_stream **out){
  if(!physical.M){
    physical=(struct mesh_ctx){.fd=-1,.len=M->length};
    int error=mesh_slot_take(&physical,M);if(error)return error;
    atomic_store(&mesh_client_at(M,physical.client)->peer_stream,physical.client);
  }
  for(uint32_t i=0;i<MESH_STREAMS;i++)if(wires[i].users && wires[i].wire==wire){
    if(atomic_load(&wires[i].failure))return EIO;
    wires[i].users++;*out=wires+i;return 0;
  }
  uint32_t index=0;int error=mesh_stream_take(&physical,wire,&index);if(error)return error;
  struct pipe_stream *s=wires+index;
  if(s->users){mesh_stream_give(&physical,index);return EAGAIN;}
  *s=(struct pipe_stream){.stream=index,.users=1,.wire=wire};
  error=mesh_stream_view(&physical,index,&s->view);
  if(error){s->users=0;mesh_stream_give(&physical,index);return error;}
  struct mesh_stream *x=mesh_stream_at(M,index);
  for(unsigned w=0;w<2;w++)for(unsigned lane=0;lane<2;lane++){
    uint64_t position=atomic_load(&x->posts[w].posted[lane]);
    atomic_store(&x->peer_presented[w][lane],position);
    if(!w)s->assigned[lane]=position;
  }
  atomic_store_explicit(&x->peer_mode,physical.client,memory_order_release);*out=s;return 0;
}
static size_t record_tickets(struct job *j,uint32_t row,struct signal *out,int fast){
  struct peer_row *r=j->plan.row+row;size_t written=0;
  pthread_mutex_lock(&ledger_lock);
  uint32_t choice=0;long double finish=0;
  for(uint32_t e=0;e<j->plan.links;e++){
    uint32_t port=j->channel[e].pipe->stream/MESH_LINK_STREAMS;
    if(ledger[port].peer!=j->peer){ledger[port].peer=j->peer;ledger[port].bytes=0;}
    long double next=(ledger[port].bytes+r->bytes)/j->plan.cable[e].bandwidth;
    if(!e || next<finish){choice=e;finish=next;}
  }
  if(fast){
    struct pipe_stream *s=j->channel[choice].pipe;
    if(s->fast && s->fast!=j){pthread_mutex_unlock(&ledger_lock);return 0;}
    s->fast=j;r->fast=1;
  }
  if(r->bytes<=MESH_FRAME)for(size_t i=r->first;i<r->end;i++)j->plan.piece[i].channel=choice;
  for(size_t first=r->first,last;first<r->end;first=last){
    uint32_t e=j->plan.piece[first].channel;
    long double bytes=0;
    for(last=first;last<r->end && j->plan.piece[last].channel==e;last++)bytes+=j->plan.piece[last].span.bytes;
    ledger[j->channel[e].pipe->stream/MESH_LINK_STREAMS].bytes+=bytes;
    struct pipe_stream *s=j->channel[e].pipe;size_t count=last-first;
    unsigned lane=0;size_t split=count;
    if(!fast){
      if(s->fast)lane=1;
      else{
        struct mesh_stream *x=mesh_stream_at(M,s->stream);
        uint64_t a=s->assigned[0]-atomic_load(&x->posts[0].done[0]),b=s->assigned[1]-atomic_load(&x->posts[0].done[1]);
        lane=b<a;split=(count+1)/2;
      }
    }
    out[written++]=(struct signal){ISSUE,row,e,lane,s->assigned[lane],first-r->first,split};
    s->assigned[lane]+=split;
    if(split<count){lane^=1;out[written++]=(struct signal){ISSUE,row,e,lane,s->assigned[lane],first-r->first+split,count-split};s->assigned[lane]+=count-split;}
  }
  pthread_mutex_unlock(&ledger_lock);
  return written;
}
static int queue_ticket(struct job *j,struct signal signal){
  if(signal.row>=j->plan.rows || signal.channel>=j->plan.links || signal.lane>=2)return EPROTO;
  struct peer_row *r=j->plan.row+signal.row;struct queue *q=&j->queues[signal.channel][r->logical.land][signal.lane];
  if(!signal.count || signal.offset>=r->end-r->first || signal.count>r->end-r->first-signal.offset)return EPROTO;
  size_t matched=0;
  for(size_t i=r->first+signal.offset;i<r->first+signal.offset+signal.count;i++){
    struct peer_piece *p=j->plan.piece+i;
    if(r->bytes<=MESH_FRAME)p->channel=signal.channel;
    if(p->channel!=signal.channel)return EPROTO;
    if(p->prefix || signal.first>=UINT64_MAX-matched)return EPROTO;
    p->prefix=signal.first+(++matched);p->row=signal.row;
    if(q->last)j->plan.piece[q->last-1].next=i+1;
    q->last=i+1;
    if(!q->post)q->post=i+1;
    if(!q->retire)q->retire=i+1;
  }
  return matched?0:EPROTO;
}
static int prepare(struct job *j){
  struct peer_layout a,b;
  int error=peer_layout(j->offer,j->local_bytes,&a);
  if(!error)error=peer_layout(j->remote,j->remote_bytes,&b);
  if(!error)error=peer_plan_compile(&j->plan,M->node,j->peer,a,b,j->records,M->data_off,MESH_FRAME);
  if(error)return error;
  struct completion *order=calloc(j->plan.rows,sizeof *order);if(!order)return ENOMEM;
  for(uint32_t k=0;k<j->plan.rows;k++){
    struct mesh_peer_transfer r=j->plan.row[k].logical;order[k]=(struct completion){r.completed,r.value,k};
  }
  qsort(order,j->plan.rows,sizeof *order,completion_order);
  for(uint32_t k=1;k<j->plan.rows;k++)if(order[k].word==order[k-1].word){
    if(order[k].value==order[k-1].value){free(order);return EINVAL;}
    j->plan.row[order[k].row].prev=order[k-1].row;j->plan.row[order[k-1].row].succ=order[k].row;
  }
  free(order);
  j->context=(struct mesh_ctx){.fd=-1,.len=(size_t)M->length};
  if((error=mesh_slot_take(&j->context,M)))return error;

  size_t footer=(size_t)j->plan.rows*sizeof(struct mesh_peer_gate);
  if(j->plan.count>(SIZE_MAX-footer)/sizeof(struct mesh_dma_span))return EOVERFLOW;
  if((error=mesh_section_create(&j->context,j->plan.count*sizeof(struct mesh_dma_span)+footer,1,&j->table)))return error;
  struct mesh_dma_span *table=mesh_section_address(&j->context,j->table,0);
  for(size_t i=0;i<j->plan.count;i++)table[i]=j->plan.piece[i].span;
  struct mesh_peer_gate *records=(void *)(table+j->plan.count);
  for(uint32_t k=0;k<j->plan.rows;k++)
    records[k]=(struct mesh_peer_gate){j->plan.row[k].logical.completed,j->plan.row[k].logical.value};
  struct mesh_client *slot=mesh_client_at(M,j->context.client);
  slot->peer_words=(uint64_t)((char *)records-(char *)M);slot->peer_count=j->plan.rows;slot->peer_client=j->owner;
  atomic_store_explicit(&slot->peer_stream,j->context.client,memory_order_release);
  for(uint32_t e=0;e<j->plan.links;e++){
    if((error=hold(j->plan.cable[e].wire,&j->channel[e].pipe)))return error;
    j->holds++;
  }
  struct signal *assign=calloc(j->plan.count,sizeof *assign);if(!assign)return ENOMEM;
  size_t n=0;uint64_t word=0,threshold=0;int monotone=1;
  for(uint32_t k=0;k<j->plan.rows;k++){
    struct peer_row *r=j->plan.row+k;
    if(r->logical.land || r->primed || !r->remote_open || r->bytes>MESH_FRAME)continue;
    struct mesh_peer_transfer l=r->logical;
    if(l.reserved || (word && (l.ready!=word || l.threshold<threshold))){monotone=0;break;}
    word=l.ready;threshold=l.threshold;
  }
  for(uint32_t k=0;k<j->plan.rows;k++){
    struct peer_row *r=j->plan.row+k;if(r->logical.land)continue;
    if(r->primed)n+=record_tickets(j,k,assign+n,0);
    else if(monotone && r->remote_open && r->bytes<=MESH_FRAME)n+=record_tickets(j,k,assign+n,1);
  }
  for(size_t i=0;!error && i<n;i++)error=queue_ticket(j,assign[i]);
  if(!error)error=peer_send(j,ACCEPT,assign,n*sizeof *assign);
  free(assign);return error;
}
static uint64_t read_word(uint64_t off){return atomic_load_explicit((_Atomic uint64_t *)((char *)M+off),memory_order_acquire);}
static void completed(uint64_t off,uint64_t value){
  _Atomic uint64_t *p=(void *)((char *)M+off);uint64_t old=atomic_load(p);
  while(old<value && !atomic_compare_exchange_weak_explicit(p,&old,value,memory_order_release,memory_order_relaxed)){}
}
static int row_ready(struct mesh_peer_transfer r){
  if(!r.ready)return 1;
  if(!r.reserved){
    uint64_t v=read_word(r.ready);return v==UINT64_MAX?-EIO:v>=r.threshold;
  }
  const struct mesh_peer_gate *g=(const void *)((char *)M+r.ready);
  for(uint32_t i=0;i<r.reserved;i++){
    uint64_t v=read_word(g[i].word);if(v==UINT64_MAX)return -EIO;if(v<g[i].threshold)return 0;
  }
  return 1;
}
static void publish(struct job *j,uint32_t k){
  struct peer_row *r=j->plan.row+k;
  if(r->prev!=UINT32_MAX && !j->plan.row[r->prev].published)return;
  for(;;){
    completed(r->logical.completed,r->logical.value);r->published=1;
    if(r->succ==UINT32_MAX)break;
    r=j->plan.row+r->succ;if(!r->done)break;
  }
}
static int run(struct job *j){
  struct peer_plan *p=&j->plan;uint32_t done=0;
  uint64_t spans=(uint64_t)((char *)mesh_section_address(&j->context,j->table,0)-(char *)M);
  uint32_t *watch=calloc(p->rows,sizeof *watch),watching=0;
  struct signal *signals=calloc(p->count,sizeof *signals);
  int error=(!watch || !signals)?ENOMEM:0;
  for(uint32_t k=0;!error && k<p->rows;k++){
    struct peer_row *r=p->row+k;
    if(r->primed || r->fast)continue;
    if(!r->logical.land || !r->local_open)watch[watching++]=k;
  }
  while(!error && done<p->rows && !atomic_load(&j->failed)){
    if(atomic_load(&j->mail)){
      struct notice *note=dequeue(j,0);
      while(note){
        struct notice *next=note->next;
        for(uint32_t i=0;i<note->count && !error;i++){
          struct signal s=note->events[i];if(s.row>=p->rows){error=EPROTO;break;}
          struct peer_row *r=p->row+s.row;
          if(s.kind==OPEN && !r->logical.land){r->remote_open=1;continue;}
          if(s.kind!=ISSUE || !r->logical.land){error=EPROTO;break;}
          error=queue_ticket(j,s);
        }
        free(note);note=next;
      }
    }
    size_t n=0;uint32_t remaining=0;
    for(uint32_t i=0;i<watching && !error;i++){
      uint32_t k=watch[i];struct peer_row *r=p->row+k;
      int ready=row_ready(r->logical);if(ready<0){error=-ready;break;}
      if(!ready || (!r->logical.land && !r->remote_open)){watch[remaining++]=k;continue;}
      if(r->logical.land)signals[n++]=(struct signal){.kind=OPEN,.row=k};
      else{
        size_t first=n;n+=record_tickets(j,k,signals+n,0);
        for(size_t t=first;t<n && !error;t++)error=queue_ticket(j,signals[t]);
      }
    }
    watching=remaining;
    if(n && !error)error=enqueue(j,1,signals,(uint32_t)n);
    for(uint32_t e=0;!error && e<p->links;e++){
      struct channel *c=j->channel+e;struct pipe_stream *s=c->pipe;
      if(atomic_load(&s->failure) || !mesh_stream_held(&physical,s->stream,p->cable[e].wire)){error=ENOLINK;break;}
      struct mesh_stream *x=mesh_stream_at(M,s->stream);
      for(unsigned w=0;w<2 && !error;w++)for(unsigned lane=0;lane<2 && !error;lane++){
        struct queue *q=&j->queues[e][w][lane];
        while(q->post){
          size_t index=q->post-1;struct peer_piece *piece=p->piece+index;
          uint64_t last=atomic_load_explicit(&x->peer_presented[w][lane],memory_order_acquire);
          if(piece->prefix!=last+1){if(piece->prefix<=last)error=EPROTO;break;}
          uint64_t through=atomic_load_explicit(&x->posts[w].done[lane],memory_order_acquire);
          size_t capacity=MESH_STREAM_ENTRIES/2;
          if(last-through>=capacity)break;
          struct peer_row *r=p->row+piece->row;
          struct mesh_entry entry={spans+index*sizeof(struct mesh_dma_span),1,0,0};
          if(r->fast && !w){entry.word=r->logical.ready;entry.base=r->logical.threshold;}
          s->view.list[w][lane*capacity+(last%capacity)]=entry;
          atomic_store_explicit(&x->peer_presented[w][lane],piece->prefix,memory_order_release);
          q->post=piece->next;
        }
        if(!q->retire || q->retire==q->post)continue;
        uint64_t through=atomic_load_explicit(&x->posts[w].done[lane],memory_order_acquire);
        struct peer_row *next=p->row+p->piece[q->retire-1].row;
        if(next->fast && next->logical.ready && read_word(next->logical.ready)==UINT64_MAX){error=EIO;break;}
        while(q->retire && q->retire!=q->post){
          struct peer_piece *piece=p->piece+q->retire-1;
          if(through<piece->prefix)break;
          c->bytes[w]+=piece->span.bytes;c->requests[w]++;q->retire=piece->next;
          struct peer_row *r=p->row+piece->row;
          if(++r->retired==r->end-r->first){r->done=1;done++;publish(j,piece->row);}
        }
      }
    }
    if(!atomic_load(&j->admitted)){atomic_store(&j->admitted,1);wake();}
    if(done<p->rows && atomic_load(&j->closed))error=ECANCELED;
    __asm__ volatile("yield");
  }
  free(signals);free(watch);return error;
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
  }else if(kind==ACCEPT && bytes%sizeof(struct signal)==0){
    if(bytes)error=enqueue(j,0,data,bytes/sizeof(struct signal));
    j->accepted=1;
  }else if(kind==ADMITTED && !bytes)j->remote_admitted=1;
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
    if(!j->started)stop_job(j,ECANCELED);
    struct kevent64_s e;EV_SET64(&e,(uint64_t)fd,EVFILT_READ,EV_DELETE,0,0,0,0,0);
    kevent64(events,&e,1,NULL,0,0,NULL);return;
  }
}
static void release(struct job *j){
  if(j->context.M){
    mesh_client_at(M,j->context.client)->peer_count=0;
    for(uint32_t e=0;e<j->holds;e++){
      struct pipe_stream *s=j->channel[e].pipe;
      pthread_mutex_lock(&ledger_lock);if(s->fast==j)s->fast=NULL;pthread_mutex_unlock(&ledger_lock);
      for(unsigned w=0;w<2;w++)for(unsigned l=0;l<2;l++)if(j->queues[e][w][l].retire)atomic_store(&s->failure,1);
      if((!--s->users || atomic_load(&s->failure)) && mesh_stream_held(&physical,s->stream,s->wire))mesh_stream_give(&physical,s->stream);
    }
    if(j->table.count)mesh_lease_return(&j->context,j->table.first);
    mesh_retire(M,j->context.client);j->context.M=NULL;
  }
}
static int held(struct job *j){
  for(uint32_t e=0;e<j->holds;e++){
    struct pipe_stream *s=j->channel[e].pipe;
    if(atomic_load(&s->failure) && s->users)return 1;
    uint64_t owner=atomic_load(&mesh_stream_at(M,s->stream)->owner);
    if(owner==(physical.client|MESH_STREAM_GIVEN))return 1;
  }
  return 0;
}
void peer_turn(void){
  struct job **at=&jobs;
  while(*at){
    struct job *j=*at;
    if(!j->started && mesh_clock_ns()>=j->expires)stop_job(j,ETIMEDOUT);
    if(!atomic_load(&j->finished) && !reachable(j->peer))stop_job(j,ENOLINK);
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
    if(atomic_load(&j->admitted) && !j->reported){
      j->reported=1;int error=peer_send(j,ADMITTED,NULL,0);if(error)stop_job(j,error);
    }
    if(j->fd>=0 && !j->sent && (failure || (j->reported && j->remote_admitted))){respond(j->fd,failure);j->sent=1;}
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
            say("peer %u group %llu wire %llu sent %llu received %llu bytes; requests %llu sent %llu received; maximum %u/%u bytes\n",j->peer,
                (unsigned long long)j->group,(unsigned long long)j->plan.cable[e].wire,
                (unsigned long long)j->channel[e].bytes[0],(unsigned long long)j->channel[e].bytes[1],
                (unsigned long long)j->channel[e].requests[0],(unsigned long long)j->channel[e].requests[1],
                j->plan.packet[e][0],j->plan.packet[e][1]);
        }
        release(j);
      }
      if((j->closed || j->fd<0) && held(j))wake();
      else if(j->closed || j->fd<0){
        if(j->fd>=0){respond(j->fd,failure);close(j->fd);}
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


void peer_service_adopt(void){
  struct mesh_client *slots=(void *)((char *)M+M->client_off);
  for(uint32_t i=0;i<MESH_CLIENTS;i++){
    struct mesh_client *s=slots+i;uint64_t service=atomic_load(&s->peer_stream);
    if(!service || service!=atomic_load(&s->owner) || !s->peer_client ||
       atomic_load(&mesh_client_at(M,s->peer_client)->peer_owner)!=s->peer_client)continue;
    if(s->peer_words%8 || s->peer_words<M->data_off || s->peer_words>M->length ||
       s->peer_count>(M->length-s->peer_words)/sizeof(struct mesh_peer_gate))continue;
    const struct mesh_peer_gate *g=(const void *)((char *)M+s->peer_words);
    for(uint64_t k=0;k<s->peer_count;k++)
      if(g[k].word>=M->data_off && !(g[k].word%8) && g[k].word<=M->length-8 && read_word(g[k].word)<g[k].threshold)
        completed(g[k].word,UINT64_MAX);
  }
  for(uint32_t i=0;i<MESH_CLIENTS;i++){atomic_store(&slots[i].peer_owner,0);atomic_store(&slots[i].peer_stream,0);}
}
