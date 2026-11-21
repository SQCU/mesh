/* The switched bridges of N nodes over a timed wire (design/switched-multipath-dma.md "Tests"): tests/switched-world.h
   (the bridge's own functions on each node's region, the hardware's wire) on a fabric file (default
   tests/fabric-20261119.txt, the five-node cluster as measured), every byte real.

   Each row is the mean and spread of a collective's completion over MEASURE runs after LEARN, each run a collective
   of its own (its own key) on the bridges as the last left them, for the all-gather and the all-reduce, with the
   weights refreshed every REFRESH_US from the estimators (Thompson sampling, eta 0.5), and with them fixed at the
   unloaded shortest-delay routes (`fixed`).  Checked every run: every byte of every receive equal to what its program
   says (the sums exact in float on integers), every unit placed exactly once (each transfer's bitmap), every slot back
   in the pool at the end, and every draw logged in rdma/tests/route's record format (split.h struct route_draw): a
   token among its table's splits, a log-probability at most 0.

   usage: sim-switched [fabric [sizes...]] */
#include "switched-world.h"

#define LEARN 6
#define MEASURE 8

static char names[MESH_SWITCH_N][16];
static uint32_t nodes,ranks;
static struct sw_link fab[64];static uint32_t nlinks;
static double duplex=1;
static uint32_t node_of(const char *s){ for(uint32_t i=0;i<nodes;i++)if(!strcmp(names[i],s))return i; fprintf(stderr,"no node %s\n",s);exit(2); }
static void load(const char *path){
  FILE *f=fopen(path,"r");if(!f){perror(path);exit(2);}
  char line[256];
  while(fgets(line,sizeof line,f)){
    char a[16],b[16],kind[16];double rate,us;unsigned n=1;
    if(line[0]=='#')continue;
    if(sscanf(line,"node %15s %15s",a,kind)==2){snprintf(names[nodes],16,"%s",a);if(!strcmp(kind,"compute"))ranks|=1u<<nodes;nodes++;}
    else if(sscanf(line,"duplex %lf",&rate)==1)duplex=rate;
    else if(sscanf(line,"link %15s %15s %lf %lf %u",a,b,&rate,&us,&n)>=4)fab[nlinks++]=(struct sw_link){node_of(a),node_of(b),n,rate,us*1e3};
  }
  fclose(f);
}

static uint64_t draws,pieces,bad;
static void on_record(void *ctx,const struct mesh_switch_record *r){
  (void)ctx;
  if(r->kind==MESH_LOG_DRAW){
    draws++;
    const struct split sp=split_of(&r->draw,r->draw.token);
    if(r->logp>1e-6f || !sp.ok)bad++;
  }else if(r->kind==MESH_LOG_PIECE)pieces++;
}
struct stats { double mean,sd; };
static struct stats run(struct sw_world *w,int kind,uint64_t S,int fixed){
  w->fixed=fixed;
  sw_world_reset(w,NULL,(uint64_t)kind*1000003+S);
  double sum=0,sq=0;
  for(uint64_t e=0;e<LEARN+MEASURE;e++){
    const double t0=w->now;
    const uint32_t o=sw_op_start(w,kind,0,S,0,t0,e);
    assert(o!=UINT32_MAX);
    sw_pass_all(w);
    while(w->op[o].end<0){
      assert(w->events);   /* nothing in flight with work owed: a routing that cannot finish */
      sw_run_until(w,w->heap[0].t+1e-9);
    }
    const double T=w->op[o].end-t0;
    sw_run_until(w,INFINITY);   /* the wire drained (what still crosses is a SEND whose every destination holds it) */
    if(!sw_world_check(w,o,e))exit(4);
    for(uint32_t i=0;i<w->nodes;i++)assert(!w->node[i].nheld);
    sw_op_end(w,o);
    if(e>=LEARN){sum+=T;sq+=T*T;}
  }
  if(!sw_world_slots_back(w))exit(5);
  return (struct stats){sum/MEASURE,sqrt(fmax(0,sq/MEASURE-(sum/MEASURE)*(sum/MEASURE)))};
}
int main(int argc,char **argv){
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  struct sw_world *w=sw_world_new(nodes,ranks,fab,nlinks,duplex,2048,160,1);
  if(getenv("COPY_GBPS"))w->copy_gbps=atof(getenv("COPY_GBPS"));
  if(getenv("REFRESH_US"))w->refresh_ns=atof(getenv("REFRESH_US"))*1e3;
  w->on_record=on_record;
  static const double rows[]={2,8,32,128,1024};
  uint64_t sizes[16];int n=0;
  for(int i=2;i<argc && n<16;i++)sizes[n++]=strtoull(argv[i],NULL,10);
  if(!n)for(size_t i=0;i<sizeof rows/sizeof *rows;i++)sizes[n++]=(uint64_t)(rows[i]*1536*2);
  printf("%-10s %9s %20s %20s\n","","S bytes","fixed","switched");
  for(int k=0;k<2;k++)for(int i=0;i<n;i++){
    const int kind=k?SW_ALLREDUCE:SW_ALLGATHER;
    const struct stats f=run(w,kind,sizes[i],1),s=run(w,kind,sizes[i],0);
    printf("%-10s %9llu %10.2f+-%6.2fus %10.2f+-%6.2fus\n",k?"all-reduce":"all-gather",(unsigned long long)sizes[i],f.mean/1e3,f.sd/1e3,s.mean/1e3,s.sd/1e3);
    fflush(stdout);
  }
  if(bad){fprintf(stderr,"sim-switched: %llu draws outside their tables\n",(unsigned long long)bad);return 6;}
  printf("sim-switched: %llu draws logged, %llu pieces logged; every byte placed exactly once, every slot back: passed\n",
         (unsigned long long)draws,(unsigned long long)pieces);
  sw_world_free(w);
  return 0;
}
