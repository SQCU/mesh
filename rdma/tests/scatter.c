/* mesh_scatter_schedule on the Studios' ring with the Mac mini (ranks 0 A, 1 B, 2 D, 3 C, 4 M; A-B 3 cables, B-D 2,
   D-C 4, C-A 1, M-A 1, M-B 1, M-C 1): the all-gather of S bytes a compute rank (M a relay: no bytes, no destination)
   at the decode's and the prefill's sizes.  Checks every schedule: each destination holds every byte of every group,
   a piece leaves a node only after it landed there, a link's pieces do not overlap.  Prints the scheduled completion
   T(S), the lower bound and the schedule of each source's shortest tree (mesh_trees_shortest, the same link model:
   its pieces in order down its tree), and the solve's time.  Rates (bytes a ns per direction, a pair's measured rate
   shared by its cables) are ARGV's or the defaults below: sizes-matrix.txt's GPU fused all-gather per direction (AB
   12.0, CD 13.6, BD 9.9, CA 6.4 GB/s; the Mini's single cables as CA's), duplex 0.93 of alone, 2.5 us a hop, 0.3 us a
   piece.  usage: scatter [page] */
#include "mesh-plan.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { R = 5, PAIRS = 7 };
static const uint32_t pair[PAIRS][2] = {{0,1},{1,2},{2,3},{3,0},{4,0},{4,1},{4,3}};
static const uint32_t cables[PAIRS] = {3,2,4,1,1,1,1};
static const double gbs[PAIRS] = {12.0,9.9,13.6,6.4,6.4,6.4,6.4};

static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }

static void check(const struct mesh_scatter_link *link,uint32_t links,const struct mesh_scatter_group *group,uint32_t groups,
                  const struct mesh_scatter_send *send,uint32_t count){
  (void)links;
  for(uint32_t g=0;g<groups;g++)for(uint32_t d=0;d<R;d++){
    if(!(group[g].to>>d&1) || d==group[g].source)continue;
    uint64_t got=0;
    for(uint32_t k=0;k<count;k++)if(send[k].group==g && link[send[k].link].to==d)got+=send[k].bytes;
    assert(got==group[g].bytes);   /* every byte once */
  }
  for(uint32_t k=0;k<count;k++){
    const struct mesh_scatter_send *s=send+k;
    const uint32_t u=link[s->link].from;
    if(u!=group[s->group].source){
      int held=0;
      for(uint32_t j=0;j<count;j++)held|=send[j].group==s->group && send[j].first==s->first && link[send[j].link].to==u && send[j].arrive<=s->start;
      assert(held);
    }
    for(uint32_t j=k+1;j<count;j++)if(send[j].link==s->link)assert(send[j].start>=s->end || send[j].end<=s->start);
  }
}

/* the shortest trees' schedule in the same model: each source's pieces in order down its tree, breadth first, each
   hop on its pair's least busy cable */
static double trees_time(const struct mesh_scatter_link *link,uint32_t links,const uint32_t *parent,uint64_t bytes,uint64_t q){
  double free_at[64]={0},makespan=0;
  const uint64_t pieces=(bytes+q-1)/q;
  double hold[4][R];
  for(uint64_t k=0;k<pieces;k++){
    const uint64_t b=k+1<pieces?q:bytes-k*q;
    for(uint32_t s=0;s<4;s++){
      for(uint32_t v=0;v<R;v++)hold[s][v]=v==s?0:INFINITY;
      for(uint32_t depth=1;depth<R;depth++)for(uint32_t v=0;v<4;v++){
        const uint32_t *p=parent+(size_t)s*R;
        uint32_t at=v,d=0;while(at!=s && d<R){at=p[at];d++;}
        if(d!=depth || v==s)continue;
        const uint32_t u=p[v];
        uint32_t best=links;double start=0;
        for(uint32_t l=0;l<links;l++)if(link[l].from==u && link[l].to==v){
          const double st=hold[s][u]>free_at[l]?hold[s][u]:free_at[l];
          if(best==links || st<start){best=l;start=st;}
        }
        assert(best<links);
        int busy=0;for(uint32_t r=0;r<links;r++)busy|=link[r].from==v && link[r].to==u && free_at[r]>start;
        const double end=start+(double)b/(busy?link[best].duplex:link[best].alone)+link[best].piece;
        free_at[best]=end;hold[s][v]=end+link[best].latency;
        if(hold[s][v]>makespan)makespan=hold[s][v];
      }
    }
  }
  return makespan;
}

int main(int argc,char **argv){
  const uint64_t page=argc>1?strtoull(argv[1],NULL,0):16384;
  struct mesh_scatter_link link[64];uint32_t links=0;
  uint32_t ml[16][2];uint64_t mb[16];uint32_t L=0;
  for(uint32_t p=0;p<PAIRS;p++)for(uint32_t c=0;c<cables[p];c++)for(int w=0;w<2;w++){
    const double rate=gbs[p]/cables[p];
    link[links++]=(struct mesh_scatter_link){pair[p][w],pair[p][1-w],rate/0.93,rate,2500,300};
  }
  for(uint32_t p=0;p<PAIRS;p++){ml[L][0]=pair[p][0];ml[L][1]=pair[p][1];mb[L]=(uint64_t)(gbs[p]*8e9);L++;}
  /* the compute ranks' map (M a member: the trees may cross it, mesh_trees_shortest never does here) */
  const struct mesh_link_map map={.kind=MESH_LINKS_GRAPH,.nodes=R,.links=L,.link=ml,.bandwidth=mb};
  uint32_t root[R],parent[R*R];double weight[R];
  assert(mesh_trees_shortest(&map,root,parent,weight)==R);
  static const uint64_t sizes[]={6144,24576,98304,393216,1572864,6291456,25165824};
  printf("%10s %8s %12s %12s %12s %8s %10s\n","S bytes","piece","T sched us","bound us","T trees us","sends","solve us");
  for(size_t i=0;i<sizeof sizes/sizeof *sizes;i++){
    struct mesh_scatter_group group[4];
    for(uint32_t s=0;s<4;s++)group[s]=(struct mesh_scatter_group){s,s*sizes[i],sizes[i],0xF & ~(1ull<<s)};
    struct mesh_scatter_send *send=NULL;uint32_t count=0;uint64_t q=0;double T=0,bound=0;
    const double t0=now_ns();
    assert(!mesh_scatter_schedule(R,link,links,group,4,page,&send,&count,&q,&T,&bound));
    const double solve=now_ns()-t0;
    check(link,links,group,4,send,count);
    assert(T>=bound*(1-1e-12));
    uint32_t through=0;for(uint32_t k=0;k<count;k++)through+=link[send[k].link].from==4;
    const uint64_t tq=page<sizes[i]?page:sizes[i];
    printf("%10llu %8llu %12.2f %12.2f %12.2f %8u %10.1f  (via M %u)\n",(unsigned long long)sizes[i],(unsigned long long)q,T/1e3,bound/1e3,
           trees_time(link,links,parent,sizes[i],tq)/1e3,count,solve/1e3,through);
    free(send);
  }
  return 0;
}
