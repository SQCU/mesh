/* Algorithmic profile of the stream plan: brute-force check on small cases, the real FFN geometry under span launches, and the plan's own cost. */
#include "mesh-stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <mach/mach_time.h>
static double ns(uint64_t t){ static mach_timebase_info_data_t i; if(!i.denom) mach_timebase_info(&i); return (double)t*i.numer/i.denom; }

static double best_ms; static uint32_t best_order[2][MESH_STREAM_TILES]; static unsigned long tried;
static void permute(const struct mesh_stream *s,uint32_t order[2][MESH_STREAM_TILES],int n,uint32_t k,struct mesh_stream_plan *p){
  if(k==s->tiles){
    if(n==0){ permute(s,order,1,0,p); return; }
    tried++;
    if(mesh_stream_simulate(s,(const uint32_t(*)[MESH_STREAM_TILES])order,p)) return;
    if(p->peak_landed[0]>s->window || p->peak_landed[1]>s->window) return;
    if(p->makespan_ns<best_ms){ best_ms=p->makespan_ns; memcpy(best_order,order,sizeof best_order); }
    return;
  }
  for(uint32_t i=k;i<s->tiles;i++){
    uint32_t t=order[n][k]; order[n][k]=order[n][i]; order[n][i]=t;
    permute(s,order,n,k+1,p);
    t=order[n][k]; order[n][k]=order[n][i]; order[n][i]=t;
  }
}
/* Every tile order of both nodes (runs grouped into spans) against the canonical peer-then-own order. */
static void brute(const char *label,struct mesh_stream s){
  struct mesh_stream_plan plan,p; memset(&p,0,sizeof p);
  int e=mesh_stream_plan(&s,&plan);
  uint32_t order[2][MESH_STREAM_TILES];
  for(int n=0;n<2;n++) for(uint32_t t=0;t<s.tiles;t++) order[n][t]=t;
  best_ms=1.0/0.0; tried=0; permute(&s,order,0,0,&p);
  printf("%-52s plan %s makespan %.1f us | brute optimum %.1f us over %lu orders | gap %.2f%%\n",label,e?strerror(e):"ok",
    e?0:plan.makespan_ns/1e3,best_ms/1e3,tried,e?0:100*(plan.makespan_ns-best_ms)/best_ms);
  if(!e && plan.makespan_ns>best_ms*1.000001){
    printf("  optimum order: n0"); for(uint32_t c=0;c<s.tiles;c++) printf(" %u",best_order[0][c]);
    printf(" | n1"); for(uint32_t c=0;c<s.tiles;c++) printf(" %u",best_order[1][c]); printf("\n");
  }
}
static void show(const char *label,struct mesh_stream s,uint32_t tile_rows){
  struct mesh_stream_plan plan; int e=mesh_stream_plan(&s,&plan);
  if(e){ printf("%-58s %s\n",label,strerror(e)); return; }
  printf("%-58s makespan %6.2f ms  idle M5 %4.1f%% M4 %4.1f%%  peak %2u/%2u of %u  launches %u/%u\n",label,plan.makespan_ns/1e6,
    100*plan.idle[0],100*plan.idle[1],plan.peak_landed[0],plan.peak_landed[1],s.window,plan.launches[0],plan.launches[1]);
  for(int n=0;n<2;n++){
    printf("  %s spans (rows):",n?"M4":"M5"); for(uint32_t i=0;i<plan.launches[n];i++) printf(" %u",mesh_stream_launch_tiles(n,i)*tile_rows);
    printf("  | idle ms: fill %.2f window %.2f peer %.2f drain %.2f\n",plan.idle_parts[n][0]/1e6,plan.idle_parts[n][1]/1e6,plan.idle_parts[n][2]/1e6,plan.idle_parts[n][3]/1e6);
  }
}
int main(void){
  /* (a) all tile orders of both nodes, 6 tiles -> 518,400. */
  struct mesh_stream tiny={.tiles=6,.owner_tiles={3,3},.window=2,.gathers_held=1,.transfer_ns=100,.input_ns=0,.node={{1,300,50,20},{1,120,0,40}}};
  brute("6 tiles, window 2, unequal producers",tiny);
  tiny.window=6; brute("6 tiles, window 6",tiny);
  tiny.input_ns=100; brute("6 tiles, window 6, input streams at link rate",tiny);
  tiny.input_ns=0; tiny.node[1].produce_ns=300; tiny.window=3; brute("6 tiles, window 3, equal producers",tiny);
  tiny.transfer_ns=400; brute("6 tiles, window 3, link slower than compute",tiny);
  struct mesh_stream mixed={.tiles=6,.owner_tiles={2,4},.window=3,.gathers_held=0,.transfer_ns=100,.input_ns=0,.node={{1,300,50,20},{1,120,0,40}}};
  brute("6 tiles, 2/4 ownership, returns freed",mixed);
  /* (b) the real FFN geometry: 4096 rows, 128-row tiles (3840 fp16 columns in a 61-page block), window 4095/244. */
  const double page=16384,block=61,rate=9.41e9,transfer=block*page/rate*1e9;
  const uint32_t tile=mesh_stream_tile(61,16384,3840,2,128);
  printf("\ntile = %u rows; transfer %.1f us per block; window %u\n",tile,transfer/1e3,4095u/244u);
  /* Measured: M5 2.23 us per neuron per 4096-row batch (+20 us per launch), M4 ANE 6.23 us (launch inside the measurement).
     Reduce: 3 reads + 1 write of 128x3840 fp16 at 200 (M5) / 100 (M4) GB/s, plus the launch overhead. */
  double reduce_bytes=4.0*128*3840*2, reduce_m5=reduce_bytes/200e9*1e9, reduce_m4=reduce_bytes/100e9*1e9;
  printf("reduce per tile: M5 %.1f us, M4 %.1f us (bandwidth estimate)\n",reduce_m5/1e3,reduce_m4/1e3);
  for(int split=0;split<2;split++){
    double m5n=split?11264:11904, m4n=split?4096:3456, p5=2.23e3*m5n/32, p4=6.23e3*m4n/32, slow=p5>p4?p5:p4;
    uint32_t spans[64],n=mesh_stream_spans(2048,128,slow/transfer,spans,64);
    printf("\nneuron split M5 %.0f / M4 %.0f: produce per tile M5 %.0f us, M4 %.0f us; rho = %.2f; owner region 2048 rows -> %u spans:",m5n,m4n,p5/1e3,p4/1e3,slow/transfer,n);
    for(uint32_t i=0;i<n;i++) printf(" %u",spans[i]); printf("\n");
    struct mesh_stream s={.tiles=32,.owner_tiles={16,16},.window=16,.gathers_held=1,.transfer_ns=transfer,.input_ns=0,
      .node={{1,p5,20e3,reduce_m5},{1,p4,0,reduce_m4}}};
    show("4096 rows, returns held, input present",s,128);
    s.gathers_held=0; show("4096 rows, returns freed on landing, input present",s,128);
    s.gathers_held=1; s.input_ns=transfer; show("4096 rows, returns held, input arrives at link rate",s,128);
    struct mesh_stream s1={.tiles=8,.owner_tiles={4,4},.window=16,.gathers_held=1,.transfer_ns=transfer,.input_ns=0,
      .node={{1,p5*4,20e3,reduce_m5},{1,p4*4,0,reduce_m4}}};
    show("1024 rows (8 tiles), returns held, input present",s1,128);
  }
  /* (c) cost of the plan at configuration. */
  struct mesh_stream s={.tiles=32,.owner_tiles={16,16},.window=16,.gathers_held=1,.transfer_ns=transfer,.input_ns=0,
    .node={{1,2.23e3*11904/32,20e3,reduce_m5},{1,6.23e3*3456/32,0,reduce_m4}}};
  struct mesh_stream_plan plan; uint64_t t0=mach_absolute_time(); for(int i=0;i<1000;i++) mesh_stream_plan(&s,&plan);
  printf("\nmesh_stream_plan cost, 32 tiles: %.1f us\n",ns(mach_absolute_time()-t0)/1000/1e3);
  return 0;
}
