// A mesh application. It knows about pages and nothing below them: no queue
// pair, no memory region, no lkey, no wire header, no retransmission.
#include "mesh.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
static double now(void){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec+t.tv_usec/1e6;}
int main(int argc,char**argv){
  const char *name = argc>1?argv[1]:"/mesh0";
  int dst = argc>2?atoi(argv[2]):-1;          // -1 = receive only
  double secs = argc>3?atof(argv[3]):4.0;
  struct mesh M;
  for(int i=0;i<200 && mesh_attach(&M,name);i++) usleep(20000);
  if(!M.h){ fprintf(stderr,"no bridge at %s\n",name); return 1; }
  printf("attached %s: %u pages x %u B = %.2f%% of node, node=%u\n",
     name, mesh_pages(&M), mesh_pagesize(&M), mesh_pct_of_node(&M), mesh_node(&M));
  unsigned long long sent=0,got=0,bad=0; double t0=now();
  while(now()-t0 < secs){
    if(dst>=0){
      uint32_t p;
      while(!mesh_acquire(&M,&p)){
        unsigned char *b=mesh_page(&M,p)+24;      // payload begins after the header
        memset(b,0xA5,256);
        ((uint32_t*)b)[0]=(uint32_t)sent;
        if(mesh_send(&M,p,256,(uint16_t)dst)){ mesh_release(&M,p); break; }
        if(++sent%100000==0) break;
      }
    }
    uint32_t p,bytes; uint16_t from;
    while(!mesh_poll(&M,&p,&bytes,&from)){
      unsigned char *b=mesh_page(&M,p)+24;
      if(b[4]!=0xA5||b[255]!=0xA5) bad++;
      got++; mesh_release(&M,p);
    }
  }
  printf("sent=%llu received=%llu corrupt=%llu  (%.2f s)\n",sent,got,bad,now()-t0);
  mesh_detach(&M);
  return 0;
}
