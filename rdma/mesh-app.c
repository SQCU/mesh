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
  const uint32_t PAY = mesh_pagesize(&M) - 24;   // a whole page, less the header
  unsigned long long sent=0,got=0,bad=0; double t0=now();
  int spin=0;
  for(;;){
    if(++spin >= 4096){ spin=0; if(now()-t0 >= secs) break; }
    int work=0;
    if(dst>=0){
      uint32_t p;
      for(int k=0;k<256 && !mesh_acquire(&M,&p);k++){
        unsigned char *b=(unsigned char*)mesh_page(&M,p)+24;
        memset(b,0xA5,PAY);                        // fill the page we are paying to send
        ((uint32_t*)b)[0]=(uint32_t)sent;
        if(mesh_send(&M,p,PAY,(uint16_t)dst)){ mesh_release(&M,p); break; }
        sent++; work=1;
      }
    }
    uint32_t p,bytes; uint16_t from;
    for(int k=0;k<256 && !mesh_poll(&M,&p,&bytes,&from);k++){
      unsigned char *b=(unsigned char*)mesh_page(&M,p)+24;
      if(b[4]!=0xA5 || b[bytes-1]!=0xA5) bad++;
      got++; mesh_release(&M,p); work=1;
    }
    if(!work && now()-t0 >= secs) break;
  }
  printf("sent=%llu received=%llu corrupt=%llu  (%.2f s)\n",sent,got,bad,now()-t0);
  mesh_detach(&M);
  return 0;
}
