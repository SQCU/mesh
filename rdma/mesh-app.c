// A mesh application, using the whole API: mesh_open, mesh_write, mesh_read.
#include "mesh.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
static double now(void){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec+t.tv_usec/1e6;}

int main(int argc,char**argv){
  double gb   = argc>1?atof(argv[1]):1.0;    // how much mesh memory to map
  int    dst  = argc>2?atoi(argv[2]):-1;     // -1 = receive only
  double secs = argc>3?atof(argv[3]):5.0;

  size_t stride, usable;
  void *p = mesh_open((size_t)(gb*1e9), &stride, &usable);
  if(!p){ fprintf(stderr,"mesh_open(%.1f GB) failed\n",gb); return 1; }
  size_t slots = (size_t)(gb*1e9)/usable;
  printf("mapped %.2f GB: %zu slots of %zu usable bytes (stride %zu)\n",gb,slots,usable,stride);

  if(dst >= 0){
    for(size_t i=0;i<slots;i++){                       // fill it once
      unsigned char *s=(unsigned char*)p+i*stride;
      memset(s,0xA5,usable); ((uint32_t*)s)[0]=(uint32_t)i;
    }
    printf("filled %.2f GB, sending to node %d\n",gb,dst);
  }

  unsigned long long wrote=0,got=0,bad=0; double t0=now();
  size_t cursor=0;
  while(now()-t0 < secs){
    if(dst>=0){
      if(cursor >= slots*usable) cursor=0;              // stream it repeatedly
      size_t n = mesh_write((char*)p + (cursor/usable)*stride, slots*usable-cursor, dst);
      cursor += n; wrote += n;
    }
    void *q; int from; size_t n;
    while((n = mesh_read(&q,&from))){
      unsigned char *b=q;
      if(b[4]!=0xA5 || b[n-1]!=0xA5) bad++;
      got += n;
    }
  }
  double dt=now()-t0;
  printf("wrote=%.2f GB read=%.2f GB corrupt=%llu  %.2f Gbit/s payload\n",
     wrote/1e9, got/1e9, bad, (dst>=0?wrote:got)*8.0/dt/1e9);
  return 0;
}
