#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <sys/time.h>
typedef struct { _Atomic unsigned req, done; unsigned nfloats, magic; float f[1]; } region_t;
int main(int argc,char**argv){
 unsigned n = argc>1?(unsigned)atoi(argv[1]):8192;
 size_t bytes = sizeof(region_t)+(size_t)n*4;
 int fd=shm_open("/meshdemo",O_CREAT|O_RDWR,0600);ftruncate(fd,bytes);
 region_t*r=mmap(0,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
 unsigned last=atomic_load(&r->req);
 struct timeval t;gettimeofday(&t,0);double deadline=t.tv_sec+t.tv_usec*1e-6+120.0;
 for(;;){
  unsigned q=atomic_load_explicit(&r->req,memory_order_acquire);
  if(q!=last){last=q;
   unsigned half=r->nfloats/2;
   for(unsigned i=0;i<half;i++)r->f[half+i]=r->f[i]*2.0f+1.0f;
   atomic_store_explicit(&r->done,q,memory_order_release);}
  gettimeofday(&t,0);if(t.tv_sec+t.tv_usec*1e-6>deadline)break;}
 munmap(r,bytes);shm_unlink("/meshdemo");return 0;}
