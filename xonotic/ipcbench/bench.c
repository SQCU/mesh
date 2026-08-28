#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdatomic.h>
#include <errno.h>
#include <pthread.h>

#define NF 4096
#define NB (NF*4)
#define ITER 2000

static double now(void){struct timeval t;gettimeofday(&t,0);return t.tv_sec+t.tv_usec*1e-6;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}
static void report(const char*name,double*s,int n){qsort(s,n,sizeof(double),cmpd);
 printf("%-34s n=%d  med %8.1f us  p90 %8.1f us  p99 %8.1f us  max %9.1f us\n",
 name,n,s[n/2]*1e6,s[(int)(n*0.90)]*1e6,s[(int)(n*0.99)]*1e6,s[n-1]*1e6);}

static float out[NF], in[NF];

static int rdall(int fd,void*b,size_t n){size_t g=0;while(g<n){ssize_t r=read(fd,(char*)b+g,n-g);if(r<=0)return -1;g+=r;}return 0;}
static int wrall(int fd,const void*b,size_t n){size_t g=0;while(g<n){ssize_t r=write(fd,(const char*)b+g,n-g);if(r<=0)return -1;g+=r;}return 0;}

static void *sockserver(void *a){int fd=*(int*)a;char *buf=malloc(NB);
 for(int i=0;i<ITER+50;i++){if(rdall(fd,buf,NB))break;for(int j=0;j<NF;j++)((float*)buf)[j]*=2.0f;if(wrall(fd,buf,NB))break;}
 free(buf);return 0;}

struct shmring{_Atomic unsigned req,resp;float a[NF],b[NF];};
static struct shmring *ring;
static volatile int shmrun=1;
static void *shmserver(void *a){(void)a;unsigned last=0;
 for(;;){unsigned r=atomic_load_explicit(&ring->req,memory_order_acquire);
  if(!shmrun)return 0;
  if(r==last){continue;}
  last=r;for(int j=0;j<NF;j++)ring->b[j]=ring->a[j]*2.0f;
  atomic_store_explicit(&ring->resp,r,memory_order_release);}}

int main(void){
 for(int i=0;i<NF;i++)out[i]=i*0.5f;
 double *s=malloc(sizeof(double)*ITER);

 /* 1. unix socket stream */
 {int sv[2];socketpair(AF_UNIX,SOCK_STREAM,0,sv);pthread_t t;pthread_create(&t,0,sockserver,&sv[1]);
  for(int i=0;i<50;i++){wrall(sv[0],out,NB);rdall(sv[0],in,NB);}
  for(int i=0;i<ITER;i++){double t0=now();wrall(sv[0],out,NB);rdall(sv[0],in,NB);s[i]=now()-t0;}
  report("unix socketpair 16KiB RT",s,ITER);close(sv[0]);pthread_join(t,0);close(sv[1]);}

 /* 1b. real AF_UNIX path socket, separate process */
 {const char*p="/tmp/meshbench.sock";unlink(p);
  int ls=socket(AF_UNIX,SOCK_STREAM,0);struct sockaddr_un un={0};un.sun_family=AF_UNIX;strcpy(un.sun_path,p);
  bind(ls,(struct sockaddr*)&un,sizeof un);listen(ls,1);
  pid_t pid=fork();
  if(pid==0){int c=socket(AF_UNIX,SOCK_STREAM,0);while(connect(c,(struct sockaddr*)&un,sizeof un)<0)usleep(1000);
   char*buf=malloc(NB);for(int i=0;i<ITER+50;i++){if(rdall(c,buf,NB))break;for(int j=0;j<NF;j++)((float*)buf)[j]*=2.0f;if(wrall(c,buf,NB))break;}_exit(0);}
  int c=accept(ls,0,0);
  for(int i=0;i<50;i++){wrall(c,out,NB);rdall(c,in,NB);}
  for(int i=0;i<ITER;i++){double t0=now();wrall(c,out,NB);rdall(c,in,NB);s[i]=now()-t0;}
  report("AF_UNIX 2-process 16KiB RT",s,ITER);close(c);close(ls);unlink(p);}

 /* 2. shared memory spin */
 {int fd=shm_open("/meshbenchshm",O_CREAT|O_RDWR,0600);ftruncate(fd,sizeof(struct shmring));
  ring=mmap(0,sizeof(struct shmring),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  memset(ring,0,sizeof *ring);pthread_t t;pthread_create(&t,0,shmserver,0);
  unsigned seq=0;
  for(int i=0;i<200;i++){memcpy((void*)ring->a,out,NB);seq++;atomic_store_explicit(&ring->req,seq,memory_order_release);
   while(atomic_load_explicit(&ring->resp,memory_order_acquire)!=seq);memcpy(in,(void*)ring->b,NB);}
  for(int i=0;i<ITER;i++){double t0=now();
   memcpy((void*)ring->a,out,NB);seq++;atomic_store_explicit(&ring->req,seq,memory_order_release);
   while(atomic_load_explicit(&ring->resp,memory_order_acquire)!=seq);memcpy(in,(void*)ring->b,NB);
   s[i]=now()-t0;}
  report("shm mmap spin 16KiB RT",s,ITER);shmrun=0;seq++;atomic_store_explicit(&ring->req,seq,memory_order_release);pthread_join(t,0);
  shm_unlink("/meshbenchshm");}

 /* 2b. shm non-blocking poll: cost of publish+check only (what a game frame actually pays) */
 {int fd=shm_open("/meshbenchshm2",O_CREAT|O_RDWR,0600);ftruncate(fd,sizeof(struct shmring));
  struct shmring*r2=mmap(0,sizeof(struct shmring),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);memset(r2,0,sizeof*r2);
  unsigned seq=0;
  for(int i=0;i<ITER;i++){double t0=now();
   memcpy((void*)r2->a,out,NB);seq++;atomic_store_explicit(&r2->req,seq,memory_order_release);
   if(atomic_load_explicit(&r2->resp,memory_order_acquire)==seq)memcpy(in,(void*)r2->b,NB);
   s[i]=now()-t0;}
  report("shm publish+poll (nonblocking)",s,ITER);shm_unlink("/meshbenchshm2");}

 /* 3. file via write+rename+read */
 {int n=200;
  for(int i=0;i<n;i++){double t0=now();
   int fd=open("/tmp/meshbench.tmp",O_CREAT|O_WRONLY|O_TRUNC,0600);wrall(fd,out,NB);close(fd);
   rename("/tmp/meshbench.tmp","/tmp/meshbench.dat");
   int g=open("/tmp/meshbench.dat",O_RDONLY);rdall(g,in,NB);close(g);
   s[i]=now()-t0;}
  report("file write+rename+read 16KiB",s,n);unlink("/tmp/meshbench.dat");}

 /* 4. ascii cost that every string-based path pays */
 {char*buf=malloc(NF*16);int n=500;
  for(int i=0;i<n;i++){double t0=now();int o=0;
   for(int j=0;j<NF;j++)o+=snprintf(buf+o,16,"%.6g ",out[j]);
   const char*p=buf;for(int j=0;j<NF;j++)in[j]=strtof(p,(char**)&p);
   s[i]=now()-t0;}
  report("ascii encode+decode 4096 floats",s,n);free(buf);}

 printf("\npayload %d floats = %d bytes each way\n",NF,NB);
 return 0;}
