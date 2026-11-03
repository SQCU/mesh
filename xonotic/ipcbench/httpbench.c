#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/time.h>
#include <curl/curl.h>
#define NF 4096
#define NB (NF*4)
#define ITER 500
static double now(void){struct timeval t;gettimeofday(&t,0);return t.tv_sec+t.tv_usec*1e-6;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}
static void report(const char*n,double*s,int c){qsort(s,c,sizeof(double),cmpd);
 printf("%-34s n=%d  med %8.1f us  p90 %8.1f us  p99 %8.1f us  max %9.1f us\n",n,c,s[c/2]*1e6,s[(int)(c*0.9)]*1e6,s[(int)(c*0.99)]*1e6,s[c-1]*1e6);}
static int port;
static char *body;
static void *srv(void*a){int ls=*(int*)a;
 for(;;){int c=accept(ls,0,0);if(c<0)return 0;int one=1;setsockopt(c,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one);
  char req[65536];
  for(;;){ssize_t n=recv(c,req,sizeof req,0);if(n<=0)break;
   char hdr[256];int hl=snprintf(hdr,sizeof hdr,"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %d\r\n\r\n",NB);
   send(c,hdr,hl,0);send(c,body,NB,0);}
  close(c);}}
static size_t sink(void*p,size_t s,size_t n,void*u){(void)p;(void)u;return s*n;}
int main(void){
 body=malloc(NB);memset(body,'a',NB);
 int ls=socket(AF_INET,SOCK_STREAM,0);int one=1;setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
 struct sockaddr_in sa={0};sa.sin_family=AF_INET;sa.sin_addr.s_addr=htonl(INADDR_LOOPBACK);sa.sin_port=0;
 bind(ls,(struct sockaddr*)&sa,sizeof sa);listen(ls,16);
 socklen_t sl=sizeof sa;getsockname(ls,(struct sockaddr*)&sa,&sl);port=ntohs(sa.sin_port);
 pthread_t t;pthread_create(&t,0,srv,&ls);
 char url[128];snprintf(url,sizeof url,"http://127.0.0.1:%d/solve",port);
 char *post=malloc(NB);memset(post,'b',NB);
 double *s=malloc(sizeof(double)*ITER);
 curl_global_init(CURL_GLOBAL_ALL);

 CURL *h=curl_easy_init();
 curl_easy_setopt(h,CURLOPT_URL,url);curl_easy_setopt(h,CURLOPT_POSTFIELDS,post);
 curl_easy_setopt(h,CURLOPT_POSTFIELDSIZE,(long)NB);curl_easy_setopt(h,CURLOPT_WRITEFUNCTION,sink);
 for(int i=0;i<20;i++)curl_easy_perform(h);
 for(int i=0;i<ITER;i++){double t0=now();curl_easy_perform(h);s[i]=now()-t0;}
 report("curl POST reused handle 16KiB",s,ITER);
 curl_easy_cleanup(h);

 for(int i=0;i<ITER;i++){double t0=now();
  CURL*e=curl_easy_init();curl_easy_setopt(e,CURLOPT_URL,url);curl_easy_setopt(e,CURLOPT_POSTFIELDS,post);
  curl_easy_setopt(e,CURLOPT_POSTFIELDSIZE,(long)NB);curl_easy_setopt(e,CURLOPT_WRITEFUNCTION,sink);
  curl_easy_perform(e);curl_easy_cleanup(e);s[i]=now()-t0;}
 report("curl POST fresh handle (uri_get)",s,ITER);
 printf("\nuri_get creates a fresh easy handle per call and never reuses connections.\n");
 return 0;}
