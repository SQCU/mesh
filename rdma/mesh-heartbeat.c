/* design/transport.md section 2, "The heartbeat" */
#include "mesh-call.h"
#include "mesh-observe.h"
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sysexits.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define REOPEN_S 60u
#define REPLACE_TICK_MS 1000
enum { TIMER_OPEN=1, TIMER_END=2 };
enum { TAG_BRIDGE=1, TAG_HOLDER=2 };

static void say(const char *format,...){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  fprintf(stderr,"%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ %.3f heartbeat: ",utc.tm_year+1900,utc.tm_mon+1,utc.tm_mday,utc.tm_hour,
          utc.tm_min,utc.tm_sec,t.tv_nsec/1000000,(double)clock_gettime_nsec_np(CLOCK_MONOTONIC)/1e9);
  va_list arguments;va_start(arguments,format);vfprintf(stderr,format,arguments);va_end(arguments);
}
static uint64_t wall_us(void){return clock_gettime_nsec_np(CLOCK_REALTIME)/1000u;}

static int events;
static char self[PATH_MAX];
static char *const *arguments;
static const char *image,*configuration;
static const int heard[]={SIGHUP,SIGUSR1,SIGTERM,SIGINT};
static pid_t bridge;
static int child;
static enum { ASKED_NONE, ASKED_RESTART, ASKED_RELEASE, ASKED_END } asked;
static uint64_t opened_at;
static int wanted=1,replacing,timer_open;
static pid_t told[3*MESH_PORTS+MESH_CLIENTS];
static uint32_t told_count;

static int watch(uint64_t ident,int16_t filter,uint16_t flags,uint32_t fflags,int64_t data,uint64_t udata){
  struct kevent64_s event;EV_SET64(&event,ident,filter,flags,fflags,data,udata,0,0);
  return kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
}
static struct hdr *region_map(size_t *size,int writable){
  const int fd=shm_open(MESH_NAME,writable?O_RDWR:O_RDONLY,0);
  if(fd<0)return NULL;
  struct stat shape;
  struct hdr *m=MAP_FAILED;
  if(!fstat(fd,&shape) && shape.st_size>=(off_t)sizeof(struct hdr))
    m=mmap(NULL,*size=(size_t)shape.st_size,PROT_READ|(writable?PROT_WRITE:0),MAP_SHARED,fd,0);
  close(fd);
  return m==MAP_FAILED?NULL:m;
}
static struct mesh_bridge_pages *region_own(struct hdr *m,size_t size){
  const uint64_t at=m->bridge_off+MESH_PORTS*sizeof(struct mesh_link_info);
  return m->magic==MESH_MAGIC && at+sizeof(struct mesh_bridge_pages)<=size?(struct mesh_bridge_pages *)((char *)m+at):NULL;
}
static pid_t bridge_of(struct hdr *m,size_t size,uint64_t *start){
  const struct mesh_bridge_pages *own=region_own(m,size);
  const pid_t was=own?(pid_t)atomic_load(&m->bridge_pid):0;
  *start=own?own->bridge_start:0;
  return was && mesh_holds(was,*start)?was:0;
}
static void region_expire(void){
  size_t size=0;
  struct hdr *m=region_map(&size,1);
  uint64_t start;
  if(m && m->magic==MESH_MAGIC && !bridge_of(m,size,&start))__atomic_store_n(&m->version,0,__ATOMIC_SEQ_CST);
  if(m)munmap(m,size);
}
static void reopen_later(void){
  say("a bridge opened in %u s\n",REOPEN_S);
  timer_open=1;
  watch(TIMER_OPEN,EVFILT_TIMER,EV_ADD|EV_ONESHOT,0,(int64_t)(1000ull*REOPEN_S),0);
}
static void replace_turn(void){
  size_t size=0;
  struct hdr *m=region_map(&size,0);
  struct mesh_bridge_pages *own=m?region_own(m,size):NULL;
  if(!own || __atomic_load_n(&m->version,__ATOMIC_ACQUIRE)!=0){
    if(m)munmap(m,size);
    replacing=0;told_count=0;
    return;
  }
  pid_t holders[3*MESH_PORTS+MESH_CLIENTS];
  uint32_t count=0;
  uint64_t start;
  const pid_t serving=bridge_of(m,size,&start);
  if(serving)holders[count++]=serving;
  for(uint32_t a=0;a<2*MESH_PORTS;a++)
    if(own->child[a].pid && mesh_holds((pid_t)own->child[a].pid,own->child[a].start))holders[count++]=(pid_t)own->child[a].pid;
  for(uint32_t c=0;c<m->clients && c<MESH_CLIENTS && m->client_off+(uint64_t)(c+1)*sizeof(struct mesh_client)<=size;c++){
    const struct mesh_client *slot=(const struct mesh_client *)((const char *)m+m->client_off)+c;
    const pid_t pid=(pid_t)(uint32_t)atomic_load(&slot->owner);
    if(pid && mesh_holds(pid,slot->start))holders[count++]=pid;
  }
  munmap(m,size);
  if(!count){
    replacing=0;told_count=0;
    if(!shm_unlink(MESH_NAME) || errno==ENOENT)say("region %s unlinked: none of its holders lives, and its memory is the system's\n",MESH_NAME);
    else {
      say("region %s not unlinked: %s\n",MESH_NAME,strerror(errno));
      if(!timer_open)reopen_later();
    }
    return;
  }
  replacing=1;
  int same=count==told_count;
  for(uint32_t k=0;k<count && same;k++)same=holders[k]==told[k];
  if(!same){
    say("region %s is expired; it is unlinked, and a bridge opened, once each of its %u holders ends or lets it go:",MESH_NAME,count);
    for(uint32_t k=0;k<count;k++)fprintf(stderr," %d",(int)holders[k]);
    fputc('\n',stderr);
    memcpy(told,holders,count*sizeof *holders);told_count=count;
  }
  for(uint32_t k=0;k<count;k++)watch((uint64_t)holders[k],EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_HOLDER);
}
static void run_again(void){
  sigset_t held;sigemptyset(&held);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)sigaddset(&held,heard[i]);
  sigprocmask(SIG_BLOCK,&held,NULL);
  struct kevent64_s left[8];
  const struct timespec none={0,0};
  sigset_t again;sigemptyset(&again);
  for(int n;(n=kevent64(events,NULL,0,left,8,0,&none))>0;)
    for(int i=0;i<n;i++)if(left[i].filter==EVFILT_SIGNAL)sigaddset(&again,(int)left[i].ident);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)if(sigismember(&again,heard[i]))kill(getpid(),heard[i]);
  say("the heartbeat runs its image again (%s)\n",self);
  execv(self,arguments);
  say("the heartbeat's image %s not run again: %s\n",self,strerror(errno));
  sigprocmask(SIG_UNBLOCK,&held,NULL);
}
static void bridge_ended(void);
static void open_bridge(void){
  if(!wanted || bridge || timer_open)return;
  replace_turn();
  if(replacing || timer_open)return;
  posix_spawnattr_t attributes;
  posix_spawn_file_actions_t actions;
  sigset_t none,defaults;
  sigemptyset(&none);sigemptyset(&defaults);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)sigaddset(&defaults,heard[i]);
  sigaddset(&defaults,SIGPIPE);
  posix_spawnattr_init(&attributes);
  posix_spawnattr_setflags(&attributes,POSIX_SPAWN_SETPGROUP|POSIX_SPAWN_SETSIGDEF|POSIX_SPAWN_SETSIGMASK);
  posix_spawnattr_setpgroup(&attributes,0);
  posix_spawnattr_setsigdefault(&attributes,&defaults);
  posix_spawnattr_setsigmask(&attributes,&none);
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
  char *const argv[]={(char *)image,(char *)configuration,NULL};
  pid_t pid=0;
  const int failed=posix_spawn(&pid,image,&actions,&attributes,argv,environ);
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attributes);
  if(failed){
    say("bridge %s not opened: %s\n",image,strerror(failed));
    reopen_later();
    return;
  }
  bridge=pid;child=1;asked=ASKED_NONE;opened_at=wall_us();
  say("bridge %d opened (%s %s)\n",(int)pid,image,configuration);
  if(watch((uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT|NOTE_EXITSTATUS,0,TAG_BRIDGE))bridge_ended();
}
static void turn(void){
  if(wanted && !bridge && !timer_open)open_bridge();
  else if(!bridge)replace_turn();
}
static void bridge_ended(void){
  int status=0;
  const pid_t was=bridge;
  const int reaped=child && waitpid(was,&status,0)==was;
  bridge=0;child=0;
  const int expired=reaped && WIFEXITED(status) && WEXITSTATUS(status)==EX_TEMPFAIL;
  if(!reaped)say("bridge %d ended\n",(int)was);
  else if(WIFEXITED(status))say("bridge %d exited %d\n",(int)was,WEXITSTATUS(status));
  else say("bridge %d ended by signal %d\n",(int)was,WIFSIGNALED(status)?WTERMSIG(status):0);
  switch(asked){
  case ASKED_END:
    region_expire();
    replace_turn();
    say("the heartbeat ends\n");
    exit(0);
  case ASKED_RESTART:
    asked=ASKED_NONE;
    run_again();
    break;
  case ASKED_RELEASE:
    region_expire();
    break;
  case ASKED_NONE:
    if(expired)say("its region is expired: a bridge is opened once the region is unlinked\n");
    else {
      say("it ended unasked after %.0f s\n",(double)(wall_us()-opened_at)/1e6);
      reopen_later();
    }
    break;
  }
  asked=ASKED_NONE;
  turn();
}
static void close_bridge(int how,int signal_number){
  if(!bridge)return;
  if(asked==ASKED_END || (asked==ASKED_RELEASE && how!=ASKED_END))return;
  asked=how;
  say("bridge %d asked to close (%s)\n",(int)bridge,signal_number==SIGHUP?"SIGHUP: restarted, its region kept":"SIGTERM: released");
  kill(bridge,signal_number);
}
static void noted(int number){(void)number;}
int main(int argc,char **argv){
  setvbuf(stderr,NULL,_IOLBF,0);
  if(argc!=3){
    say("usage: mesh-heartbeat BRIDGE CONFIGURATION: the bridge's image (rdma/mesh-flow) and its configuration's property "
        "list (bin/mesh-user.sh writes both into the job's definition; etc/bridge/README.md)\n");
    return 1;
  }
  arguments=argv;image=argv[1];configuration=argv[2];
  uint32_t room=sizeof self;
  char resolved[PATH_MAX];
  if(_NSGetExecutablePath(self,&room) || !realpath(self,resolved)){say("the heartbeat's own image not read\n");return 1;}
  memcpy(self,resolved,sizeof self);
  sigset_t held,pending;sigemptyset(&held);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)sigaddset(&held,heard[i]);
  sigprocmask(SIG_BLOCK,&held,NULL);
  if((events=kqueue())<0 || fcntl(events,F_SETFD,FD_CLOEXEC)){say("events: %s\n",strerror(errno));return 1;}
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++){
    if(watch((uint64_t)heard[i],EVFILT_SIGNAL,EV_ADD,0,0,0)){say("events: %s\n",strerror(errno));return 1;}
    struct sigaction action={.sa_handler=noted,.sa_flags=SA_RESTART};
    sigaction(heard[i],&action,NULL);
  }
  signal(SIGPIPE,SIG_IGN);
  sigpending(&pending);
  sigprocmask(SIG_UNBLOCK,&held,NULL);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)if(sigismember(&pending,heard[i]))kill(getpid(),heard[i]);
  uint64_t start=0;
  size_t size=0;
  struct hdr *m=region_map(&size,0);
  const pid_t serving=m?bridge_of(m,size,&start):0;
  if(m)munmap(m,size);
  if(serving && !watch((uint64_t)serving,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_BRIDGE)){
    bridge=serving;child=0;asked=ASKED_NONE;opened_at=start?start:wall_us();
    say("bridge %d found serving region %s; watched\n",(int)serving,MESH_NAME);
  }
  else open_bridge();
  int ending=0;
  for(;;){
    struct kevent64_s ready[8];
    struct timespec tick={REPLACE_TICK_MS/1000,(long)(REPLACE_TICK_MS%1000)*1000000};
    const int count=kevent64(events,NULL,0,ready,8,0,replacing?&tick:NULL);
    if(count<0 && errno!=EINTR){say("events: %s\n",strerror(errno));return 1;}
    if(count==0 && replacing)turn();
    for(int i=0;i<count;i++){
      const struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL && (e->ident==SIGHUP || e->ident==SIGUSR1)){
        wanted=e->ident==SIGHUP;
        if(timer_open){timer_open=0;watch(TIMER_OPEN,EVFILT_TIMER,EV_DELETE,0,0,0);}
        if(bridge)close_bridge(wanted?ASKED_RESTART:ASKED_RELEASE,wanted?SIGHUP:SIGTERM);
        else if(wanted){run_again();turn();}
        else {region_expire();replace_turn();}
      }
      else if(e->filter==EVFILT_SIGNAL){
        wanted=0;
        if(!bridge){region_expire();replace_turn();say("signal %d: the heartbeat ends\n",(int)e->ident);return 0;}
        close_bridge(ASKED_END,SIGTERM);
        if(!ending){
          ending=1;
          watch(TIMER_END,EVFILT_TIMER,EV_ADD|EV_ONESHOT,0,(int64_t)(3*MESH_PLATFORM_NS/1000000),0);
        }
      }
      else if(e->filter==EVFILT_TIMER && e->ident==TIMER_OPEN){timer_open=0;turn();}
      else if(e->filter==EVFILT_TIMER && e->ident==TIMER_END){
        say("bridge %d has not ended %.0f s after it was asked; the heartbeat ends without it\n",(int)bridge,
            (double)(3*MESH_PLATFORM_NS)/1e9);
        region_expire();
        return 0;
      }
      else if(e->filter==EVFILT_PROC && e->udata==TAG_BRIDGE && (pid_t)e->ident==bridge)bridge_ended();
      else if(e->filter==EVFILT_PROC && e->udata==TAG_HOLDER)turn();
    }
  }
}
