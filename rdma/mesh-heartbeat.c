/* design/transport.md section 2, "The heartbeat": the node's bridge supervised by one small process, the job launchd
   keeps alive (bin/mesh-user.sh).  It has no device, no region and no socket of the mesh: it opens the bridge (spawned
   in a process group of its own, so a heartbeat that ends leaves it serving), closes it when asked (a signal, its exit
   then heard as a kernel event, never inferred from time passing), and is the one process that unlinks an expired
   region, once none of the region's processes lives (the bridge, its children and its clients, each recorded in the
   region by its pid and start: mesh_alive), before it opens a bridge again: a bridge is never opened while another is
   ending, and no second region is ever made beside a live process of the first (a process in its exit is none:
   mesh_started).

   Its operator's signals (`launchctl kill SIG user/$(id -u)/io.mesh.bridge`):
   - SIGHUP: open, or restart: the bridge closed by SIGHUP (its region left for the next to adopt), this heartbeat's own
     image run again (a deploy's new binaries both), and the bridge opened again (once an expired region is unlinked);
   - SIGUSR1: release: the bridge closed by SIGTERM (its children end, so the kernel frees their devices' objects; its
     region expired, so every client's next attach or communicator returns the expiry), the region unlinked once no
     process of it lives (its memory then the system's), and no bridge opened until SIGHUP;
   - SIGTERM, SIGINT (launchd's stop of the job, the system's shutdown): release, then this heartbeat ends (0, which
     launchd's KeepAlive {SuccessfulExit: false} does not restart).
   The bridge keeps its region at every stop of its own; a release is the heartbeat's: once the bridge it closed has
   ended, it expires the region (region_expire).  A bridge that ends unasked is opened again REOPEN_S later (a mesh is
   never restarted more than once a minute); one that ended finding its region expired (EX_TEMPFAIL, mesh-link.c
   region_open) is opened again once the heartbeat has unlinked that region.  All of this is one state machine on the
   heartbeat's main thread (enum heartbeat_state, heartbeat_step), moved by these signals, the bridge's exit, the exits
   of an expired region's processes and each state's deadline, every one of them read in its kqueue's one wait. */
#include "mesh.h"
#include "mesh-observe.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
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
enum { TAG_BRIDGE=1, TAG_PROCESS=2 };

/* a line of the log: its time and the message, one write (stderr is line-buffered from main) */
static void say(const char *format,...){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  fprintf(stderr,"%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ %.3f heartbeat: ",utc.tm_year+1900,utc.tm_mon+1,utc.tm_mday,utc.tm_hour,
          utc.tm_min,utc.tm_sec,t.tv_nsec/1000000,(double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW)/1e9);
  va_list arguments;va_start(arguments,format);vfprintf(stderr,format,arguments);va_end(arguments);
}
static uint64_t wall_us(void){return clock_gettime_nsec_np(CLOCK_REALTIME)/1000u;}
static uint64_t uptime_ns(void){return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);}

static int events;
static char self[PATH_MAX];
static char *const *arguments;
static const char *image,*configuration;
static const int heard[]={SIGHUP,SIGUSR1,SIGTERM,SIGINT};
/* design/transport.md section 2: the heartbeat's state, one machine, its main thread's alone (heartbeat_step), with
   the bridge of the states that have one (`bridge`: the one this heartbeat opened or found serving) and the time it
   served from (`opened_at`, wall microseconds), and the state's deadline (`due`, uptime nanoseconds; 0 none):
   - SERVING: a bridge serves, nothing asked of it;
   - RESTARTING: the bridge asked to close by SIGHUP (its region left for the next to adopt); at its exit this
     heartbeat's image runs again, and a bridge is opened;
   - RELEASING: the bridge asked to close by SIGTERM, a release; at its exit the region is expired and replaced, then a
     bridge opened where `reopen` (a SIGHUP since the release was asked), else none;
   - ENDING: the bridge asked to close by SIGTERM, this heartbeat's end; at its exit, or at `due` (3 MESH_PLATFORM_NS
     after it was asked), the region is expired and the heartbeat ends;
   - REOPENING: no bridge; one opened at `due`, REOPEN_S after an unasked end or a spawn that failed (a mesh is never
     restarted more than once a minute);
   - REPLACING: no bridge; an expired region waits for its processes (region_replace), read again at each one's exit
     and at `due` (REPLACE_TICK_MS on, for a client that lets go and lives on; REOPEN_S on after an unlink that failed),
     then a bridge opened where `reopen` (a restart, an unasked end, a SIGHUP), else RELEASED;
   - RELEASED: no bridge, and none opened until SIGHUP */
enum heartbeat_state { HEARTBEAT_SERVING, HEARTBEAT_RESTARTING, HEARTBEAT_RELEASING, HEARTBEAT_ENDING, HEARTBEAT_REOPENING,
                       HEARTBEAT_REPLACING, HEARTBEAT_RELEASED };
/* what moves it: the operator's signals (SIGHUP, SIGUSR1, SIGTERM or SIGINT), the bridge's exit, the state's deadline,
   and the exit of an expired region's process */
enum heartbeat_event { HEARTBEAT_RESTART, HEARTBEAT_RELEASE, HEARTBEAT_END, HEARTBEAT_EXIT, HEARTBEAT_DUE, HEARTBEAT_PROCESS };
static struct { enum heartbeat_state state; pid_t bridge; int reopen; uint64_t opened_at,due; } heartbeat;
/* the processes REPLACING last named */
static pid_t told[1+2*MESH_PORTS+MESH_CLIENTS];
static uint32_t told_count;
static void enter(enum heartbeat_state state,uint64_t due){heartbeat.state=state;heartbeat.due=due;}

/* the node's region by name, mapped (read-only, or read-write where `writable`) and its descriptor closed at once:
   NULL where none is named */
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
/* region `m`'s bridge area, where it lies within its `size` bytes, else NULL */
static struct mesh_bridge_pages *region_own(struct hdr *m,size_t size){
  const uint64_t at=m->bridge_off+MESH_PORTS*sizeof(struct mesh_link_info);
  return m->magic==MESH_MAGIC && at+sizeof(struct mesh_bridge_pages)<=size?(struct mesh_bridge_pages *)((char *)m+at):NULL;
}
/* the live bridge serving region `m` (its `bridge_pid` of the start it recorded), or 0 */
static pid_t bridge_of(struct hdr *m,size_t size,uint64_t *start){
  const struct mesh_bridge_pages *own=region_own(m,size);
  const pid_t was=own?(pid_t)atomic_load(&m->bridge_pid):0;
  *start=own?own->bridge_start:0;
  return was && mesh_alive(was,*start)?was:0;
}
static pid_t region_bridge(uint64_t *start){
  size_t size=0;
  struct hdr *m=region_map(&size,0);
  const pid_t was=m?bridge_of(m,size,start):0;
  if(m)munmap(m,size);
  return was;
}
/* the end of a release: the named region, where no bridge serves it, made expired (layout version 0), as its bridge's
   release stop makes it (mesh-link.c), for a bridge that ended keeping it (a restart's close the release overtook, a
   bridge launchd ran itself, or an end before its stop) */
static void region_expire(void){
  size_t size=0;
  struct hdr *m=region_map(&size,1);
  uint64_t start;
  if(m && m->magic==MESH_MAGIC && !bridge_of(m,size,&start))__atomic_store_n(&m->version,0,__ATOMIC_SEQ_CST);
  if(m)munmap(m,size);
}
static void reopen_later(void){
  say("a bridge opened in %u s\n",REOPEN_S);
  enter(HEARTBEAT_REOPENING,uptime_ns()+UINT64_C(1000000000)*REOPEN_S);
}

/* design/transport.md section 2: an expired region (magic, layout version 0) unlinked once none of its processes
   lives: its bridge (`bridge_pid`, `bridge_start`), each child its bridges recorded, and each client slot's process
   (its owner's pid, `start`), each identified by its pid and start (mesh_alive, so a reused pid is none of them).
   Until then their exits are watched on this heartbeat's queue (mesh-observe.h mesh_exits_watch, each exit a
   HEARTBEAT_PROCESS event), and they are named as they change.  1 where no expired region is left (none is named, the
   region is live, or it is now unlinked), 0 where one of its processes lives, -1 where its unlink failed */
static int region_replace(void){
  size_t size=0;
  struct hdr *m=region_map(&size,0);
  struct mesh_bridge_pages *own=m?region_own(m,size):NULL;
  if(!own || __atomic_load_n(&m->version,__ATOMIC_ACQUIRE)!=0){
    if(m)munmap(m,size);
    told_count=0;
    return 1;
  }
  struct mesh_process processes[1+2*MESH_PORTS+MESH_CLIENTS];
  uint32_t count=0;
  processes[count++]=(struct mesh_process){.pid=(uint32_t)atomic_load(&m->bridge_pid),.start=own->bridge_start};
  for(uint32_t a=0;a<2*MESH_PORTS;a++)processes[count++]=own->child[a];
  for(uint32_t c=0;c<m->clients && c<MESH_CLIENTS && m->client_off+(uint64_t)(c+1)*sizeof(struct mesh_client)<=size;c++){
    const struct mesh_client *slot=(const struct mesh_client *)((const char *)m+m->client_off)+c;
    processes[count++]=(struct mesh_process){.pid=(uint32_t)atomic_load(&slot->owner),.start=slot->start};
  }
  munmap(m,size);
  count=mesh_exits_watch(events,processes,count,TAG_PROCESS);
  if(!count){
    told_count=0;
    if(!shm_unlink(MESH_NAME) || errno==ENOENT){
      say("region %s unlinked: none of its processes lives, and its memory is the system's\n",MESH_NAME);
      return 1;
    }
    say("region %s not unlinked: %s; tried again in %u s\n",MESH_NAME,strerror(errno),REOPEN_S);
    return -1;
  }
  int same=count==told_count;
  for(uint32_t k=0;k<count && same;k++)same=(pid_t)processes[k].pid==told[k];
  if(!same){
    say("region %s is expired; it is unlinked once each of its %u processes ends or lets it go:",MESH_NAME,count);
    for(uint32_t k=0;k<count;k++){fprintf(stderr," %u",processes[k].pid);told[k]=(pid_t)processes[k].pid;}
    fputc('\n',stderr);
    told_count=count;
  }
  return 0;
}

/* the bridge's exit watched, its status told with it where it is this heartbeat's child (`mine`, NOTE_EXITSTATUS): 0,
   or 1 where it has already exited (ESRCH).  Any other error ends this heartbeat with EX_OSERR, since no event would
   tell that exit: launchd opens the heartbeat again (its KeepAlive), and its start finds the bridge serving */
static int bridge_watch(pid_t pid,int mine){
  struct kevent64_s watch;
  EV_SET64(&watch,(uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT|(mine?NOTE_EXITSTATUS:0),0,TAG_BRIDGE,0,0);
  if(!kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))return 0;
  if(errno==ESRCH)return 1;
  say("bridge %d's exit not watched (%s): the heartbeat ends, and launchd opens it again\n",(int)pid,strerror(errno));
  exit(EX_OSERR);
}

/* this heartbeat's image run again (a deploy's new one), its operator's signals left pending across the exec (blocked:
   the new image reads them, main) */
static void run_again(void){
  sigset_t blocked;sigemptyset(&blocked);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)sigaddset(&blocked,heard[i]);
  sigprocmask(SIG_BLOCK,&blocked,NULL);
  /* a signal the kqueue heard since its last read raised again once the kqueue is drained (the kqueue records a
     blocked signal's delivery too), now blocked, so it is pending across the exec */
  struct kevent64_s left[8];
  const struct timespec none={0,0};
  sigset_t again;sigemptyset(&again);
  for(int n;(n=kevent64(events,NULL,0,left,8,0,&none))>0;)
    for(int i=0;i<n;i++)if(left[i].filter==EVFILT_SIGNAL)sigaddset(&again,(int)left[i].ident);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)if(sigismember(&again,heard[i]))kill(getpid(),heard[i]);
  say("the heartbeat runs its image again (%s)\n",self);
  execv(self,arguments);
  say("the heartbeat's image %s not run again: %s\n",self,strerror(errno));
  sigprocmask(SIG_UNBLOCK,&blocked,NULL);
}

static void bridge_exited(const struct kevent64_s *e);
/* a bridge opened (SERVING): spawned in a process group of its own, its signals' dispositions the defaults and none
   blocked, its input /dev/null, its log this heartbeat's; a spawn that fails is tried again REOPEN_S later.  Every
   earlier bridge of this heartbeat's that has ended is reaped first: kqueue tells a process's exit as it begins
   (NOTE_EXIT), so the one bridge_exited found not yet reapable is reaped here */
static void bridge_open(void){
  while(waitpid(-1,NULL,WNOHANG)>0){}
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
  heartbeat.bridge=pid;heartbeat.opened_at=wall_us();enter(HEARTBEAT_SERVING,0);
  say("bridge %d opened (%s %s)\n",(int)pid,image,configuration);
  if(bridge_watch(pid,1))bridge_exited(NULL);
}

/* the expired region replaced (region_replace), then a bridge opened where `reopen`, else none (RELEASED); while it
   is not, REPLACING */
static void replace(int reopen){
  const int left=region_replace();
  heartbeat.reopen=reopen;
  if(left<=0)enter(HEARTBEAT_REPLACING,uptime_ns()+(left?UINT64_C(1000000000)*REOPEN_S:UINT64_C(1000000)*REPLACE_TICK_MS));
  else if(reopen)bridge_open();
  else enter(HEARTBEAT_RELEASED,0);
}

/* design/transport.md section 2: the bridge's end, as asked or not.  Its status is known where its process is reaped
   (waitpid, which never waits here) or, for this heartbeat's child not yet reapable, from its exit's event
   (NOTE_EXITSTATUS); a bridge found serving, not this heartbeat's child (ECHILD), tells none */
static void bridge_exited(const struct kevent64_s *e){
  const pid_t was=heartbeat.bridge;
  int status=0,known=waitpid(was,&status,WNOHANG)==was;
  if(!known && e && (e->fflags&NOTE_EXITSTATUS)){status=(int)e->data;known=1;}
  heartbeat.bridge=0;
  if(!known)say("bridge %d ended\n",(int)was);
  else if(WIFEXITED(status))say("bridge %d exited %d\n",(int)was,WEXITSTATUS(status));
  else say("bridge %d ended by signal %d\n",(int)was,WIFSIGNALED(status)?WTERMSIG(status):0);
  if(heartbeat.state==HEARTBEAT_ENDING){
    region_expire();
    region_replace();
    say("the heartbeat ends\n");
    exit(0);
  }
  if(heartbeat.state==HEARTBEAT_RESTARTING){run_again();replace(1);}
  else if(heartbeat.state==HEARTBEAT_RELEASING){region_expire();replace(heartbeat.reopen);}
  else if(known && WIFEXITED(status) && WEXITSTATUS(status)==EX_TEMPFAIL){
    say("its region is expired: a bridge is opened once the region is unlinked\n");
    replace(1);
  }
  else {
    say("it ended unasked after %.0f s\n",(double)(wall_us()-heartbeat.opened_at)/1e6);
    reopen_later();
  }
}

/* the bridge asked to close by `signal_number` (SIGHUP: a restart, its region left for the next bridge; SIGTERM: a
   release or this heartbeat's end) */
static void bridge_ask(int signal_number){
  say("bridge %d asked to close (%s)\n",(int)heartbeat.bridge,signal_number==SIGHUP?"SIGHUP: restarted, its region left for the next bridge":"SIGTERM: released");
  kill(heartbeat.bridge,signal_number);
}

/* design/transport.md section 2: the heartbeat's one transition function (enum heartbeat_state), event `event` (its
   kernel event `e`, where it has one) taken in the current state */
static void heartbeat_step(enum heartbeat_event event,const struct kevent64_s *e){
  const enum heartbeat_state state=heartbeat.state;
  switch(state){
  case HEARTBEAT_SERVING:
  case HEARTBEAT_RESTARTING:
  case HEARTBEAT_RELEASING:
    if(event==HEARTBEAT_EXIT)bridge_exited(e);
    else if(event==HEARTBEAT_END){bridge_ask(SIGTERM);enter(HEARTBEAT_ENDING,uptime_ns()+3*MESH_PLATFORM_NS);}
    else if(event==HEARTBEAT_RELEASE && state!=HEARTBEAT_RELEASING){bridge_ask(SIGTERM);heartbeat.reopen=0;enter(HEARTBEAT_RELEASING,0);}
    else if(event==HEARTBEAT_RELEASE)heartbeat.reopen=0;
    else if(event==HEARTBEAT_RESTART && state==HEARTBEAT_SERVING){bridge_ask(SIGHUP);enter(HEARTBEAT_RESTARTING,0);}
    else if(event==HEARTBEAT_RESTART && state==HEARTBEAT_RELEASING)heartbeat.reopen=1;
    return;
  case HEARTBEAT_ENDING:
    if(event==HEARTBEAT_EXIT)bridge_exited(e);
    else if(event==HEARTBEAT_DUE){
      say("bridge %d has not ended %.0f s after it was asked; the heartbeat ends without it\n",(int)heartbeat.bridge,
          (double)(3*MESH_PLATFORM_NS)/1e9);
      region_expire();
      exit(0);
    }
    return;
  case HEARTBEAT_REOPENING:
  case HEARTBEAT_REPLACING:
  case HEARTBEAT_RELEASED:
    if(event==HEARTBEAT_RESTART){run_again();replace(1);}
    else if(event==HEARTBEAT_RELEASE){region_expire();replace(0);}
    else if(event==HEARTBEAT_END){
      region_expire();
      region_replace();
      say("signal %d: the heartbeat ends\n",e?(int)e->ident:0);
      exit(0);
    }
    else if(event==HEARTBEAT_DUE && state==HEARTBEAT_REOPENING)replace(1);
    else if((event==HEARTBEAT_DUE || event==HEARTBEAT_PROCESS) && state==HEARTBEAT_REPLACING)replace(heartbeat.reopen);
    return;
  }
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
  /* the operator's signals: blocked while their filters are added (a run_again leaves them blocked and pending across
     the exec), each then given a handler that does nothing (an ignored signal is discarded, a pending one too), and
     each one pending since before its filter raised again once unblocked, so its filter hears it */
  sigset_t blocked,pending;sigemptyset(&blocked);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)sigaddset(&blocked,heard[i]);
  sigprocmask(SIG_BLOCK,&blocked,NULL);
  if((events=kqueue())<0 || fcntl(events,F_SETFD,FD_CLOEXEC)){say("events: %s\n",strerror(errno));return 1;}
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++){
    struct kevent64_s e;
    EV_SET64(&e,(uint64_t)heard[i],EVFILT_SIGNAL,EV_ADD,0,0,0,0,0);
    if(kevent64(events,&e,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){say("events: %s\n",strerror(errno));return 1;}
    struct sigaction action={.sa_handler=noted,.sa_flags=SA_RESTART};
    sigaction(heard[i],&action,NULL);
  }
  signal(SIGPIPE,SIG_IGN);
  sigpending(&pending);
  sigprocmask(SIG_UNBLOCK,&blocked,NULL);
  for(size_t i=0;i<sizeof heard/sizeof *heard;i++)if(sigismember(&pending,heard[i]))kill(getpid(),heard[i]);
  uint64_t start=0;
  const pid_t serving=region_bridge(&start);
  if(serving && !bridge_watch(serving,0)){
    heartbeat.bridge=serving;heartbeat.opened_at=start?start:wall_us();enter(HEARTBEAT_SERVING,0);
    say("bridge %d found serving region %s; watched\n",(int)serving,MESH_NAME);
  }
  else replace(1);
  /* the one wait: the queue's next events, or the state's deadline */
  for(;;){
    struct kevent64_s ready[8];
    const uint64_t at=uptime_ns(),rest=heartbeat.due>at?heartbeat.due-at:0;
    const struct timespec wait={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    const int count=kevent64(events,NULL,0,ready,8,0,heartbeat.due?&wait:NULL);
    if(count<0 && errno!=EINTR){say("events: %s\n",strerror(errno));return 1;}
    for(int i=0;i<count;i++){
      const struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL)
        heartbeat_step(e->ident==SIGHUP?HEARTBEAT_RESTART:e->ident==SIGUSR1?HEARTBEAT_RELEASE:HEARTBEAT_END,e);
      else if(e->filter==EVFILT_PROC && e->udata==TAG_BRIDGE && heartbeat.bridge && (pid_t)e->ident==heartbeat.bridge)
        heartbeat_step(HEARTBEAT_EXIT,e);
      else if(e->filter==EVFILT_PROC && e->udata==TAG_PROCESS)heartbeat_step(HEARTBEAT_PROCESS,e);
    }
    if(heartbeat.due && uptime_ns()>=heartbeat.due)heartbeat_step(HEARTBEAT_DUE,NULL);
  }
}
