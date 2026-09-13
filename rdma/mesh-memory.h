#ifndef MESH_MEMORY_H
#define MESH_MEMORY_H
#include <libproc.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

struct mesh_process_memory { pid_t pid; uint64_t bytes; };

/* design/algorithm-sources.md#memory-warning */
static int mesh_memory_order(const void *a,const void *b){
  uint64_t x=((const struct mesh_process_memory *)a)->bytes,y=((const struct mesh_process_memory *)b)->bytes;
  return (x<y)-(x>y);
}

/* design/algorithm-sources.md#memory-warning */
static void mesh_memory_warning(uint64_t bytes,int creating){
  uint64_t ram=0; size_t size=sizeof ram;
  vm_statistics64_data_t vm={0}; mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
  mach_port_t host=mach_host_self(); vm_size_t page=0;
  int known=!sysctlbyname("hw.memsize",&ram,&size,NULL,0) && ram;
  int available=host_page_size(host,&page)==KERN_SUCCESS && host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS;
  mach_port_deallocate(mach_task_self(),host);
  uint64_t headroom=((uint64_t)vm.free_count+vm.inactive_count)*page;
  if(known && bytes<=ram/5*4 && (!creating || (available && headroom>=ram/5 && bytes<=headroom-ram/5))) return;
  printf("WARNING: mesh %s %.3f GiB of shared memory; RAM %.3f GiB; 80%% threshold %.3f GiB; free + inactive estimate %.3f GiB.\n",
    creating?"is about to allocate":"is attaching to",bytes/1073741824.,ram/1073741824.,ram*.8/1073741824.,headroom/1073741824.);
  printf("WARNING: %s %s Available memory is an estimate, not a reservation.\n",
    known?"Arena or projected usage exceeds 80% of RAM.":"RAM accounting unavailable.",available?"":"Available-memory accounting unavailable.");
  fflush(stdout);
  int capacity=proc_listpids(PROC_ALL_PIDS,0,NULL,0);
  pid_t *pids=capacity>0?calloc(1,(size_t)capacity):NULL;
  struct mesh_process_memory *processes=capacity>0?calloc((size_t)capacity/sizeof(pid_t),sizeof *processes):NULL;
  if(!pids || !processes){ printf("WARNING: process memory list unavailable.\n"); free(pids); free(processes); fflush(stdout); return; }
  int length=proc_listpids(PROC_ALL_PIDS,0,pids,capacity),n=0,missing=0;
  uint64_t total=0,shown=0;
  for(int i=0;i<(length<capacity?length:capacity)/(int)sizeof(pid_t);i++){
    if(pids[i]<=0 || pids[i]==getpid()) continue;
    struct rusage_info_v2 usage={0};
    if(proc_pid_rusage(pids[i],RUSAGE_INFO_V2,(rusage_info_t *)&usage)){ missing++; continue; }
    processes[n++]=(struct mesh_process_memory){pids[i],usage.ri_phys_footprint}; total+=usage.ri_phys_footprint;
  }
  qsort(processes,(size_t)n,sizeof *processes,mesh_memory_order);
  printf("Other processes covering 80%% of measured physical footprint: PID  GiB  executable\n");
  for(int i=0;i<n && (double)shown<(double)total*.8;i++){
    char path[PROC_PIDPATHINFO_MAXSIZE]="<unavailable>";
    proc_pidpath(processes[i].pid,path,sizeof path);
    printf("%d  %.3f  %s\n",processes[i].pid,processes[i].bytes/1073741824.,path);
    shown+=processes[i].bytes;
  }
  printf("Measured other-process footprint %.3f GiB; %d unreadable processes. Shared accounting is not additive headroom.\n",total/1073741824.,missing);
  if(length<=0) printf("WARNING: process enumeration unavailable.\n");
  free(processes); free(pids); fflush(stdout);
}
#endif
