/* Read-only install preflight, using the same holder identity as mesh-heartbeat.
   This is an observer, never a client and never an owner of region memory. */
#include "mesh.h"
#include "mesh-dataflow.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
/* Layout 153 and 154 differ in transport entries/pins, not holder identity. These
   exact header/client offsets were compared against the 153 committed header;
   a future ABI change must extend this explicit compatibility proof. */
_Static_assert(sizeof(struct hdr)==192 && offsetof(struct hdr,version)==4 &&
               offsetof(struct hdr,clients)==28 && offsetof(struct hdr,client_off)==80,"install observer header ABI");
_Static_assert(sizeof(struct mesh_client)==64 && offsetof(struct mesh_client,owner)==0 &&
               offsetof(struct mesh_client,start)==8,"install observer client ABI");
static int install_state_valid(const struct hdr *m,size_t bytes){
  return bytes>=sizeof *m && m->magic==MESH_MAGIC &&
    (m->version==0 || m->version==153 || m->version==154) && m->clients<=MESH_CLIENTS &&
    m->client_off>=sizeof *m && m->client_off%_Alignof(struct mesh_client)==0 && m->client_off<=bytes &&
    (uint64_t)m->clients*sizeof(struct mesh_client)<=bytes-m->client_off;
}
static uint32_t install_state_clients(const struct hdr *m,pid_t *out){
  uint32_t count=0;
  for(uint32_t c=0;c<m->clients;c++){
    const struct mesh_client *slot=(const struct mesh_client *)((const char *)m+m->client_off)+c;
    pid_t pid=(pid_t)(uint32_t)atomic_load_explicit(&slot->owner,memory_order_acquire);
    if(pid && mesh_holds(pid,slot->start))out[count++]=pid;
  }
  return count;
}
int main(void){
  int fd=shm_open(MESH_NAME,O_RDONLY,0);
  if(fd<0){if(errno!=ENOENT)return 1;puts("{\"present\":false,\"clients\":[]}");return 0;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){close(fd);return 2;}
  size_t bytes=(size_t)st.st_size;
  struct hdr *m=mmap(NULL,bytes,PROT_READ,MAP_SHARED,fd,0);close(fd);
  if(m==MAP_FAILED)return 3;
  if(!install_state_valid(m,bytes)){munmap(m,bytes);return 4;}
  pid_t clients[MESH_CLIENTS];uint32_t count=install_state_clients(m,clients);
  printf("{\"present\":true,\"clients\":[");
  for(uint32_t c=0;c<count;c++)printf("%s%d",c?",":"",(int)clients[c]);
  puts("]}");munmap(m,bytes);return 0;
}
