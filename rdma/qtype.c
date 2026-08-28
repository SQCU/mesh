#include <infiniband/verbs.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static struct ibv_context *c; static struct ibv_pd *pd; static struct ibv_cq *cq;
static void tdn(void){ if(cq)ibv_destroy_cq(cq); if(pd)ibv_dealloc_pd(pd); if(c)ibv_close_device(c); }
int main(int argc,char**argv){
  const char*dev=argc>1?argv[1]:NULL; atexit(tdn);
  struct ibv_device **dl=ibv_get_device_list(NULL); struct ibv_device*d=NULL;
  for(int i=0;dl&&dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){d=dl[i];break;}
  if(!d){puts("no device");return 1;}
  c=ibv_open_device(d); struct ibv_port_attr pa; ibv_query_port(c,1,&pa);
  if(pa.state!=IBV_PORT_ACTIVE){puts("port not active");return 2;}
  pd=ibv_alloc_pd(c); cq=ibv_create_cq(c,64,NULL,NULL,0);
  struct { const char*n; enum ibv_qp_type t; } types[] = {
    {"RC (reliable connection, hw retransmit)", IBV_QPT_RC},
    {"UC (unreliable connection)",              IBV_QPT_UC},
    {"UD (unreliable datagram)",                IBV_QPT_UD},
  };
  for(unsigned i=0;i<sizeof types/sizeof*types;i++){
    struct ibv_qp_init_attr ia={.send_cq=cq,.recv_cq=cq,.qp_type=types[i].t,
      .cap={.max_send_wr=64,.max_recv_wr=64,.max_send_sge=1,.max_recv_sge=1}};
    struct ibv_qp *q=ibv_create_qp(pd,&ia);
    if(q){ printf("  %-42s SUPPORTED\n", types[i].n); ibv_destroy_qp(q); }
    else   printf("  %-42s rejected (errno %d)\n", types[i].n, errno);
  }
  struct ibv_device_attr da; ibv_query_device(c,&da);
  printf("  device max_qp_rd_atom=%d (0 means no reliable ops)\n", da.max_qp_rd_atom);
  return 0;
}
