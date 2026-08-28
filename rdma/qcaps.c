#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
static struct ibv_context *c; static struct ibv_pd *pd; static struct ibv_cq *cq; static struct ibv_qp *qp;
static void tdn(void){ if(qp)ibv_destroy_qp(qp); if(cq)ibv_destroy_cq(cq); if(pd)ibv_dealloc_pd(pd); if(c)ibv_close_device(c); }
int main(int argc,char**argv){
  const char *dev = argc>1?argv[1]:NULL;
  atexit(tdn);
  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl){puts("no devices");return 1;}
  struct ibv_device *d=NULL;
  for(int i=0;dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){d=dl[i];break;}
  if(!d){puts("device not found");return 1;}
  c=ibv_open_device(d); if(!c){puts("open failed");return 1;}
  struct ibv_port_attr pa; ibv_query_port(c,1,&pa);
  printf("  device %s  port %s\n", ibv_get_device_name(d), ibv_port_state_str(pa.state));
  if(pa.state!=IBV_PORT_ACTIVE){ puts("  port not active; stopping"); return 2; }
  pd=ibv_alloc_pd(c); if(!pd){puts("alloc_pd failed");return 1;}
  cq=ibv_create_cq(c,4096,NULL,NULL,0); if(!cq){puts("create_cq failed");return 1;}
  int req[] = {4095, 2048, 1024, 256, 64, 0};
  for(int r=0; req[r]; r++){
    struct ibv_qp_init_attr ia={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=req[r],.max_recv_wr=req[r],.max_send_sge=1,.max_recv_sge=1}};
    struct ibv_qp *q=ibv_create_qp(pd,&ia);
    if(!q){ printf("  requested %5d -> create_qp FAILED\n", req[r]); continue; }
    struct ibv_qp_attr a; struct ibv_qp_init_attr ib;
    if(ibv_query_qp(q,&a,IBV_QP_CAP,&ib)){ printf("  requested %5d -> query_qp failed\n",req[r]); }
    else printf("  requested send/recv_wr=%-5d  GRANTED send=%-5u recv=%-5u sge=%u/%u\n",
        req[r], ib.cap.max_send_wr, ib.cap.max_recv_wr, ib.cap.max_send_sge, ib.cap.max_recv_sge);
    ibv_destroy_qp(q);
  }
  return 0;
}
