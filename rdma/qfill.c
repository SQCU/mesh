#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static struct ibv_context *c; static struct ibv_pd *pd; static struct ibv_cq *cq;
static struct ibv_qp *qp; static struct ibv_mr *mr; static void *mem;
static void tdn(void){ if(qp)ibv_destroy_qp(qp); if(cq)ibv_destroy_cq(cq); if(mr)ibv_dereg_mr(mr); if(pd)ibv_dealloc_pd(pd); if(c)ibv_close_device(c); }
int main(int argc,char**argv){
  const char *dev=argc>1?argv[1]:NULL; atexit(tdn);
  struct ibv_device **dl=ibv_get_device_list(NULL); struct ibv_device *d=NULL;
  for(int i=0;dl&&dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){d=dl[i];break;}
  if(!d){puts("no device");return 1;}
  c=ibv_open_device(d); struct ibv_port_attr pa; ibv_query_port(c,1,&pa);
  if(pa.state!=IBV_PORT_ACTIVE){puts("port not active");return 2;}
  pd=ibv_alloc_pd(c);
  size_t span=15u<<20; posix_memalign(&mem,4096,span); memset(mem,0,span);
  mr=ibv_reg_mr(pd,mem,span,IBV_ACCESS_LOCAL_WRITE);
  printf("  %-8s %-10s %-12s %-12s %s\n","pagesz","frames","recv posted","send posted","frames used");
  int sizes[]={4096,8192,16384,65536,0};
  for(int s=0;sizes[s];s++){
    int pg=sizes[s], fr=pg/4096;
    cq=ibv_create_cq(c,4096,NULL,NULL,0);
    struct ibv_qp_init_attr ia={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=4095,.max_recv_wr=4095,.max_send_sge=1,.max_recv_sge=1}};
    qp=ibv_create_qp(pd,&ia);
    struct ibv_qp_attr at={.qp_state=IBV_QPS_INIT,.pkey_index=0,.port_num=1,.qp_access_flags=0};
    ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS);
    int nr=0; struct ibv_recv_wr *bad;
    for(;;){
      int slot=nr % (int)(span/pg);
      struct ibv_sge g={(uintptr_t)((char*)mem+(size_t)slot*pg),(uint32_t)pg,mr->lkey};
      struct ibv_recv_wr w={.wr_id=(uint64_t)nr,.sg_list=&g,.num_sge=1};
      if(ibv_post_recv(qp,&w,&bad)) break;
      if(++nr > 8192) break;
    }
    printf("  %-8d %-10d %-12d %-12s %d\n", pg, fr, nr, "-", nr*fr);
    ibv_destroy_qp(qp); qp=NULL; ibv_destroy_cq(cq); cq=NULL;
  }
  return 0;
}
