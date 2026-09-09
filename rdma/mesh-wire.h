#ifndef MESH_WIRE_H
#define MESH_WIRE_H
#include "mesh.h"
#include <string.h>

#define MESH_SCOPED UINT32_C(0x4d590000)
struct mesh_frame { struct shdr h; struct mesh_epoch epoch; uint16_t source, target; };
static inline int mesh_epoch_equal(struct mesh_epoch a,struct mesh_epoch b){ return a.high==b.high && a.low==b.low; }
static inline int mesh_epoch_set(struct mesh_epoch e){ return e.high || e.low; }
static inline size_t mesh_stream_header(const struct mstream *s){ return mesh_epoch_set(s->scope.epoch)?sizeof(struct mesh_frame):MESH_OFF; }
static inline size_t mesh_stream_payload(struct hdr *m,const struct mstream *s){ return s->chunk?s->chunk:mesh_pay(m)-mesh_stream_header(s); }
static inline size_t mesh_frame_decode(const void *data,size_t bytes,struct mesh_frame *frame){
  if(bytes<MESH_OFF) return 0;
  *frame=(struct mesh_frame){0}; memcpy(&frame->h,data,MESH_OFF);
  if(frame->h.k<=K_ABORTED_TX) return MESH_OFF;
  if((frame->h.k&UINT32_C(0xffff0000))!=MESH_SCOPED || bytes<sizeof *frame) return 0;
  memcpy(frame,data,sizeof *frame);
  if(!mesh_epoch_set(frame->epoch)) return 0;
  frame->h.k&=UINT32_C(0xffff); return sizeof *frame;
}
static inline size_t mesh_frame_encode(void *data,const struct mesh_frame *frame,size_t payload){
  if(!mesh_epoch_set(frame->epoch)){ memcpy(data,&frame->h,MESH_OFF); return MESH_OFF+payload; }
  struct mesh_frame out=*frame; out.h.k|=MESH_SCOPED;
  memcpy(data,&out,sizeof out); return sizeof out+payload;
}
static inline int mesh_frame_receive(uint32_t kind){
  return kind==K_DATA || kind==K_FIN || kind==K_CLOSE || kind==K_OPEN || kind==K_ABORT_RX || kind==K_ABORTED_RX || kind==K_DIGEST;
}
static inline int mesh_frame_reply(uint32_t kind){
  return kind==K_CLOSE?K_CLOSED:kind==K_ABORT_RX?K_ABORTED_TX:kind==K_ABORT_TX?K_ABORTED_RX:-1;
}
static inline size_t mesh_resident_reply(void *data,size_t bytes,uint16_t node){
  if(bytes<sizeof(struct wire)+sizeof(struct mesh_frame)) return 0;
  struct wire *wire=data; struct mesh_frame frame;
  if(wire->dst!=node || mesh_frame_decode((char*)data+sizeof *wire,bytes-sizeof *wire,&frame)!=sizeof frame ||
     frame.source!=wire->src || frame.target!=node) return 0;
  int kind=mesh_frame_reply(frame.h.k); if(frame.h.k!=K_CLOSE) return 0;
  uint16_t peer=wire->src;
  *wire=(struct wire){node,peer,0}; frame.h.k=(uint32_t)kind; frame.h.off=0;
  frame.source=node; frame.target=peer;
  return sizeof *wire+mesh_frame_encode((char*)data+sizeof *wire,&frame,0);
}
#endif
