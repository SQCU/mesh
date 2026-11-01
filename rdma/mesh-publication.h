#pragma once
#include <stdint.h>
#include "metal-capabilities.h"

/* Counts have concurrent contributors. Publishing a smaller completed prefix
   cannot overwrite a newer one. Payload stores precede this release. The
   Metal spelling in mesh.metal implements this same maximum over the word. */
static inline void mesh_publish(volatile uint64_t *word,uint64_t value) {
    uint64_t prior=__atomic_load_n(word,__ATOMIC_RELAXED);
    while(prior<value && !__atomic_compare_exchange_n(word,&prior,value,0,__ATOMIC_RELEASE,__ATOMIC_RELAXED)){}
}

/* Native code consumes the same count pages and diagnostic fields as mesh.metal.
   All pointers and thresholds are prepared operands. These functions observe
   actual remote production; they never choose tasks or reconstruct a schedule.
   Preparation supplies a positive effective load budget. Its local counter
   begins here, when the numerical leaf first observes the needed page. */
static inline void mesh_step_observe(volatile uint64_t *word,uint64_t target,void *violation,
                                    uint32_t polls,uint32_t channel,uint64_t entry) {
    uint64_t *v=(uint64_t *)violation;
    uint64_t seen;uint32_t at=0;
    do {
        seen=__atomic_load_n(word,__ATOMIC_ACQUIRE);
    } while(seen<target && ++at<polls && ((at%MESH_STOP_POLL_PERIOD)!=0 ||
              !violation || !__atomic_load_n((uint32_t *)(v+1),__ATOMIC_RELAXED)));
    if(!polls || (seen!=UINT64_MAX && seen>=target))return;
    const uint64_t cause=seen==UINT64_MAX?MESH_OBSERVATION_LOST:
        __atomic_load_n(v+6,__ATOMIC_RELAXED)?MESH_OBSERVATION_REVOKED:MESH_OBSERVATION_BOUND;
    if(__atomic_exchange_n((uint32_t *)(v+1),1,__ATOMIC_RELAXED))return;
    v[0]=cause;v[3]=channel;v[4]=polls;v[5]=seen;
    mesh_publish(v+2,entry+1);
}

static inline uint32_t mesh_graph_prefix(uint32_t *state,uint32_t head,uint32_t count,uint32_t *prefix) {
    uint32_t at=__atomic_load_n(prefix,__ATOMIC_ACQUIRE);
    while(at<count && __atomic_load_n(state+head+at,__ATOMIC_ACQUIRE)==2) {
        uint32_t expected=at;
        if(__atomic_compare_exchange_n(prefix,&expected,at+1,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))at++;
        else at=expected;
    }
    return at;
}
#ifdef __cplusplus
/* Publish is a compiled functor. No callback lookup occurs in the numerical loop. */
template<class Publish> static inline void mesh_graph_settle(uint32_t *state,uint32_t head,uint32_t count,
                                                            uint32_t *prefix,Publish &publish) {
    const uint32_t through=mesh_graph_prefix(state,head,count,prefix);
    if(through)publish.at(through);
}
#endif
