"""Actual NCCL membership realization and host arithmetic over the real compiler.

Only memory/device boundaries are local fixtures. Native function bodies are
extracted verbatim and linked with the production logical/physical compilers.
The transaction evaluator comes from the existing physical-routing test.
"""
from pathlib import Path
import shutil
import re
import subprocess
import tempfile
import unittest

from test_native_lifetimes import function

ROOT = Path(__file__).resolve().parents[1]


def structure(source, name):
    start = source.index('struct '+name+' {')
    return source[start:source.index('};', start)+2]


HEAD = r'''
#include "nccl.h"
#include "mesh.h"
#include "mesh-plan.h"
#include "mesh-write-routes.h"
#include "nccl-mesh-metal.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
struct fixture_context { struct hdr *M; };
struct attach { struct fixture_context context; };
struct ncclComm { int rank,nranks; const char *name; uint64_t page;
  struct mesh_fabric fabric; struct mesh_paths *writes,*tree_hops; struct attach *attach; uint32_t count;
  struct mesh_link_map map; ncclMeshLinkRate *rates; int nrates,scheduled; ncclMeshScheduleStats schedule;
  struct { uint32_t peer,cable; } *channels; };
void *metal_contents(void *buffer) { return buffer; }
'''

CASES = r'''
static struct attach attachment;
static struct ncclComm comms[N];
static struct call calls[N];
static struct entry selected;
static uint32_t tree_roots[N*N],tree_parents[N*N*N];
static double tree_weights[N*N];
static int64_t *source[N];
static uint64_t input_mask=5,output_mask=8;
static uint32_t certified_aliases;

static void declared_fixture(uint32_t what,int whole){
  for(uint32_t r=0;r<N;r++){
    calls[r]=(struct call){.comm=comms+r,.what=(int)what,.root=3,.count=ELEMENTS,.type=ncclInt64,.op=ncclAvg,.post=1};
    ncclMeshCall descriptor={.contributors=&input_mask,.recipients=&output_mask};
    assert(!declared_members(calls+r,&descriptor));
    assert(calls[r].participants==(uint32_t)__builtin_popcountll(input_mask));
    assert(!!call_contributes(calls+r)==!!(input_mask>>r&1));
    assert(!!call_receives(calls+r)==!!(output_mask>>r&1));
    source[r]=(int64_t *)((char *)attachment.context.M+4096+r*16384);
    for(uint32_t x=0;x<ELEMENTS;x++)source[r][x]=100*r+x;
    calls[r].send=(input_mask>>r&1)?source[r]:NULL;
    calls[r].send_at=(struct where){NULL,(size_t)calls[r].send};
    assert(sent_bytes(calls+r)==((input_mask>>r&1)?ELEMENTS*8:0));
  }
  selected.whole=whole;
}

static void free_calls(void){
  for(uint32_t r=0;r<N;r++){
    steps_free(calls[r].bound);free(calls[r].contributors);free(calls[r].recipients);
    free(calls[r].segments);free(calls[r].complement);calls[r]=(struct call){0};
  }
}

static void expected_physical(uint32_t what,int whole,struct mesh_routed_program *physical){
  uint32_t parts,moves;mesh_compile_room(&selected.trees,&parts,&moves);
  uint32_t *segment=calloc(parts,sizeof *segment),*first=calloc(N+1,sizeof *first);
  double *weights=calloc(parts,sizeof *weights);struct mesh_move *move=calloc(moves,sizeof *move);
  struct mesh_program program;
  assert(!mesh_compile_members(&selected.trees,what,3,whole,&input_mask,&output_mask,&program,segment,weights,first,move));
  uint32_t offsets[N+1]={0},initialized[N];struct mesh_step *logical=calloc(moves,sizeof *logical);
  for(uint32_t r=0;r<N;r++){
    initialized[r]=1+program.scratch;
    offsets[r+1]=offsets[r]+mesh_program_steps_cut(&program,r,(struct mesh_operand){ncclInt64+1,8,ELEMENTS},NULL,1,logical+offsets[r]);
  }
  assert(!mesh_routed_compile(&fabric,comms[0].tree_hops,offsets,logical,initialized,1,physical));
  free(segment);free(first);free(weights);free(move);free(logical);
}

static uint64_t physical_signature(const struct mesh_routed_program *p){
  uint64_t sum=0;
  for(uint32_t r=0;r<N;r++)for(uint32_t k=0;k<p->rank[r].count;k++){
    const struct mesh_transfer *t=p->rank[r].transfer+k;
    sum=sum*UINT64_C(1099511628211)+t->step.first+17*t->step.piece.elements+31*t->step.buffer+7*t->step.op+t->cable;
  }
  return sum;
}

static uint64_t lower_and_run(uint32_t what,int whole){
  declared_fixture(what,whole);
  struct mesh_routed_program actual={.nodes=N,.rank=calloc(N,sizeof *actual.rank)},expected;
  expected_physical(what,whole,&expected);
  int64_t **data[N];
  for(uint32_t r=0;r<N;r++){
    assert(!collective_steps(calls+r,&selected,(struct mesh_operand){ncclInt64+1,8,ELEMENTS},NULL));
    struct steps *s=calls[r].bound;const struct mesh_routed_rank *e=expected.rank+r;
    assert(s&&s->count==e->count&&s->buffers==e->buffers&&s->initialized==e->initialized);
    assert(!memcmp(s->demand,expected.demand,sizeof(uint64_t)*N*N));
    struct mesh_routed_rank *a=actual.rank+r;
    a->count=s->count;a->buffers=s->buffers;a->initialized=s->initialized;
    a->extent=malloc(s->buffers*sizeof *a->extent);memcpy(a->extent,s->extent,s->buffers*sizeof *a->extent);
    a->transfer=calloc(s->count+1,sizeof *a->transfer);
    a->consume=malloc((s->count+1)*sizeof *a->consume);
    memcpy(a->consume,s->consume_order,s->count*sizeof *a->consume);
    data[r]=calloc(s->buffers,sizeof *data[r]);
    for(uint32_t k=0;k<s->count;k++){
      assert(s->cable[k]==e->transfer[k].cable&&!memcmp(s->steps+k,&e->transfer[k].step,sizeof *s->steps));
      a->transfer[k]=(struct mesh_transfer){.step=s->steps[k],.cable=s->cable[k]};
    }
    for(uint32_t b=0;b<s->buffers;b++){
      const size_t count=s->extent[b]/8+ELEMENTS;
      data[r][b]=malloc(count*sizeof(int64_t));
      for(size_t x=0;x<count;x++)data[r][b][x]=-123456789;
      /* The native stage function must leave a relay-only rank untouched. */
      if(b<s->initialized)stage(calls+r,(unsigned char *)data[r][b]);
      if(!(input_mask>>r&1))for(size_t x=0;x<count;x++)assert(data[r][b][x]==-123456789);
    }
  }
  /* Exercise production lifetime binding on the ordinary generated plan,
     including reduce-then-broadcast COPY reuse of partial-result storage. */
  for(uint32_t r=0;r<N;r++){
    struct steps *st=calls[r].bound;comms[r].count=E;
    for(uint32_t b=0;b<st->buffers;b++)st->buffer[b]=(struct where){NULL,(size_t)data[r][b]};
    calls[r].direct=plan_direct(calls+r,st->buffer[0]);
    struct schedule schedules[E]={0};
    for(uint32_t e=0;e<E;e++){
      schedules[e].out=calloc(st->count+1,sizeof(struct message));
      schedules[e].in=calloc(st->count+1,sizeof(struct message));
    }
    for(uint32_t k=0;k<st->count;k++){
      const struct mesh_step *step=st->steps+k;struct schedule *q=schedules+st->cable[k];
      const struct where origin=step->op==MESH_STEP_SEND&&st->read&&named(st->read[k])?st->read[k]:st->buffer[step->buffer];
      struct message m={.c=calls+r,.step=k,.offset=step->first*8,.bytes=step->piece.elements*8,
        .data=shifted(origin,step->first*8),.end={k+1}};
      if(step->op==MESH_STEP_SEND)q->out[q->outs++]=m;else q->in[q->ins++]=m;
    }
    for(uint32_t e=0;e<E;e++)for(size_t i=0;i<schedules[e].ins;i++){
      const struct message *m=schedules[e].in+i;
      if(st->steps[m->step].op!=MESH_STEP_COPY)continue;
      for(uint32_t q=0;q<E;q++)for(size_t j=0;j<schedules[q].outs;j++){
        const struct message *o=schedules[q].out+j;
        if(o->step>=m->step)continue;
        const uintptr_t a=(uintptr_t)address_of(m->data),b=(uintptr_t)address_of(o->data);
        if(a>=b+o->bytes || b>=a+m->bytes)continue;
        int proved=0;
        for(uint32_t k=st->retired_first[m->step];k<st->retired_first[m->step+1];k++)proved|=st->retired[k]==o->step;
        assert(proved);certified_aliases++;
      }
    }
    ncclResult_t result=bind_retirements(comms+r,schedules);
    if(result)fprintf(stderr,"rank %u what %u whole %d lifetime binding: %d\n",r,what,whole,result);
    assert(!result);
    for(uint32_t e=0;e<E;e++){
      for(size_t k=0;k<schedules[e].ins;k++)free(schedules[e].in[k].retire);
      free(schedules[e].out);free(schedules[e].in);
    }
  }
  if(comms[0].tree_hops==paths)simulate(&actual,data);
  else {
    uint32_t used=0;
    uint64_t bytes=0,demand=0;
    for(uint32_t r=0;r<N;r++){
      const struct mesh_routed_rank *rank=actual.rank+r;
      assert(rank->buffers==rank->initialized);
      for(uint32_t k=0;k<rank->count;k++){
        const struct mesh_transfer *t=rank->transfer+k;
        used|=1u<<t->cable;
        if(t->step.op==MESH_STEP_SEND)bytes+=t->step.piece.elements*t->step.piece.element_bytes;
      }
      for(uint32_t d=0;d<N;d++)demand+=expected.demand[r*N+d];
    }
    assert(bytes==demand);simulate_cables(&actual,data,used);
  }
  for(uint32_t r=0;r<N;r++){
    int64_t output[ELEMENTS];for(uint32_t x=0;x<ELEMENTS;x++)output[x]=-1;
    calls[r].sum=(struct where){NULL,(size_t)data[r][0]};
    calls[r].recv=call_receives(calls+r)?output:NULL;
    finish(calls+r);
    int64_t mean=0;uint32_t participants=0;
    for(uint32_t q=0;q<N;q++)if(input_mask>>q&1){mean+=100*q;participants++;}
    mean/=participants;
    for(uint32_t x=0;x<ELEMENTS;x++)assert(output[x]==(call_receives(calls+r)?mean+(int64_t)x:-1));
    for(uint32_t b=0;b<calls[r].bound->buffers;b++)free(data[r][b]);free(data[r]);
  }
  const uint64_t signature=physical_signature(&actual);
  mesh_routed_free(&actual);mesh_routed_free(&expected);free_calls();return signature;
}

static void inactive_nullable_and_avg(void){
  declared_fixture(MESH_ALLREDUCE,1);
  void *guard=mmap(NULL,4096,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);assert(guard!=MAP_FAILED);
  struct call *inactive=calls+1;
  inactive->send=guard;inactive->send_at=(struct where){NULL,(size_t)guard};inactive->recv=guard;
  assert(sent_bytes(inactive)==0);
  stage(inactive,guard);finish(inactive);
  struct steps none={0};inactive->bound=&none;
  assert(!plan_direct(inactive,(struct where){NULL,(size_t)guard}));inactive->bound=NULL;
  /* Floating Avg's premultiplier and integer Avg's post-division both count
     contributors, not total physical ranks or result recipients. */
  struct call average={.comm=comms,.type=ncclFloat32,.op=ncclAvg,.premultiplied=1};
  call_bind_roles(&average,calls);float scale;memcpy(&scale,average.pre,4);assert(scale==0.5f);
  average.send=source[0];average.count=1;
  munmap(guard,4096);free_calls();
}

static void direct_source_binding(void){
  declared_fixture(MESH_ALLREDUCE,1);
  struct call *c=calls;
  assert(!collective_steps(c,&selected,(struct mesh_operand){ncclInt64+1,8,ELEMENTS},NULL));
  const struct where output={NULL,(size_t)((char *)attachment.context.M+262144)};
  assert(plan_direct(c,output));
  int direct=0;
  for(uint32_t k=0;k<c->bound->count;k++)if(c->bound->read[k].offset){
    assert(c->bound->steps[k].op==MESH_STEP_SEND||c->bound->fresh[k]);
    assert(address_of(c->bound->read[k])==(unsigned char *)source[0]);direct++;
  }
  assert(direct&&c->complements==0);free_calls();
}

static void invalid_roles(void){
  uint64_t none=0,only0=1,only3=8;
  struct call c={.comm=comms,.what=MESH_REDUCE,.root=3,.type=ncclInt64};
  ncclMeshCall d={.contributors=&none,.recipients=&only3};
  assert(declared_members(&c,&d)==ncclInvalidArgument);free(c.contributors);free(c.recipients);
  c=(struct call){.comm=comms,.what=MESH_BROADCAST,.root=3,.type=ncclInt64};d.contributors=&only0;
  assert(declared_members(&c,&d)==ncclInvalidArgument);free(c.contributors);free(c.recipients);
  c=(struct call){.comm=comms,.what=MESH_ALLGATHER,.count=3,.type=ncclInt64};
  assert(!declared_members(&c,&d));assert(c.total==3&&c.segments[0]==3);
  for(uint32_t r=1;r<N;r++)assert(!c.segments[r]);
  free(c.contributors);free(c.recipients);free(c.segments);
}

static void explicit_program_validation(void){
  const struct mesh_trees trees=selected.trees;
  uint32_t parts,moves;mesh_compile_room(&trees,&parts,&moves);
  uint32_t *segment=calloc(parts,sizeof *segment),*first=calloc(N+1,sizeof *first);
  double *weight=calloc(parts,sizeof *weight);struct mesh_move *move=calloc(moves,sizeof *move);
  assert(segment&&first&&weight&&move);
  assert(!mesh_compile_members(&trees,MESH_REDUCE,3,0,&input_mask,&output_mask,
                              &selected.program,segment,weight,first,move));
  selected.trees=(struct mesh_trees){0};
  declared_fixture(MESH_REDUCE,0);
  for(uint32_t r=0;r<N;r++)assert(!collective_steps(calls+r,&selected,(struct mesh_operand){ncclInt64+1,8,ELEMENTS},NULL));
  free_calls();
  /* The same explicit raw-program path must reject an all-rank program when
     its declaration removes inputs. It cannot recompile around the custom
     program or allow stage(NULL) followed by reading uninitialized storage. */
  assert(!mesh_compile(&trees,MESH_REDUCE,3,0,NULL,&selected.program,segment,weight,first,move));
  declared_fixture(MESH_REDUCE,0);
  for(uint32_t r=0;r<N;r++){
    assert(collective_steps(calls+r,&selected,(struct mesh_operand){ncclInt64+1,8,ELEMENTS},NULL)!=ncclSuccess);
    assert(!calls[r].bound);
  }
  free_calls();program_free(&selected.program);selected.trees=trees;
}

int main(void){
  uint32_t link[E][2];uint64_t capacity[E];
  for(uint32_t e=0;e<E;e++){memcpy(link[e],cables[e].node,sizeof link[e]);capacity[e]=cables[e].bandwidth;}
  struct mesh_link_map map={.kind=MESH_LINKS_GRAPH,.nodes=N,.links=E,.link=link,.bandwidth=capacity};
  uint32_t count=mesh_trees_pack(&map,NULL,0,N,tree_roots,tree_parents,tree_weights);assert(count>N);
  for(uint32_t t=0;t<count;t++)tree_weights[t]=(double)((3*t+1)%7)-4.0;
  selected.trees=(struct mesh_trees){N,count,tree_roots,tree_parents,tree_weights};
  for(uint32_t a=0;a<N;a++)for(uint32_t b=0;b<N;b++)if(a!=b){
    paths[a*N+b]=(struct mesh_paths){.first=path_first[a*N+b],.cable=path_edges[a*N+b],.log_weight=weights[a*N+b]};
    enumerate(a,b,a,1u<<a,0);
  }
  attachment.context.M=calloc(1,1u<<20);assert(attachment.context.M);
  attachment.context.M->length=1u<<20;attachment.context.M->data_off=4096;
  for(uint32_t r=0;r<N;r++)comms[r]=(struct ncclComm){.rank=(int)r,.nranks=N,.name="member fixture",.page=1,
      .fabric=fabric,.writes=paths,.tree_hops=paths,.attach=&attachment};
  const uint64_t weighted=lower_and_run(MESH_REDUCE,0); /* root3 has no source operand */
  double saved[N*N];memcpy(saved,tree_weights,count*sizeof *saved);
  for(uint32_t t=0;t<count;t++)tree_weights[t]=-3*tree_weights[t];
  assert(weighted!=lower_and_run(MESH_REDUCE,0)); /* Changing selected tree shares changes actual fragments. */
  memcpy(tree_weights,saved,count*sizeof *saved);
  const uint64_t phased=lower_and_run(MESH_ALLREDUCE,0),whole=lower_and_run(MESH_ALLREDUCE,1);
  assert(phased!=whole); /* Native specialization retained the configured form. */
  inactive_nullable_and_avg();direct_source_binding();invalid_roles();explicit_program_validation();
  input_mask=output_mask=(1u<<N)-1;certified_aliases=0;
  lower_and_run(MESH_ALLREDUCE,0);lower_and_run(MESH_ALLREDUCE,1);assert(certified_aliases);
  struct mesh_paths hops[N*N];
  for(uint32_t p=0;p<N*N;p++)hops[p]=mesh_tree_hop(paths+p);
  for(uint32_t r=0;r<N;r++)comms[r].tree_hops=hops;
  certified_aliases=0;lower_and_run(MESH_ALLREDUCE,0);lower_and_run(MESH_ALLREDUCE,1);assert(certified_aliases);
  free(attachment.context.M);
  puts("native roles, selected trees/phase, all cables, nullable inputs, contributor Avg PASS");return 0;
}
'''


class NativeMembers(unittest.TestCase):
    def test_native_membership_reaches_common_compiler_and_host_functions(self):
        self.compile_fixture()

    def test_live_ragged_masked_average_byte_lifetimes(self):
        self.compile_fixture(live=True)

    def compile_fixture(self, live=False):
        compiler = shutil.which('clang') or shutil.which('cc')
        if compiler is None:
            self.skipTest('C compiler unavailable')
        native = (ROOT/'nccl-mesh.c').read_text()
        routed_test = (ROOT/'tests/routed.c').read_text()
        arithmetic = native[native.index('enum { WHAT_COUNTED'):native.index('/* -- communicators')]
        declarations = '\n'.join(structure(native, name) for name in ('where', 'entry', 'scatter_schedule', 'steps', 'call', 'message', 'schedule', 'spans'))
        functions = ('named', 'shifted', 'address_of', 'bind_retirements', 'program_free', 'reducing', 'unit_of', 'counted', 'rank_member',
            'call_contributes', 'call_receives', 'call_participants', 'call_bind_roles', 'elements', 'seg_first', 'segment_at',
            'call_name', 'sent_bytes', 'scatter_release', 'steps_free', 'bind_routed_steps', 'routed_steps',
            'scatter_steps', 'scheduled_program', 'collective_steps',
            'stage', 'finish', 'in_region', 'spans_meet', 'spans_add', 'plan_direct', 'declared_members')
        body = '\n'.join(function(native, name) for name in functions)
        simulation = routed_test[routed_test.index('enum { N='):routed_test.index('static void check_collective')]
        self.assertIn('buffers[r][s->buffer][s->first+x]+=staged[r][k][x];', simulation)
        simulation = simulation.replace('buffers[r][s->buffer][s->first+x]+=staged[r][k][x];',
            'combine_into(ncclInt64,COMBINE_SUM,&buffers[r][s->buffer][s->first+x],&staged[r][k][x],1);')
        cases = CASES
        head = HEAD
        if live:
            start = simulation.index('enum { N=')
            end = simulation.index('static const struct mesh_fabric')
            simulation = simulation[:start] + '''enum { N=3, E=6, ELEMENTS=131101 };
static const struct mesh_cable cables[E]={
  {1,80000000000ull,{0,1}},{2,80000000000ull,{0,1}},
  {3,80000000000ull,{0,1}},{4,80000000000ull,{0,1}},
  {5,80000000000ull,{0,2}},{6,80000000000ull,{1,2}}
};
''' + simulation[end:]
            cases = cases[:cases.index('int main(void){')] + r'''
int main(void){
  uint32_t link[E][2];uint64_t capacity[E];
  for(uint32_t e=0;e<E;e++){memcpy(link[e],cables[e].node,sizeof link[e]);capacity[e]=cables[e].bandwidth;}
  struct mesh_link_map map={.kind=MESH_LINKS_GRAPH,.nodes=N,.links=E,.link=link,.bandwidth=capacity};
  uint32_t count=mesh_trees_pack(&map,NULL,0,N,tree_roots,tree_parents,tree_weights);assert(count);
  selected.trees=(struct mesh_trees){N,count,tree_roots,tree_parents,tree_weights};
  struct mesh_paths *defaults=NULL;assert(!mesh_write_routes_default(&fabric,&defaults));
  memcpy(paths,defaults,sizeof paths);free(defaults);
  attachment.context.M=calloc(1,1u<<24);assert(attachment.context.M);
  attachment.context.M->length=1u<<24;attachment.context.M->data_off=4096;
  for(uint32_t r=0;r<N;r++)comms[r]=(struct ncclComm){.rank=(int)r,.nranks=N,.name="live fixture",.page=16384,
      .fabric=fabric,.writes=paths,.tree_hops=paths,.attach=&attachment};
  input_mask=output_mask=7;lower_and_run(MESH_ALLREDUCE,0);
  input_mask=3;output_mask=4;lower_and_run(MESH_ALLREDUCE,1);
  input_mask=3;output_mask=5;lower_and_run(MESH_ALLREDUCE,0);
  for(uint32_t p=0;p<N*N;p++){free((void*)paths[p].first);free((void*)paths[p].cable);free((void*)paths[p].log_weight);}
  free(attachment.context.M);puts("live ragged masked Avg PASS");return 0;
}
'''
            cases = cases.replace('4096+r*16384', '4096+r*2097152').replace('NULL,1,logical', 'NULL,16384,logical')
            cases = cases.replace('initialized,1,physical', 'initialized,16384,physical')
            cases = cases.replace('.root=3', '.root=0').replace('what,3,whole', 'what,0,whole')
            # Match the live operand type and byte width, retaining the real
            # production bindings and delayed-DMA arithmetic evaluator.
            simulation = re.sub(r'\bint64_t\b', 'int32_t', simulation)
            cases = re.sub(r'\bint64_t\b', 'int32_t', cases)
            for old, new in [('ncclInt64', 'ncclInt32'),
                             ('*8', '*4'), ('/8', '/4'), (',8,', ',4,')]:
                simulation = simulation.replace(old, new)
                cases = cases.replace(old, new)
        with tempfile.TemporaryDirectory(prefix='mesh-native-members-') as directory:
            source, executable = Path(directory)/'members.c', Path(directory)/'members'
            source.write_text(head+arithmetic+declarations+body+simulation+cases)
            built = subprocess.run([compiler, '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                '-g', '-fsanitize=address,undefined', '-I', str(ROOT), str(source),
                str(ROOT/'mesh-collective.c'), str(ROOT/'mesh-routed.c'), str(ROOT/'mesh-fabric.c'),
                '-lm', '-o', str(executable)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout+ran.stderr)
            self.assertIn('live ragged masked Avg PASS' if live else 'contributor Avg PASS', ran.stdout)


if __name__ == '__main__':
    unittest.main()
