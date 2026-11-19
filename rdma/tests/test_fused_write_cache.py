"""Exercise the production Torch generation binder without a NIC or Torch.

The tensor allocator and native plan boundary are mocks. The production
fused_turn body and FusedForm storage are extracted unchanged to check exact
views, disjoint relay generations, and absence of repeated metadata setup.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_native_lifetimes import function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'torch-mesh/torch_mesh/_stream.mm'
MOCKS = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("check"); } while (0)
namespace at {
struct Storage { int64_t elements, id; };
struct Tensor {
  std::shared_ptr<Storage> storage;
  int64_t first=0, count=0, bytes=2;
  int64_t element_size() const { return bytes; }
  Tensor narrow(int, int64_t begin, int64_t length) const {
    assert(begin>=0 && length>=0 && begin+length<=count);
    return {storage,first+begin,length,bytes};
  }
};
}
static int64_t allocations=0, preparations=0, executions=0, destroyed=0;
static int local_rank=0;
static at::Tensor region_empty(std::vector<int64_t> sizes,const at::Tensor &like,bool mps) {
  assert(mps && sizes.size()==1 && sizes[0]>=0);
  return {std::make_shared<at::Storage>(at::Storage{sizes[0],++allocations}),0,sizes[0],like.bytes};
}
struct Operand { const void *at; size_t count; int type; };
struct Group {
  bool metal; std::deque<at::Tensor> buffers;
  explicit Group(bool value):metal(value){}
  Operand moving(const at::Tensor &t) { assert(metal); buffers.push_back(t); return {&buffers.back(),(size_t)t.count,(int)t.bytes}; }
};
struct Plan { at::Tensor source,landing; size_t count,stride; int source_rank,destination; uint32_t repeats; };
using ncclMeshWritePlan_t=Plan*;
static int64_t communicator(int64_t comm) { return comm; }
static void checked(int value) { assert(!value); }
static int ncclMeshWritePrepare(const void *source,void *landing,size_t count,int type,int from,int to,
                               int64_t comm,int metal,uint32_t repeats,size_t stride,size_t packet,Plan **out) {
  assert(from!=to && comm && metal==1 && !packet && out && !*out);
  const auto a=*static_cast<const at::Tensor*>(source), b=*static_cast<const at::Tensor*>(landing);
  assert(a.bytes==type && b.bytes==type && a.count==(int64_t)count && b.count==(int64_t)count);
  const auto &binding=local_rank==from?a:b;
  assert((binding.first+count)*type+(repeats-1)*stride<=(size_t)(binding.storage->elements*type));
  *out=new Plan{a,b,count,stride,from,to,repeats}; preparations++; return 0;
}
static int ncclMeshWritePlanDestroy(Plan *p) { if(p){destroyed++;delete p;} return 0; }
'''
EXECUTE = r'''
static std::pair<std::vector<std::vector<uint64_t>>,std::vector<std::vector<uint64_t>>> execute_fused_writes(
    const PreparedWrites &plans,int64_t,int64_t,int64_t repeats,int64_t) {
  executions++;
  for(auto *p:plans.handles) {
    assert(p && p->count && p->repeats==repeats);
    if(local_rank!=p->source_rank && local_rank!=p->destination) {
      assert(p->source.storage!=p->landing.storage);
      assert(p->landing.first==0 && p->stride==p->count*p->source.bytes);
      assert(p->landing.storage->elements==(int64_t)(repeats*p->count));
    } else assert(p->source.storage==p->landing.storage && p->source.first==p->landing.first);
    for(auto *q:plans.handles) if(p!=q && local_rank!=p->source_rank && local_rank!=p->destination)
      assert(p->landing.storage!=q->landing.storage);
  }
  return {{{1,2,3,4,5,6}},{{6,5,4,3,2,1}}};
}
'''
CASES = r'''
static FusedForm form(std::vector<int64_t> counts) {
  FusedForm f; f.comm=1; f.rank=local_rank; f.ranks=counts.size(); f.channels=6;
  f.counts=counts; f.starts={0};
  for(auto n:counts)f.starts.push_back(f.starts.back()+n);
  f.slot=f.starts.back(); return f;
}
static void recurrent(int nodes,int repeats,int bytes) {
  std::vector<int64_t> counts;
  for(int r=0;r<nodes;r++)counts.push_back(2*r+3);
  for(local_rank=0;local_rank<nodes;local_rank++) {
    auto f=form(counts); at::Tensor x; x.bytes=bytes;
    const auto before=preparations, oldexec=executions;
    for(int k=0;k<4*repeats;k++) {
      assert(fused_turn(f,x,repeats)==k%repeats);
      assert(preparations-before==2*nodes*(nodes-1));
      assert(executions-oldexec==k/repeats+1);
      assert(f.pbind.size()==144 && f.lbind.size()==144);
      const auto &ring=*f.ring;
      const auto &plans=ring.plans[f.half]->handles;
      size_t i=0;
      for(int source=0;source<nodes;source++)for(int destination=0;destination<nodes;destination++)if(source!=destination) {
        const auto *p=plans[i++];
        assert(p->source_rank==source && p->destination==destination && p->count==(size_t)counts[source]);
        assert(p->source.first==f.starts[source]);
        if(local_rank==source || local_rank==destination)assert(p->stride==(size_t)(f.slot*bytes));
      }
      assert(i==plans.size());
    }
  }
}
static void empty_source_generations() {
  local_rank=0; auto f=form({3,0,7}); at::Tensor x;
  const auto before=preparations;
  std::vector<int64_t> generations;
  for(int k=0;k<9;k++) {
    assert(fused_turn(f,x,3)==k%3);
    assert(preparations-before==8*(k/3+1));
    if(k%3==0)generations.push_back(f.ring->slots[f.half][0].storage->id);
  }
  assert(generations[0]!=generations[1] && generations[0]!=generations[2]);
  auto empty=form({0,0,0}); auto count=preparations;
  assert(fused_turn(empty,x,1)==0 && preparations==count);
}
static void configured_fresh_generations() {
  local_rank=1; auto f=form({3,0,7}); at::Tensor x;
  for(int batch=0;batch<3;batch++) {
    FusedForm::Ring ring;
    fused_allocate_ring(f,ring,x,2);
    f.prepared_fresh[2].push_back(std::move(ring));
  }
  const auto prepared=preparations, allocated=allocations;
  std::vector<int64_t> generations;
  for(int i=0;i<6;i++) {
    assert(fused_turn(f,x,2)==i%2);
    assert(preparations==prepared && allocations==allocated);
    if(i%2==0)generations.push_back(f.ring->slots[f.half][0].storage->id);
  }
  assert(generations[0]!=generations[1] && generations[1]!=generations[2]);
  assert(f.prepared_fresh[2].empty());
}
int main() {
  for(int nodes:{2,3,4})for(int repeats:{1,3})for(int bytes:{2,4})recurrent(nodes,repeats,bytes);
  empty_source_generations(); configured_fresh_generations(); assert(preparations==destroyed);
  puts("generic exact WRITE generations PASS");
}
'''


class FusedWriteCache(unittest.TestCase):
    def test_production_exact_generations_and_preparation_cache(self):
        compiler = shutil.which('clang++')
        if not compiler:
            self.skipTest('clang++ required')
        source = SOURCE.read_text()
        prepared = source[source.index('struct PreparedWrites {'):source.index('\nstatic std::pair', source.index('struct PreparedWrites {'))]
        form = source[source.index('struct FusedForm {'):source.index('\nstatic auto *fused_forms_', source.index('struct FusedForm {'))]
        body = (MOCKS + prepared + form + EXECUTE + function(source, 'binding_words')
                + function(source, 'fused_allocate_ring') + function(source, 'fused_turn') + CASES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'test.cc').write_text(body)
            build = subprocess.run([compiler, '-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                                    '-fsanitize=address,undefined', str(path / 'test.cc'), '-o', str(path / 'test')],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(path / 'test')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('generic exact WRITE generations PASS', run.stdout)


if __name__ == '__main__':
    unittest.main()
