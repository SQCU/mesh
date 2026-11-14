# Exact requested-ANE operation9 diagnostic

Same compiler/ane-native-command.md input/model/placement, compile_module only. Existing original lowering records
and Builder roots are retained; public OperationInfo/ValueInfo are queried before owner cleanup. The selected
source record is _Sort, so its actual constructor interval is retained instead of inventing a generic body.

Environment: TORCH_DEVICE_BACKEND_AUTOLOAD=0; DYLD_INSERT_LIBRARIES=/Users/mdot/metal-microbench/.build/libmetal_recording.dylib;
DYLD_LIBRARY_PATH=/Users/mdot/metal-microbench/.build:/Users/mdot/dox/mesh/rdma;
PYTHONPATH=/Users/mdot/dox/mesh/compiler:/Users/mdot/dox/mesh/rdma.
Cwd /Users/mdot/dox/mesh; interpreter /Users/mdot/.venv-mesh-uv/bin/python -u.

```python
import ctypes as C
import json
from pathlib import Path
import sys
import torch
from mesh_compiler import lower_metal,native,metal,operations
from mesh_compiler.driver import compile_module
held={}
class OpInfo(C.Structure):
    _fields_=[('identity',C.c_uint64)]+[(n,C.c_uint32) for n in ('place','erased','kind','padding')]
def tensor(t):
    return {'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),'offset':t.storage_offset()}
def snapshot(f,error):
    p,lib,target=f.f_locals['constructing'],f.f_locals['compiler'],f.f_locals['self']
    lib.metal_ir_operation_info.restype,lib.metal_ir_operation_info.argtypes=None,[C.c_void_p,C.c_uint32,C.POINTER(OpInfo)]
    oi=OpInfo();lib.metal_ir_operation_info(p,9,C.byref(oi))
    out={'error':str(error),'program':p,'calls':list(f.f_locals['calls']),'values':int(lib.metal_ir_values(p)),'operations':int(lib.metal_ir_operations(p)),'operation9':{n:getattr(oi,n) for n,_ in oi._fields_},'records':[]}
    b,leaves,module=held[p]
    for node,emissions in leaves.items():
        for leaf in emissions:
            if not leaf.first<=9<leaf.end:continue
            g=leaf.record;r={'source_node':node.name,'target':str(node.target),'arguments':str(node.args),'record_type':type(g).__name__,'record_node':g.node,'first':leaf.first,'end':leaf.end,'operands':[],'regions':{}}
            if isinstance(g,operations._Sort):
                r['sort']={'axis':g.axis,'k':g.k,'descending':g.descending,'bounded':g.extent is not None}
                r['constructed_operations']=[]
                for oid in range(leaf.first,leaf.end):
                    info=OpInfo();lib.metal_ir_operation_info(p,oid,C.byref(info));r['constructed_operations'].append({'operation':oid,**{n:getattr(info,n) for n,_ in info._fields_}})
                for role in ('x','values','indices','extent'):
                    t=getattr(g,role)
                    if t is None:continue
                    root=b.roots[operations._key(t)];v=metal.ValueInfo();lib.metal_ir_value_info(p,root,C.byref(v))
                    r['operands'].append({'role':role,'root':root,**tensor(t),'native':{n:getattr(v,n) for n,_ in v._fields_}})
                out['records'].append(r)
                continue
            r.update(sizes=g.sizes,iterators=g.iterators,extents=g.extents)
            for role in ('inputs','outputs','symbols'):
                for i,item in enumerate(getattr(g,role,())):
                    t=item if role=='symbols' else item[0]
                    root=b.roots[operations._key(t)];v=metal.ValueInfo();lib.metal_ir_value_info(p,root,C.byref(v))
                    r['operands'].append({'role':role,'ordinal':i,'root':root,'map':None if role=='symbols' else item[1],**tensor(t),'native':{n:getattr(v,n) for n,_ in v._fields_}})
            for name in ('body','combine','finalize'):
                region=getattr(g,name,None)
                if region is not None:
                    r['regions'][name]={'code':region.code,'kinds':region.kinds,'binary':[{ 'index':i,'opcode':c[1],'name':next((k for k,v in operations.BINARY.items() if v==c[1]),None),'kind':region.kinds[i],'lhs':{'index':c[2],'code':region.code[c[2]],'kind':region.kinds[c[2]]},'rhs':{'index':c[3],'code':region.code[c[3]],'kind':region.kinds[c[3]]}} for i,c in enumerate(region.code) if c[0]=='binary']}
            out['records'].append(r)
    out['body_keys']=[{'function':i,'key':str(b.key),'program':b.machine.program if b.machine else None} for i,b in enumerate(target.bodies)]
    Path('output_data/compiler-workgroup-20261114/executor/ane-op9.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'captured':True,'calls':out['calls'],'records':len(out['records']),'error':str(error)}),flush=True)
def trace(f,event,arg):
    if f.f_code is lower_metal.construct.__code__ and event=='return' and arg is not None:held[arg.program]=(f.f_locals['builder'],f.f_locals['leaves'],f.f_locals['module'])
    if f.f_code is native.Target.constructed.__code__ and event=='exception':
        sys.settrace(None);snapshot(f,arg[1])
    return trace
class Hermitian(torch.nn.Module):
    def forward(self,a):return torch.linalg.eigh(a)
a=torch.tensor([[2.,1.],[1.,4.]],dtype=torch.float32)
sys.settrace(trace)
try:compile_module(Hermitian(),(a,),{'p':(0,'ane')},lambda node:'p')
except OSError as e:print('Expected compile stop:',e,flush=True)
finally:sys.settrace(None)
```
