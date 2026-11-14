# Exact requested-ANE root14 diagnostic command

The same compiler/ane-native-command.md input/model/placement; compile_module only. A temporary sys.settrace hook
collects original lowering correspondence at construct return and reads existing public metadata at the live
Target.constructed exception, before Owned cleanup. No added numerical operations, standalone row execution or
alternate evaluator. The diagnostic source below is retained as command evidence, not a test program.

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
from mesh_compiler import lower_metal, native, metal, operations
from mesh_compiler.driver import compile_module

held = {}
class OpInfo(C.Structure):
    _fields_ = [('identity', C.c_uint64)] + [(n, C.c_uint32) for n in ('place','erased','kind','padding')]

def describe_tensor(t):
    return {'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),'offset':t.storage_offset()}

def snapshot(frame, error):
    local = frame.f_locals
    p, lib, target = local['constructing'], local['compiler'], local['self']
    lib.metal_ir_value_shape.restype, lib.metal_ir_value_shape.argtypes = C.c_uint32, [C.c_void_p,C.c_uint32,C.POINTER(C.c_uint64),C.c_uint32]
    lib.metal_ir_value_layout.restype, lib.metal_ir_value_layout.argtypes = C.c_uint32, [C.c_void_p,C.c_uint32,C.POINTER(metal.Term),C.c_uint32,C.POINTER(C.c_int64)]
    lib.metal_ir_operation_info.restype, lib.metal_ir_operation_info.argtypes = None, [C.c_void_p,C.c_uint32,C.POINTER(OpInfo)]
    result = {'error':str(error),'program':p,'calls':list(local['calls']),'values':int(lib.metal_ir_values(p)),'operations':int(lib.metal_ir_operations(p)),'aliases':[],'source':[]}
    for v in range(result['values']):
        info=metal.ValueInfo();lib.metal_ir_value_info(p,v,C.byref(info))
        if info.storage != 14: continue
        n=lib.metal_ir_value_shape(p,v,None,0); shape=(C.c_uint64*n)();lib.metal_ir_value_shape(p,v,shape,n)
        offset=C.c_int64();n=lib.metal_ir_value_layout(p,v,None,0,C.byref(offset));terms=(metal.Term*n)();lib.metal_ir_value_layout(p,v,terms,n,C.byref(offset))
        result['aliases'].append({'value':v,**{name:getattr(info,name) for name,_ in info._fields_},'scalar_name':operations.KINDS.get(info.scalar),'shape':list(shape),'view_offset':offset.value,'layout':[{name:getattr(t,name) for name,_ in t._fields_} for t in terms]})
    built=held.get(p)
    if built:
        builder, leaves, module = built
        for node, emissions in leaves.items():
            for leaf in emissions:
                refs=[]
                for role in ('inputs','outputs'):
                    for ordinal, item in enumerate(getattr(leaf.record,role,())):
                        t=item[0]
                        if builder.roots.get(operations._key(t)) == 14:
                            refs.append({'role':role,'ordinal':ordinal,**describe_tensor(t)})
                if not refs: continue
                records=[]
                for op in range(leaf.first,leaf.end):
                    info=OpInfo();lib.metal_ir_operation_info(p,op,C.byref(info));records.append({'operation':op,**{name:getattr(info,name) for name,_ in info._fields_}})
                val=node.meta.get('val')
                result['source'].append({'node':node.name,'target':str(node.target),'arguments':str(node.args),'source_meta':describe_tensor(val) if isinstance(val,torch.Tensor) else str(val),'record':type(leaf.record).__name__,'record_node':leaf.record.node,'emitted':records,'operands':refs})
    result['body_keys']=[{'function':i,'key':str(b.key),'program':b.machine.program if b.machine else None} for i,b in enumerate(target.bodies)]
    Path('output_data/compiler-workgroup-20261114/executor/ane-root14.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'captured':True,'aliases':len(result['aliases']),'source_records':len(result['source']),'error':result['error']}),flush=True)

def trace(frame,event,arg):
    if frame.f_code is lower_metal.construct.__code__ and event=='return' and arg is not None:
        held[arg.program]=(frame.f_locals['builder'],frame.f_locals['leaves'],frame.f_locals['module'])
    if frame.f_code is native.Target.constructed.__code__ and event=='exception':
        sys.settrace(None)
        snapshot(frame,arg[1])
    return trace

class Hermitian(torch.nn.Module):
    def forward(self,a): return torch.linalg.eigh(a)
a=torch.tensor([[2.,1.],[1.,4.]],dtype=torch.float32)
sys.settrace(trace)
try:
    compile_module(Hermitian(),(a,),{'p':(0,'ane')},lambda node:'p')
except OSError as e:
    print('Expected compile stop:',e,flush=True)
finally:
    sys.settrace(None)
```
