"""The tile kernels (tests/route/*.metal) as MLX kernels, from the same source text: each `kernel void NAME(...)` of a
file becomes an mx.fast.metal_kernel whose body is the kernel's own body, and whose header is the file with its kernels
taken out and its local includes inlined (tile.h, int8mm.h, scan.h, ...), compiled with ROUTE_MLX defined.

A kernel parameter is bound by its declaration:
  device const T *x            an input array x
  device T *x                  an output array x
  constant S &x / constant T *x   an input array x (uint32 words: the struct as the kernels' C layout), read through
                               device memory (ROUTE_TBL: the helpers' tables in device memory)
  uint t [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]], ...   MLX's own, renamed
fwd.metal's function constants (STORE, STAGE) are template parameters.

Math: the products' kernels without contraction of a * b + c into fma (act.h's bit-for-bit contract with forward.c,
which is compiled with -ffp-contract=off); credit.metal in fast math, as built for the Metal library."""
import os
import re

import mlx.core as mx

ROUTE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/route

_ATTR = {'thread_position_in_grid': 3, 'thread_position_in_threadgroup': 3, 'threadgroup_position_in_grid': 3,
         'threads_per_threadgroup': 3, 'threadgroups_per_grid': 3, 'thread_index_in_threadgroup': 1,
         'simdgroup_index_in_threadgroup': 1, 'thread_index_in_simdgroup': 1}


def _inline(name, seen):
    out = []
    for line in open(os.path.join(ROUTE, name)).read().splitlines():
        m = re.match(r'\s*#include\s+"([^"]+)"', line)
        if m:
            if m.group(1) not in seen:
                seen.add(m.group(1)); out.append(_inline(m.group(1), seen))
            continue
        out.append(line)
    return '\n'.join(out)


def _match(text, i, open_, close):
    """the index past the bracket matching the one at text[i]"""
    depth = 0
    for k in range(i, len(text)):
        depth += (text[k] == open_) - (text[k] == close)
        if depth == 0:
            return k + 1
    raise ValueError('unbalanced')


def _split_params(text):
    out, depth, cur = [], 0, ''
    for ch in text:
        depth += ch in '([' ; depth -= ch in ')]'
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Source:
    """a .metal file: its kernels (name -> (params, body)) and the header the rest of it makes"""

    def __init__(self, name):
        text = _inline(name, set())
        self.kernels, rest, at = {}, [], 0
        for m in re.finditer(r'\bkernel\s+void\s+(\w+)\s*\(', text):
            p0 = m.end() - 1; p1 = _match(text, p0, '(', ')')
            b0 = text.index('{', p1); b1 = _match(text, b0, '{', '}')
            self.kernels[m.group(1)] = (_split_params(text[p0 + 1:p1 - 1]), text[b0 + 1:b1 - 1])
            rest.append(text[at:m.start()]); at = b1
        rest.append(text[at:])
        self.header = ''.join(rest) + '\n'


_sources = {}


class Kernel:
    """one kernel of a file as an MLX kernel; call with its inputs (name -> array), its outputs (name -> (shape,
    dtype)), the grid in threadgroups and the threads a threadgroup; returns the outputs in the kernel's order"""

    def __init__(self, file, name, template=(), fast=False, tag='', defines=''):
        src = _sources.setdefault(file, Source(file))
        params, body = src.kernels[name]
        self.inputs, self.outputs, pro = [], [], []
        for p in params:
            decl, attr = re.match(r'(.*?)\s*\[\[\s*(\w+)', p).groups()
            nm = re.findall(r'\w+', decl)[-1]
            ty = decl[:decl.rindex(nm)].strip()
            if attr in _ATTR:
                n = {'uint': 1, 'uint2': 2, 'uint3': 3}[ty]
                pro.append(f'const {ty} {nm} = {ty}({attr}{"" if _ATTR[attr] == 1 else ".x" if n == 1 else ".xy" if n == 2 else ""});')
                continue
            base = re.sub(r'\b(device|constant|const)\b', '', ty).replace('*', '').replace('&', '').strip()
            if ty.startswith('constant') and '&' in ty:
                self.inputs.append(nm); pro.append(f'device const {base} &{nm} = *(device const {base} *){nm}_;')
            elif ty.startswith('constant') or 'const' in ty.split('*')[0].split():
                self.inputs.append(nm); pro.append(f'device const {base} *{nm} = (device const {base} *){nm}_;')
            else:
                self.outputs.append(nm); pro.append(f'device {base} *{nm} = (device {base} *){nm}_;')
        header = ('#pragma METAL fp contract(off)\n' if not fast else '') + '#define ROUTE_MLX 1\n#define ROUTE_TBL device const\n' + defines + src.header
        self.template = list(template)
        self.k = mx.fast.metal_kernel(name=f'{name}{tag}', input_names=[f'{n}_' for n in self.inputs], output_names=[f'{n}_' for n in self.outputs],
                                      source='\n'.join(pro) + '\n' + body, header=header,
                                      compile_options={'math_mode': 'fast' if fast else 'safe'})

    def __call__(self, inputs, outputs, groups, threads, init=None):
        g = tuple(groups) + (1,) * (3 - len(groups))
        # MLX binds an array of fewer than 8 elements in constant memory; the kernels read every input from device memory
        ins = [inputs[n] if inputs[n].size >= 8 else mx.concatenate([inputs[n].reshape(-1), mx.zeros((8 - inputs[n].size,), inputs[n].dtype)]) for n in self.inputs]
        out = self.k(inputs=ins, output_shapes=[outputs[n][0] for n in self.outputs],
                     output_dtypes=[outputs[n][1] for n in self.outputs], grid=(g[0] * threads, g[1], g[2]), threadgroup=(threads, 1, 1),
                     template=self.template or None, init_value=init)
        return dict(zip(self.outputs, out))
