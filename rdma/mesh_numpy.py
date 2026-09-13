from dataclasses import dataclass


@dataclass(eq=False)
class Array:
    op: str
    operands: tuple = ()
    scalar: float = 0.0
    target: str = ''

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __add__(self, other):
        return Array('add', (self, other)) if isinstance(other, Array) else Array('shift', (self,), float(other))

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __mul__(self, other):
        return Array('multiply', (self, other)) if isinstance(other, Array) else Array('scale', (self,), float(other))

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __radd__(self, other):
        return self + other

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __rmul__(self, other):
        return self * other

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __matmul__(self, other):
        return einsum('ik,kj->ij', self, other)

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def into(self, target):
        self.target = target
        return self

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__' or kwargs:
            return NotImplemented
        operations = {'tanh': tanh, 'exp': exp, 'add': lambda a, b: a + b, 'multiply': lambda a, b: a * b}
        return operations[ufunc.__name__](*inputs) if ufunc.__name__ in operations else NotImplemented

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def __array_function__(self, function, types, args, kwargs):
        return einsum(*args, **kwargs) if function.__name__ == 'einsum' else NotImplemented


# design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
def input(name):
    return Array('input', (), target=name)


# design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
def tanh(x):
    return Array('tanh', (x,))


# design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
def exp(x):
    return Array('exp', (x,))


# design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
def einsum(indices, x, y):
    if indices != 'ik,kj->ij':
        raise ValueError('Supported contraction: ik,kj->ij')
    return Array('contract', (x, y))


# design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
def emit_c(expression, name, inputs):
    declarations = [f'const struct mesh_view *{arg}' for arg in inputs]
    lines = ['/* design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming */',
             f'static struct mesh_tensor *{name}(struct mesh_algebra *a, size_t count, ' + ', '.join(declarations) + ', struct mesh_view *p, struct mesh_view z) {',
             '  int error=0;']
    values = {}

    # design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming
    def visit(node):
        if node in values:
            return values[node]
        if node.op == 'input':
            values[node] = node.target
            return node.target
        alpha, beta = 1.0, 0.0
        operand = node.operands[0]
        if node.op == 'shift':
            beta = node.scalar
            if operand.op == 'scale' and not operand.target:
                alpha = operand.scalar
                operand = operand.operands[0]
        elif node.op == 'scale':
            alpha = node.scalar
        x = visit(operand)
        y = visit(node.operands[1]) if len(node.operands) > 1 else None
        index = len(values)
        output = node.target or f'v{index}'
        if node.op == 'contract':
            lines.append(f'  struct mesh_tensor *partial{index}=mesh_algebra_contract(a,{x},{y},count,z,1);')
            values[node] = f'partial{index}'
            return values[node]
        if not node.target:
            lines.extend([f'  struct mesh_view {output}[count];',
                          '  for(size_t q=0;q<count;q++) {',
                          f'    struct mesh_shape shape={{{x}[q].rows,{x}[q].columns,MESH_F32}};',
                          '    struct mesh_tensor *t=mesh_tensor_create(a,&shape,1,0);',
                          '    if(!t)return NULL;',
                          f'    {output}[q]=mesh_tensor_view(t,0);', '  }'])
        operations = {'scale': 'MESH_AFFINE', 'shift': 'MESH_AFFINE', 'add': 'MESH_ADD', 'multiply': 'MESH_MULTIPLY', 'tanh': 'MESH_TANH', 'exp': 'MESH_EXP'}
        if node.op == 'add':
            beta = 1.0
        second = f'{y}[q]' if y else '(struct mesh_view){0}'
        lines.extend(['  for(size_t q=0;q<count;q++) {',
                      f'    error=mesh_algebra_bind(a,{operations[node.op]},{x}[q],{second},{output}[q],{alpha:.9g},{beta:.9g});',
                      '    if(error){errno=error;return NULL;}', '  }'])
        values[node] = output
        return output

    if expression.op != 'contract':
        raise ValueError('The compiled result must be a partitioned contraction')
    result = visit(expression)
    lines.extend([f'  return {result};', '}'])
    return '\n'.join(lines) + '\n'

