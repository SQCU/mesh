import mesh_numpy as np

x, remote, w = (np.input(name) for name in ('x', 'remote', 'w'))
u = (0.5 * x + 0.125).into('p')
a = np.tanh(u + remote)
y = np.einsum('ik,kj->ij', a, w)
print(np.emit_c(y, 'symbolic_contract', ('x', 'remote', 'w')), end='')
