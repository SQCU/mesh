# numerical-50: two admission guards removed (SAME64), operations.py only, source HOLD, unstaged
- **`_lstsq_shapes` (2050–2052):** removed `self.device.type != 'cuda' or selected == 'gels'`, PyTorch's CUDA-provider check (lstsq-pytorch.reference.cpp 3499–3503). Mesh lowers every placed lstsq with its own `_least_squares` (gels/gelsy) and `_least_squares_svd` (gelsd/gelss). The driver-name, dimension and dtype/device checks, `_lstsq_driver`'s default choice and the residual/fixed metadata are unchanged.
- **`_elementwise` (554–555):** removed the integral `not isinstance(alpha, (bool, int))` NotImplementedError.
  - Producer proof: nondefault alpha enters through `_arithmetic`'s captured `given.get('alpha', 1)` (628 complex, 639 real), or is forwarded unchanged by the complex `where` recursion (548). Other internal numerical callers use the default 1.
  - torch/_refs add 1239–1247 and sub 1892–1897 already validate alpha at source dispatch (sub only when alpha != 1, so a float 1.0 is source-accepted). `_apply` (462) uses alpha only for add/sub and drops it when alpha == 1. No arithmetic was widened or recast.
- **Comments:** the rejection-filter comment is at both sites.
- **Checks:** AST passes. pyflakes shows only the 2 warnings already at HEAD. Delta is +5/−4. No import, target, test, staging or backup.
