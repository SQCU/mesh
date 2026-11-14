# Requested-ANE failure: exact storage 14 source

One authorized compile-only observation of the same public 2×2 real Hermitian input and requested ANE placement.
Stop: `Target.constructed` exception event, after `metal_ir_construct` returns EINVAL and before ownership unwinds.
No Program construction, preparation, invocation, numerical mutation or production source edit.

Actual native owner: prefix call 0, 69 Values and 28 operations at the failure.
- Root14: TEMPORARY, scalar9/I64, 8 bytes, shape[1], identity2; no data, buffer, zeroed flag or transport.
- Alias15: same root/type, scalar logical shape[], offset0.
- Actual retained Builder/FX emission: `scalar_tensor_1 = aten.scalar_tensor.default(2)` creates root14;
  operation2, identity2, ANE/place2, not erased.
- Actual consumer: `where_1 = aten.where.self(bitwise_and, scalar_tensor_1, scalar_tensor)`, source result
  int64[3,2]; emitted operation9 is erased by construction. Root14 is input ordinal1.

The failing root is the `n=2` branch scalar of `_qdwh_eigh_state`'s `torch.where(entry,n,0)` before its later
`.to(torch.int32)`, not the arange tensor. The native MIL scalar mapping rejects I64 at Classes. No generic
cast/legalization/fallback is justified by this observation. Numerical owner receives the exact source fact
for comparison with the retained int32 agenda formulation.

Collection: existing lower_metal.construct return retains its original Builder roots and emitted leaf/FX
correspondence; existing public ValueInfo/shape/layout/OperationInfo getters are read inside the live exception
hook. It never calls Builder.value again or examines native handles after cleanup. Exact copied metadata is
ane-root14.json. Process exit0, expected construction error reproduced, stderr empty. No new public ABI.
