# BF16 matrix exponential and empty Hermitian results

Source ee6f0294, subsequent diagnostic-branch cleanup ca95f84a; matched native v38 build from executor. Both observations use the public compile_module → Program path on requested MPS, without a diagnostic hook or target fallback. Exact source/library hashes are in the two source JSON files and remained unchanged through each process.

- BF16 matrix_exp: the existing matrix-exp-repeat caller's two 2×2 inputs, now BF16. Six native rows restored without solving, finite BF16[2,2,2] result, relative errors [0.0010987932599577557, 0.0016953163667203488] against double. Source-formula squaring counts are [0,2], not a measured native iteration count. Exit0, empty stderr.
- Empty EIGH: zero matrix extent [0,0] and empty batch [0,2,2] in one public call. One native row restored without solving; returned shapes [0], [0,0], [0,2], [0,2,2], all F32 and zero elements, matching Torch. Exit0, empty stderr.

These observations cover the changed opmath/result-cast and static empty-result handoffs. They do not establish ANE precision, all numerical domains, or completion of scalar conditional execution. The existing matrix-exp NaN/error/recovery evidence is separate; no such BF16 error observation was run here.
