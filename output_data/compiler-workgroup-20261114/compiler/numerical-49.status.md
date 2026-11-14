# numerical-49: bool argmax/argmin admission removed, complex diagnostic corrected (SAME64), operations.py only, source HOLD
- **`_extreme` (operations.py 6205–6211):** removed the extra `NotImplementedError` ("PyTorch 2.14 defines no bool form") for bool argmax/argmin.
  - Path: `_typed(..., REAL)` already admits bool (`REAL = tuple(SCALAR)`, 104) and the int64 result. The existing body then loads bool as I32 with 0/1 lowest/highest and `_ordered`'s ties (1155–1165). torch/_meta_registrations.py 7777–7793 `argmax_argmin_meta` checks only the dimension and empty reduction, and returns int64.
  - The rejection-filter comment is at that site.
- **`_complex_value` (924–925):** the terminal error now reads "its structured complex lowering is unfinished (design/compilation-composition-change-request.md item 1)", replacing the unestablished "PyTorch defines no complex form".
- **Checks:** AST passes. pyflakes shows only the 2 warnings already at HEAD. Delta is +5/−4. No import, target, test, staging or backup, and no runtime result claimed.
