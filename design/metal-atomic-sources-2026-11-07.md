# Metal's atomic header: source lines, quoted verbatim

Read 2026-11-07 from the Metal toolchain on this machine:
`/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.R8x4KP/Metal.xctoolchain/usr/metal/32023/lib/clang/32023.883/include/metal/metal_atomic`
(GPUCompiler 32023.883).

## The 64-bit atomic type (line 104)

```
typedef _atomic<ulong> atomic_ulong;
```

## The types an atomic store takes (lines 163–171)

```
struct _valid_store_type : false_type
{
};

template <typename T>
struct _valid_store_type<T, typename enable_if<_disjunction<is_same<T, device int *>,
                                                            is_same<T, threadgroup int *>,
                                                            is_same<T, device uint *>,
                                                            is_same<T, threadgroup uint *>
```

Lines 172–176 add `device float *` where `__HAVE_ATOMIC_FLOAT__` is defined, and a bool type where
`__HAVE_ATOMIC_BOOL__` is.

## The types an atomic max takes (lines 636–641), and its declaration (lines 646–656)

```
struct _valid_max_type : false_type
{
};

template <typename T>
struct _valid_max_type<T, typename enable_if<_disjunction<is_same<T, device ulong *>>::value>::type> : true_type
```

```
template <typename T, typename U, typename _E = typename enable_if<_valid_max_type<device T *>::value && is_convertible<T, U>::value>::type>
METAL_FUNC void atomic_max_explicit(
```
