# Views on shared memory: the sources' own words

Retrieved under a brief that assumed one contiguous offset per value. The operator rejected that on 2026-11-06: paging
is the constraint, and only a function typed `contiguous_input` or `contiguous_output` needs its operands as runs. The
file is kept for its quotes.

Retrieved 2026-11-06 with curl (no browser); a web search was used only to find URLs. Quotations are verbatim from the copy named under "Read at". Page numbers are those of that copy.

Conventions:
- A hyphen split across a line is resolved to the printed word (e.g. "non-increasing", "x-, y- and z-").
- "[…]" marks an omission.
- Subscripts are written with `_` (e.g. `first_op_t`).
- In the Apple documentation quotes, backticks mark a symbol the page renders as code or as a link.
- Code and header excerpts are spliced byte-for-byte from the files named.

Every prose quotation was checked mechanically against the text extracted from its copy (case, whitespace and punctuation ignored). Status is **read** or **not read**; nothing is quoted from a source marked not read.

---

## 1. The array view: van der Walt, Colbert, Varoquaux (2011)

**Status: read.**

- **Bibliographic:** Stéfan van der Walt, S. Chris Colbert, Gaël Varoquaux, "The NumPy array: a structure for efficient numerical computation", *Computing in Science & Engineering* 13(2), 2011, pp. 22–30, doi:10.1109/MCSE.2011.37. Preprint arXiv:1102.1523v1 [cs.MS], 8 Feb 2011.
- **Read at:** https://arxiv.org/pdf/1102.1523 (v1, 8 pages; the only version). Page 1 of that copy says: "This article is published in IEEE Computing in Science and Engineering. Please refer to the published version if accessible, as it contains editor’s improvements. (c) 2011 IEEE." The published IEEE version was not read.

p. 1 (opening):

> Underneath the hood, a NumPy array is really just a convenient way of describing one or more blocks of computer memory, so that the numbers represented may be easily manipulated.

p. 2, section "The structure of a NumPy array: a view on memory":

> A NumPy array (also called an “ndarray”, short for N-dimensional array) describes memory, using the following attributes:
>
> Data pointer the memory address of the first byte in the array.
>
> Data type description the kind of elements contained in the array, for example floating point numbers or integers.
>
> Shape the shape of the array, for example (10, 10) for a ten-by-ten array, or (5, 5, 5) for a block of data describing a mesh grid of x-, y- and z-coordinates.
>
> Strides the number of bytes to skip in memory to proceed to the next element. For a (10, 10) array of bytes, for example, the strides may be (10, 1), in other words: proceed one byte to get to the next column and ten bytes to locate the next row.
>
> Flags which define whether we are allowed to modify the array, whether memory layout is C- or Fortran-contiguous¹, and so forth.

> NumPy’s strided memory model deserves particular attention, as it provides a powerful way of viewing the same memory in different ways without copying data.

> The arrays x and y point to the same memory (i.e., if we modify the values in y we also modify those in x), but the strides for y have been changed so that only every second element is seen along either axis. y is said to be a view on x:

> Views need not be created using slicing only. By modifying strides, for example, an array can be transposed or reshaped at zero cost (no memory needs to be copied). Moreover, the strides, shape and dtype attributes of an array may be specified manually by the user (provided they are all compatible), enabling a plethora of ways in which to interpret the underlying data.

p. 3:

> In each of these cases, the resulting array points to the same memory. The difference lies in the way the data is interpreted, based on shape, strides and data-type. Since no data is copied in memory, these operations are extremely efficient.

p. 6, "Sharing data" and "The array interface for foreign blocks of memory":

> As shown above, performance is often improved by preventing repeated copying of data in memory. In this section, we show how NumPy may make use of foreign memory–in other words, memory that is not allocated or controlled by NumPy–without copying data.

> To facilitate such exchanges without copying the already allocated memory, NumPy defines an array interface that specifies how a given object exposes a block of memory. NumPy knows how to view any object with a valid __array_interface__ dictionary attribute as an array. Its most important values are data (address of the data in memory), shape and typestr (the kind of elements stored).

p. 8, "Conclusion":

> Its sophisticated memory description allows a wide variety of operations to be performed without copying any data in memory, bringing significant performance gains as data-sets grow large.

**Note on "offset":** the paper does not use the word "offset" (I searched the full extracted text). The base of its description is the "Data pointer": "the memory address of the first byte in the array".

---

## 2. DLPack: `dmlc/dlpack`, `include/dlpack/dlpack.h`

**Status: read.**

- **Bibliographic:** DLPack, repository https://github.com/dmlc/dlpack. I read `include/dlpack/dlpack.h` and `LICENSE` at commit `6eca9e643fcf7e870b7a1c99ee7fb166c5c2c459`, the tip of `main` on retrieval (committed 2026-11-04T15:49:35Z, "[CI] Upgrade docs dependencies for Python 3.14 runners (#195)"). The header declares `DLPACK_MAJOR_VERSION 1`, `DLPACK_MINOR_VERSION 4`.
- **Read at:** https://raw.githubusercontent.com/dmlc/dlpack/6eca9e643fcf7e870b7a1c99ee7fb166c5c2c459/include/dlpack/dlpack.h (sha256 `61d04d0898d4eda1b7fca6a63d8076892c24d04d2dd44aca16452b13acef0ef2`) and the `LICENSE` at the same commit.
- **License:** Apache License 2.0. `LICENSE` lines 1–2, 189 and 191:

```
                                 Apache License
                           Version 2.0, January 2004
[…]
   Copyright 2017 by Contributors
[…]
   Licensed under the Apache License, Version 2.0 (the "License");
```

  The header's own notice (lines 1–5):

```c
/*!
 *  Copyright (c) 2017 -  by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 */
```

`DLTensor` with its comments, `dlpack.h` lines 229–290:

```c
/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*!
   * \brief The data pointer points to the allocated data. This will be CUDA
   * device pointer, cl_mem handle in OpenCL, or id<MTLBuffer> for Metal. It
   * may be opaque on some device types. This pointer is always aligned to 256 bytes as in CUDA. The
   * `byte_offset` field should be used to point to the beginning of the data.
   *
   * Note that as of Nov 2021, multiple libraries (CuPy, PyTorch, TensorFlow,
   * TVM, perhaps others) do not adhere to this 256 byte alignment requirement
   * on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
   * (after which this note will be updated); at the moment it is recommended
   * to not rely on the data pointer being correctly aligned.
   *
   * For given DLTensor, the size of memory required to store the contents of
   * data is calculated as follows:
   *
   * \code{.c}
   * static inline size_t GetDataSize(const DLTensor* t) {
   *   size_t size = 1;
   *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
   *     size *= t->shape[i];
   *   }
   *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *   return size;
   * }
   * \endcode
   *
   * Note that if the tensor is of size zero, then the data pointer should be
   * set to `NULL`.
   */
  void* data;
  /*! \brief The device of the tensor */
  DLDevice device;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief The data type of the pointer*/
  DLDataType dtype;
  /*!
   * \brief The shape of the tensor
   *
   *  When ndim == 0, shape can be set to NULL.
   */
  int64_t* shape;
  /*!
   * \brief strides of the tensor (in number of elements, not bytes),
   *  can not be NULL if ndim != 0, must points to
   *  an array of ndim elements that specifies the strides,
   *  so consumer can always rely on strides[dim] being valid for 0 <= dim < ndim.
   *
   *  When ndim == 0, strides can be set to NULL.
   *
   *  \note Before DLPack v1.2, strides can be NULL to indicate contiguous data.
   *        This is not allowed in DLPack v1.2 and later. The rationale
   *        is to simplify the consumer handling.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DLTensor;
```

Device type for Metal, lines 88–89:

```c
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
```

What the managed wrapper is for, lines 292–297:

```c
/*!
 * \brief C Tensor object, manage memory of DLTensor. This data structure is
 *  intended to facilitate the borrowing of DLTensor by another framework. It is
 *  not meant to transfer the tensor. When the borrowing framework doesn't need
 *  the tensor, it should call the deleter to notify the host that the resource
 *  is no longer needed.
```

---

## 3. Apple: unified memory and `MTLStorageModeShared`

**Status: read** (documentation through its JSON endpoints, plus the local SDK headers).

Each page was read from its DocC JSON at `https://developer.apple.com/tutorials/data/documentation/<path>.json`; the HTML pages (URL in brackets) were not fetched. The JSON carries no revision date; retrieval date 2026-11-06.

### 3a. "Choosing a resource storage mode for Apple GPUs" (article)

Read at: https://developer.apple.com/tutorials/data/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus.json [https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus].

Abstract:

> Select an appropriate storage mode for your textures and buffers on Apple GPUs.

Overview:

> Apple GPUs have a unified memory model in which the CPU and the GPU share system memory. However, CPU and GPU access to that memory depends on the storage mode you choose for your resources. The `MTLStorageMode.shared` mode defines system memory that both the CPU and the GPU can access. The `MTLStorageMode.private` mode defines system memory that only the GPU can access.

> The `MTLStorageMode.memoryless` mode defines tile memory within the GPU that only the GPU can access. Tile memory has higher bandwidth, lower latency, and consumes less power than system memory.

"Choose a resource storage mode for buffers or textures" (term list):

> Populate and update on the CPU: Data shared by the CPU and GPU. Use `MTLStorageMode.shared`. The CPU and GPU share data. This is the default for buffer and texture storage.

> Populate on CPU and access frequently on GPU: Shared integrated memory for the CPU and GPU. Use `MTLStorageMode.shared`.

Note at the end of the article:

> You can create only textures, not buffers, using `MTLStorageMode.memoryless` mode. You can’t use buffers as memoryless render targets.

### 3b. `MTLStorageMode.shared` (enumeration case)

Read at: https://developer.apple.com/tutorials/data/documentation/metal/mtlstoragemode/shared.json [https://developer.apple.com/documentation/metal/mtlstoragemode/shared]. The page lists availability as macOS 10.11, iOS 9.0 and later.

Abstract:

> The CPU and GPU share access to the resource, allocated in system memory.

Discussion:

> This is the default storage mode for `MTLBuffer` instances on integrated GPUs and both `MTLBuffer` and `MTLTexture` instances on Apple silicon GPUs. On non-Apple family GPUs, the shared storage mode isn’t available for `MTLTexture` instances.

> When either the CPU or GPU changes the contents of the resource, you’re responsible for synchronizing access to the texture from the other participant. Ensure that all changes you schedule on either the CPU or GPU for a resource that uses shared memory complete before accessing that resource on the other processor.

The page for the resource option `MTLResourceOptions.storageModeShared` (https://developer.apple.com/tutorials/data/documentation/metal/mtlresourceoptions/storagemodeshared.json) has the same abstract and discussion, word for word.

### 3c. "Setting resource storage modes" (article)

Read at: https://developer.apple.com/tutorials/data/documentation/metal/setting-resource-storage-modes.json [https://developer.apple.com/documentation/metal/setting-resource-storage-modes].

> Storage modes are only set when creating an instance, and the system default allows for access to memory from both the CPU and GPU. Metal selects the default mode for resources depending on hardware.

> For Apple silicon GPUs the default is `MTLStorageMode.shared`.

### 3d. `MTLDevice.hasUnifiedMemory` (property)

Read at: https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/hasunifiedmemory.json [https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory].

> A Boolean value that indicates whether the GPU shares all of its memory with the CPU.

> A GPU with unified memory (`true`) is typically an integrated GPU. A GPU with dedicated memory (`false`) may take additional time to synchronize managed resources or copy data into private GPU resources.

### 3e. Local SDK headers

SDK: `xcrun --sdk macosx --show-sdk-path` gives `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`, which resolves to `MacOSX26.4.sdk` (`--show-sdk-version` 26.4.1, build 25E251).

`Metal.framework/Headers/MTLResource.h` (sha256 `6100bf17139327113e3a0e6782f1cd3f08493eee37432f92afb8c37bd39dca78`). Its notice:

```c
//
//  MTLResource.h
//  Metal
//
//  Copyright (c) 2014 Apple Inc. All rights reserved.
//
```

`MTLStorageMode`, lines 57–63 and 82–88:

```c
/*!
 @enum MTLStorageMode
 @abstract Describes location and CPU mapping of MTLTexture.
 @constant MTLStorageModeShared
 In this mode, CPU and device will nominally both use the same underlying memory when accessing the contents of the texture resource.
 However, coherency is only guaranteed at command buffer boundaries to minimize the required flushing of CPU and GPU caches.
 This is the default storage mode for iOS Textures.
[…]
typedef NS_ENUM(NSUInteger, MTLStorageMode)
{
    MTLStorageModeShared  = 0,
    MTLStorageModeManaged API_AVAILABLE(macos(10.11), macCatalyst(13.0)) API_UNAVAILABLE(ios) = 1,
    MTLStorageModePrivate = 2,
    MTLStorageModeMemoryless API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(10.0)) = 3,
} API_AVAILABLE(macos(10.11), ios(9.0));
```

`MTLResourceStorageModeShared` in the `MTLResourceOptions` documentation comment, lines 118–121:

```c
 @constant MTLResourceStorageModeShared
 In this mode, CPU and device will nominally both use the same underlying memory when accessing the contents of the resource.
 However, coherency is only guaranteed at command buffer boundaries to minimize the required flushing of CPU and GPU caches.
 This is the default storage mode for iOS Textures.
```

`Metal.framework/Headers/MTLDevice.h` (sha256 `d1ef1bb5bc85faeeec9bc85f49124a453a8cd33b1182b0a71a41ba0dc4ebbb89`), lines 508–514:

```c
/*!
 @property hasUnifiedMemory
 @abstract Returns YES if this GPU shares its memory with the rest of the machine (CPU, etc.)
 @discussion Some GPU architectures do not have dedicated local memory and instead only use the same memory shared with the rest
 of the machine.  This property will return YES for GPUs that fall into that category.
*/
@property (readonly) BOOL hasUnifiedMemory API_AVAILABLE(macos(10.15), ios(13.0));
```

---

## 4. Offset assignment in one arena: Pisarchyk and Lee (2020)

**Status: read.**

- **Bibliographic:** Yury Pisarchyk, Juhyun Lee (Google Research, Mountain View, CA), "Efficient Memory Management for Deep Neural Net Inference", Resource-Constrained Machine Learning (ReCoML) Workshop of MLSys 2020, Austin, TX, USA, 2020. arXiv:2001.03288 (v1 10 Jan 2020; v3 16 Feb 2020).
- **Read at:** https://arxiv.org/pdf/2001.03288 (v3, 7 pages).

p. 1, Abstract:

> We explore various strategies to smartly share memory buffers among intermediate tensors in deep neural nets.

p. 1, §1 Introduction:

> These tensors are materialized with memory buffers: A tensor of shape [B, H, W, C] translates to a memory buffer of size B × H × W × C × sizeof(float). To reduce the overhead of dynamic memory allocation, the memory buffers for intermediate tensors are typically allocated before running the model, but these can take up a significant amount of memory.

> Fortunately, the intermediate tensors do not have to co-exist in memory; thanks to the mostly sequential execution of the network due to data dependency, only one operator is active at any given point in time, and only its immediate input and output intermediate tensors are needed.

p. 2, §2 Related Work:

> TensorFlow Lite (TFLite) GPU employs a memory manager for its GPU buffers (Lee et al., 2019). Two approximations are explored for this NP-complete resource management problem (Sethi, 1975), which is similar to register allocation problem, but more complex due to different sizes of tensors. (Sekiyama et al., 2018) solve the memory allocation problem as a special case of 2D strip packing problem.

p. 2, §3 Definition of Terms (sizes and lifetimes as data):

> Tensor Usage Interval of an intermediate tensor t is defined as the pair {first_op_t, last_op_t}, where first_op_t and last_op_t are the indices of the first and the last operator that use t as its input or output, respectively. The indices are from a topological sort of the neural network which is also the operators’ execution order. For the remainder of the paper, we assume that this order is fixed. Note that no two tensors with intersecting usage intervals can share memory.

> Tensor Usage Record of an intermediate tensor t is defined as the triple {first_op_t, last_op_t, size_t}, where size_t is t’s aligned size in bytes.

> Operator Breadth of an operator is defined as the sum of all tensor sizes in its profile.

p. 2, §4 The Shared Objects Approach (the alternative):

> There are broadly two ways of sharing memory which are discussed in this and the following section. We call the first Shared Objects where each memory buffer (“shared object”) is assigned to an intermediate tensor at a given time. […] This approach is most suitable for GPU textures.

p. 4, §5 The Offset Calculation Approach:

> We call the other memory sharing approach Offset Calculation where a large chunk of memory is pre-allocated and the intermediate tensors are given parts of the memory by the offsets within the memory block. The main objective is to minimize the size of the allocated memory block. While the solution to this problem shows best performance in terms of total allocated memory, it is only applicable to CPU memory or GPU buffers, but not GPU textures which need to be accessed as a whole. The solution of Shared Objects problem can be converted to the solution of Offset Calculation problem by placing the shared objects contiguously in memory. The opposite is not true as memory footprints of tensors with non-intersecting usage intervals can still overlap.

> The Offsets Calculation problem can be seen as a special case of 2D strip packing problem (Sekiyama et al., 2018). A set of rectangular items with fixed coordinates by one axes into a container to minimize its size by other dimension. If the height of a container is treated as the allocation time axis, then we need to minimize the container’s width which corresponds to the memory footprint.

pp. 4–5, §5.1 Theoretical Lower Bound:

> During the execution of any operator of the neural network all tensors in its profile need to be present in memory. Their total size is equal to this operator’s breadth. It means, that any strategy will provide us with memory consumption greater or equal to any operator breadth, and the lower bound for Offset Calculation is equal to the maximum among all operator breadths.

p. 5, §5.2 Greedy by Size for Offset Calculation:

> We first iterate through tensor usage records in non-increasing order by their size (L.1,6). For each record, we check already assigned tensors whose usage intervals intersect with that of the current tensor (L.10–13) to find the smallest gap in memory between them such that current tensor fits into that gap (L.9,14–17). If such a gap is found, the current tensor is allocated to this gap. Otherwise, we allocate it after the rightmost tensor whose usage interval intersect with that of the current tensor (L.19–20). We assign the corresponding offset to current tensor and the tensor becomes assigned (L.21–23) as shown in Figure 6.

p. 5, §5.3 Greedy by Breadth for Offset Calculation:

> Greedy by Breadth can also be converted for Offsets Calculation in a similar fashion.

> While Greedy by Breadth does not perform well for Offset Calculation compared to Greedy by Size, it still outperforms the prior work on some networks, e.g. MobileNet v2.

p. 5, §6 Evaluation:

> For Offset Calculation, Greedy by Size performs best as shown in Table 2. It achieves the theoretical lower bound on all selected neural networks, except DeepLab, where it still falls within 8% of the lower bound.

p. 6, §7 Conclusion:

> In general, i.e. CPU inference or GPU inference with buffers, it is recommended to explore the two best strategies for the Offset Calculation problem, Greedy by Size and Strip Packing Bestfit. For the Shared Objects problem, e.g. GPU inference with textures, Greedy by Size Improved and Greedy by Breadth are recommended for pre-inference exploration.

> The strategies presented assume that the sizes of intermediate tensors are known in advance. This assumption may not be true for recurrent neural networks with long short-term memory units (Hochreiter & Schmidhuber, 1997) including intermediate tensors with dynamically changing sizes. For such cases, the algorithms need to be run multiple times saving information about allocation from all runs in one place. The first run will allocate only those tensors whose sizes are known at the beginning, and the second run will allocate those tensors whose sizes become known after calculation of the first dynamic tensor, etc.

p. 6, Acknowledgements:

> We would like to thank Andrei Kulik for the initial brainstorming and the TFLite team for adopting our strategies to TFLite’s memory manager, especially Terry Heo for the final implementation.

---

## 5. Offset assignment as a compile-time problem: Gergov (1999); Garey and Johnson (1979) [SR2]

### 5a. J. Gergov, "Algorithms for compile-time memory optimization", SODA 1999

**Status: not read.** No quotation.

- **Bibliographic, as recorded by others (not checked against the paper):** OpenAlex W2015752867: "Algorithms for compile-time memory optimization", Jordan Gergov, 1999, no DOI, no open-access location; its landing page is the dblp SODA '99 table of contents. Reference [16] of arXiv:2608.14471 (Silva, Soares, Pereira, "Memory Allocation for Constant-Bounded Programs", 2026): "Jordan Gergov. 1999. Algorithms for compile-time memory optimization. In SODA (Baltimore, Maryland, USA). Society for Industrial and Applied Mathematics, USA, 907–908."
- **Attempts:**
  - dl.acm.org search: HTTP 403, Cloudflare challenge.
  - dblp.org and dblp.uni-trier.de: "Making sure you're not a bot!" (Anubis) challenge.
  - pure.mpg.de (MPI record for Gergov): Anubis challenge, HTTP 403.
  - api.semanticscholar.org: HTTP 429 on all four tries.
  - Crossref: no record.
  - OpenAlex: no open-access copy.
  - Not used: a Scribd upload whose provenance couldn't be checked.

### 5b. M. R. Garey and D. S. Johnson, *Computers and Intractability: A Guide to the Theory of NP-Completeness*, W. H. Freeman, San Francisco, 1979, problem [SR2] "Dynamic Storage Allocation"

**Status: not read.** No quotation.

- **Bibliographic (Internet Archive metadata):** item `computersintract0000gare`, "Computers and intractability : a guide to the theory of NP-completeness", Garey, Michael R, San Francisco : W. H. Freeman, 1979, ISBN 0716710455. I could not confirm the problem number [SR2] or its page.
- **Attempts:** the Internet Archive item is `access-restricted-item: true` (print-disabled collection). Its public search-inside endpoint returned HTTP 403 "Item not available". I made no attempt on the restricted files.

### 5c. A substitute that is not one of the requested sources (read)

P. Crescenzi and V. Kann (eds.), *A compendium of NP optimization problems* (web edition), entry "MINIMUM DYNAMIC STORAGE ALLOCATION", signed "Viggo Kann 2000-03-20". **Read at:** https://www.csc.kth.se/~viggo/wwwcompendium/node163.html. The page's formulas are images; the LaTeX below is the images' `alt` text, verbatim.

> MINIMUM DYNAMIC STORAGE ALLOCATION
>
> INSTANCE: Set A of items to be stored, each $a\in A$ having a size $s(a)\in Z^+$, an arrival time $r(a) \in Z^{+}$, and a departure time $d(a) \in Z^{+}$.
>
> SOLUTION: A feasible allocation of storage for A, i.e., a function $\sigma: A \rightarrow Z^{+}$ such that, for all $a,a' \in A$, if $I(a) \cap I(a')$ is nonempty then either $d(a) \leq r(a')$ or $d(a') \leq r(a)$.
>
> MEASURE: The size of the allocation, i.e., $\max_{a \in A}(\sigma(a)+s(a)-1)$.
>
> Good News: Approximable within 5 [ 192 ].

The compendium's bibliography (https://www.csc.kth.se/~viggo/wwwcompendium/node275.html) gives [192] as: "Gergov, J. (1996), “Approximation algorithms for dynamic storage allocation”, Proc. 4th Ann. European Symp. on Algorithms, Lecture Notes in Comput. Sci. 1136, Springer-Verlag, 52-61." The entry uses $I(a)$ without defining it on that page. It cites neither Garey and Johnson nor a problem number.

---

## 6. Why paging is used at all: Kwon et al. (2023), PagedAttention

**Status: read.**

- **Bibliographic:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP ’23, October 23–26, 2023, Koblenz, Germany, doi:10.1145/3600006.3613165. arXiv:2309.06180v1 [cs.LG], 12 Sep 2023.
- **Read at:** https://arxiv.org/pdf/2309.06180 (v1, 16 pages).

p. 1, Abstract:

> High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems.

p. 1, Figure 1 caption, and §1:

> The parameters (gray) persist in GPU memory throughout serving. The memory for the KV cache (red) is (de)allocated per serving request. A small amount of memory (yellow) is used ephemerally for activation.

> Approximately 65% of the memory is allocated for the model weights, which remain static during serving. Close to 30% of the memory is used to store the dynamic states of the requests.

p. 2, §1 Introduction:

> In this paper, we observe that existing LLM serving systems [31, 60] fall short of managing the KV cache memory efficiently. This is mainly because they store the KV cache of a request in contiguous memory space, as most deep learning frameworks [33, 39] require tensors to be stored in contiguous memory. However, unlike the tensors in the traditional deep learning workloads, the KV cache has unique characteristics: it dynamically grows and shrinks over time as the model generates new tokens, and its lifetime and length are not known a priori.

> First, the existing systems [31, 60] suffer from internal and external memory fragmentation. To store the KV cache of a request in contiguous space, they pre-allocate a contiguous chunk of memory with the request’s maximum length (e.g., 2048 tokens). This can result in severe internal fragmentation, since the request’s actual length can be much shorter than its maximum length (e.g., Fig. 11). Moreover, even if the actual length is known a priori, the pre-allocation is still inefficient: As the entire chunk is reserved during the request’s lifetime, other shorter requests cannot utilize any part of the chunk that is currently unused.

> Second, the existing systems cannot exploit the opportunities for memory sharing. […] However, memory sharing is not possible in the existing systems because the KV cache of the sequences is stored in separate contiguous spaces.

> To address the above limitations, we propose PagedAttention, an attention algorithm inspired by the operating system’s (OS) solution to memory fragmentation and sharing: virtual memory with paging. PagedAttention divides the request’s KV cache into blocks, each of which can contain the attention keys and values of a fixed number of tokens. In PagedAttention, the blocks for the KV cache are not necessarily stored in contiguous space.

p. 4, §3 Memory Challenges in LLM Serving:

> Scheduling for unknown input & output lengths. The requests to an LLM service exhibit variability in their input and output lengths. This requires the memory management system to accommodate a wide range of prompt lengths. In addition, as the output length of a request grows at decoding, the memory required for its KV cache also expands and may exhaust available memory for incoming requests or ongoing generation for existing prompts.

p. 4, §3.1 Memory Management in Existing Systems:

> Since most operators in current deep learning frameworks [33, 39] require tensors to be stored in contiguous memory, previous LLM serving systems [31, 60] also store the KV cache of one request as a contiguous tensor across the different positions. Due to the unpredictable output lengths from the LLM, they statically allocate a chunk of memory for a request based on the request’s maximum possible sequence length, irrespective of the actual input or eventual output length of the request.

p. 5, §4.1 PagedAttention and §4.2 KV Cache Manager:

> To address the memory challenges in §3, we introduce PagedAttention, an attention algorithm inspired by the classic idea of paging [25] in operating systems. Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space.

> The key idea behind vLLM’s memory manager is analogous to the virtual memory [25] in operating systems. OS partitions memory into fixed-sized pages and maps user programs’ logical pages to physical pages. Contiguous logical pages can correspond to non-contiguous physical memory pages, allowing user programs to access memory as though it were contiguous. Moreover, physical memory space needs not to be fully reserved in advance, enabling the OS to dynamically allocate physical pages as needed.

p. 15, References. [25] is source 8 below:

> [25] Tom Kilburn, David BG Edwards, Michael J Lanigan, and Frank H Sumner. 1962. One-level storage system. IRE Transactions on Electronic Computers 2 (1962), 223–235.

---

## 7. Virtual contiguity: Prabhu et al. (2025), vAttention

**Status: read.**

- **Bibliographic:** Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, Ashish Panwar, "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention", in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS ’25)*, March 30–April 3, 2025, Rotterdam, doi:10.1145/3669940.3707256 (CC BY 4.0). arXiv:2405.04437 (v1 7 May 2024; v3 29 Jan 2025).
- **Read at:** https://arxiv.org/pdf/2405.04437 (v3, 18 pages).

p. 1, Abstract:

> PagedAttention is a popular approach for dynamic memory allocation in LLM serving systems. It enables on-demand allocation of GPU memory to mitigate KV cache fragmentation — a phenomenon that crippled the batch size (and consequently throughput) in prior systems. However, in trying to allocate physical memory at runtime, PagedAttention ends up changing the virtual memory layout of the KV cache from contiguous to non-contiguous. Such a design leads to non-trivial programming and performance overheads.

> We present vAttention — an approach that mitigates fragmentation in physical memory while retaining the virtual memory contiguity of the KV cache. We achieve this by decoupling the allocation of virtual and physical memory using CUDA virtual memory management APIs.

p. 2, §1 Introduction:

> Efficiently allocating GPU memory for the KV cache is challenging for two reasons. First, the per-request KV cache grows slowly (one token per iteration), and second, a request’s decode length (or its total KV cache size) is not known ahead of time.

> Inspired by demand paging in OS-based virtual memory systems, vLLM introduced PagedAttention [51] that allocates small blocks of GPU memory on demand i.e., when previously allocated blocks are fully utilized and the model continues to generate more tokens.

> However, we show that PagedAttention faces a fundamental consequence of dynamic memory allocation: dynamically allocated objects are not guaranteed to be contiguous. Note that user-level objects are allocated in virtual memory. Therefore, in trying to enable dynamic allocation of physical memory, PagedAttention ends up changing the virtual memory layout of KV cache from contiguous to non-contiguous.

> We present vAttention – a memory management approach that retains the virtual contiguity of KV cache while enabling dynamic allocation of physical memory.

p. 3, §2.2 Fragmentation and PagedAttention:

> Serving LLMs with high throughput requires careful allocation of GPU memory. This is challenging because the total context length of a request is not known in advance. Serving systems worked around this challenge by pre-reserving KV cache space assuming that each context is as long as the maximum length supported by the model (e.g., 200K for Yi-34B-200K). vLLM shows that this strategy is prone to severe internal fragmentation. In fact, vLLM showed that prior reservation is suboptimal even if the context lengths are known in advance. This is because the per-request KV cache grows one token at a time and hence prior reservation wastes memory over the entire lifetime of a request.

p. 3, §3.1 Requires Re-writing the Attention Kernel:

> Conventional implementations of the attention operator assume that the two input tensors K and V (Equation 2) are stored in contiguous memory. By departing from the conventional memory layout, PagedAttention requires an implementation of the attention operator to be modified so as to compute attention scores over non-contiguous KV cache blocks.

p. 5, §4 Insights into LLM Serving Systems (with its footnote 1):

> Observation-1: KV cache memory requirement is predictable on a per-iteration basis. Due to auto-regressive decoding, once a request enters the decode phase, its KV cache size increases uniformly by one token per iteration.

> ¹Note that this prediction is possible only at the granularity of individual iterations; the total KV cache requirement of a request remains unknown as it depends on the total number of output tokens in the request.

p. 5, §5 vAttention: Design and Implementation; §5.1 Design Overview; §5.1.1:

> Our primary observation is that physical memory fragmentation can be avoided without making KV cache non-contiguous in virtual memory. To realize this, vAttention decouples the allocation of virtual memory from physical memory by leveraging system support for demand paging (instead of implementing demand paging in user space, as in PagedAttention).

> vAttention employs distinct allocation policies for virtual and physical memory. Specifically, we allocate a large contiguous buffer for the KV cache in virtual memory ahead-of-time (similar to systems prior to PagedAttention) while deferring the allocation of physical memory to runtime (similar to PagedAttention). This design preserves virtual contiguity of KV cache without fragmenting physical memory.

> Since virtual memory is abundant, we pre-allocate it in size that is large enough to hold the KV cache of the maximum batch size (configurable) that needs to be supported. In doing so, we assume that each request’s context length is same as the maximum supported by the model.

---

## 8. Paging's origin: Kilburn, Edwards, Lanigan, Sumner (1962)

**Status: read in a reprint** (an HTML transcription of the 1971 book reprint). The IRE original was **not read**.

- **Bibliographic:** T. Kilburn, D. B. G. Edwards, M. J. Lanigan, F. H. Sumner, "One-Level Storage System", *IRE Transactions on Electronic Computers*, vol. EC-11, no. 2, pp. 223–235, April 1962, doi:10.1109/TEC.1962.5219356 (Crossref, which gives the container title as "IEEE Transactions on Electronic Computers"). Reprinted as Chapter 23 (pp. 276–290) of C. Gordon Bell and Allen Newell, *Computer Structures: Readings and Examples*, McGraw-Hill, 1971, ISBN 07-004357-4. The reprint's footnote 1 reads: "IRE Trans., EC-11, vol. 2, pp. 223-235, April, 1962."
- **Read at:** https://gordonbell.azurewebsites.net/computer_structures__readings_and_examples/00000296.htm through `00000310.htm`, Gordon Bell's site. That transcription has OCR slips: the byline reads "D. B. C. Edwards / M. I. Lanigan", and microseconds appear as "m sec". The passages below are free of such slips. Page numbers are the reprint's.

Reprint p. 276, Summary:

> After a brief survey of the basic Atlas machine, the paper describes an automatic system which in principle can be applied to any combination of two storage systems so that the combination can be regarded by the machine user as a single level.

Reprint p. 276, §1 Introduction:

> In practice on the first installation at Manchester University a total of 10⁵ words are provided, but though it is just technically feasible to make this in one level it is much more economical to provide a core store (16,000 words) and drum (96,000 words) combination.

(The transcription prints the exponent as "10 5"; the superscript is restored here.)

> Hence a system has been devised to make the core drum store combination appear to the programmer as a single level of storage, the requisite transfers of information taking place automatically. There are a number of additional benefits derived from the scheme adopted, which include relative addressing so that routines can operate anywhere in the store, and a "lock out" facility to prevent interference between different programs simultaneously held in the store.

Reprint p. 279, §3 One-level store concept:

> Each of these methods has its disadvantages, in the first case, that of expense, and in the second case, that of inconvenience to the user, who is obliged to program transfers of information between the two types of store and this can be time consuming.

> The two-level storage scheme has obvious economic advantages, and inconvenience to the machine user can be eliminated by making the transfer arrangements completely automatic. In Atlas a completely automatic system has been provided with techniques for minimizing the transfer times. In this way the core and drum are merged into an apparent single level of storage with good performance and at moderate cost.

> The central store is subdivided into blocks of 512 words as shown by the address arrangements in Fig. 2b. The main core store is also partitioned into blocks of this size which for identification purposes are called pages. Associated with each of these core store page positions is a "page address register" (P.A.R.) which contains the address of the block of information at present occupying that page position.

Reprint p. 280:

> The provision of the Page Address Registers, the equivalence circuitry, and the learning program have permitted the core store and drum to be regarded by the ordinary machine user as a one-level store, and the system has the additional feature of "floating address" operation, i.e., any block of information can be stored in any absolute position in either core or drum store.

---

## Comparison: what the sources themselves say

### Q1. For tensors whose sizes and lifetimes are known ahead: one contiguous offset per tensor in one arena, read through views?

The arena with one offset per tensor, given sizes and lifetimes known ahead (Pisarchyk and Lee, §§1, 3, 5, 7):

> a large chunk of memory is pre-allocated and the intermediate tensors are given parts of the memory by the offsets within the memory block

> Tensor Usage Record of an intermediate tensor t is defined as the triple {first_op_t, last_op_t, size_t}

> Note that no two tensors with intersecting usage intervals can share memory.

> the memory buffers for intermediate tensors are typically allocated before running the model

> The strategies presented assume that the sizes of intermediate tensors are known in advance.

The same problem in its classical form, from the substitute (5c). The requested Gergov and Garey–Johnson texts were not read.

> Set A of items to be stored, each $a\in A$ having a size $s(a)\in Z^+$, an arrival time $r(a) \in Z^{+}$, and a departure time $d(a) \in Z^{+}$.

One contiguous region per tensor, as the serving papers describe the frameworks and kernels (Kwon et al. §1; Prabhu et al. §3.1):

> most deep learning frameworks [33, 39] require tensors to be stored in contiguous memory

> Conventional implementations of the attention operator assume that the two input tensors K and V (Equation 2) are stored in contiguous memory.

Read through views: a base address plus shape and strides, with no copy (van der Walt et al. pp. 2–3; DLPack):

> Data pointer the memory address of the first byte in the array.

> In each of these cases, the resulting array points to the same memory. The difference lies in the way the data is interpreted, based on shape, strides and data-type. Since no data is copied in memory, these operations are extremely efficient.

```c
   * may be opaque on some device types. This pointer is always aligned to 256 bytes as in CUDA. The
   * `byte_offset` field should be used to point to the beginning of the data.
[…]
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
```

One memory, used by both CPU and GPU (Apple):

> Apple GPUs have a unified memory model in which the CPU and the GPU share system memory.

> A Boolean value that indicates whether the GPU shares all of its memory with the CPU.

The limits these same sources state:

> it is only applicable to CPU memory or GPU buffers, but not GPU textures which need to be accessed as a whole. (Pisarchyk and Lee, §5)

> This assumption may not be true for recurrent neural networks with long short-term memory units (Hochreiter & Schmidhuber, 1997) including intermediate tensors with dynamically changing sizes. For such cases, the algorithms need to be run multiple times saving information about allocation from all runs in one place. (Pisarchyk and Lee, §7)

> In this mode, CPU and device will nominally both use the same underlying memory when accessing the contents of the texture resource. However, coherency is only guaranteed at command buffer boundaries to minimize the required flushing of CPU and GPU caches. (`MTLResource.h`, `MTLStorageModeShared`)

> The `MTLStorageMode.memoryless` mode defines tile memory within the GPU that only the GPU can access. (Apple, "Choosing a resource storage mode for Apple GPUs")

What no single source states: Pisarchyk and Lee assign offsets in one memory block but never mention strides or views. NumPy and DLPack describe views (a base address with shape and strides; DLPack adds `byte_offset`), but neither describes assigning offsets in a shared block by tensor lifetime. The NumPy paper never uses the word "offset". DLPack's `data` comment notes that several libraries "always use `byte_offset=0`".

### Q2. Do the sources motivate paging only for memory whose size isn't known ahead?

Unknown or growing size, as stated:

> the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. (Kwon et al., Abstract)

> unlike the tensors in the traditional deep learning workloads, the KV cache has unique characteristics: it dynamically grows and shrinks over time as the model generates new tokens, and its lifetime and length are not known a priori. (Kwon et al., §1)

> Due to the unpredictable output lengths from the LLM, they statically allocate a chunk of memory for a request based on the request’s maximum possible sequence length (Kwon et al., §3.1)

> First, the per-request KV cache grows slowly (one token per iteration), and second, a request’s decode length (or its total KV cache size) is not known ahead of time. (Prabhu et al., §1)

The same two papers also argue for paging when the length is known, because the memory grows during its lifetime:

> Moreover, even if the actual length is known a priori, the pre-allocation is still inefficient: As the entire chunk is reserved during the request’s lifetime, other shorter requests cannot utilize any part of the chunk that is currently unused. (Kwon et al., §1)

> In fact, vLLM showed that prior reservation is suboptimal even if the context lengths are known in advance. This is because the per-request KV cache grows one token at a time and hence prior reservation wastes memory over the entire lifetime of a request. (Prabhu et al., §2.2)

Kwon et al. give a second motive, sharing between sequences:

> Second, the existing systems cannot exploit the opportunities for memory sharing. (Kwon et al., §1)

What the virtual memory is for in vAttention: keeping the growing KV cache contiguous while physical pages arrive at runtime.

> we allocate a large contiguous buffer for the KV cache in virtual memory ahead-of-time (similar to systems prior to PagedAttention) while deferring the allocation of physical memory to runtime (similar to PagedAttention). This design preserves virtual contiguity of KV cache without fragmenting physical memory. (Prabhu et al., §5.1)

The memory these papers do not page:

> Approximately 65% of the memory is allocated for the model weights, which remain static during serving. (Kwon et al., §1)

The origin of paging, in Kilburn et al.: two kinds of store (core and drum) made to look like one level, plus relocation and protection between programs.

> Hence a system has been devised to make the core drum store combination appear to the programmer as a single level of storage, the requisite transfers of information taking place automatically. (Kilburn et al., §1)

> which include relative addressing so that routines can operate anywhere in the store, and a "lock out" facility to prevent interference between different programs simultaneously held in the store. (Kilburn et al., §1)

The arena paper's answer to unknown sizes is to rerun the offset assignment, not to page:

> For such cases, the algorithms need to be run multiple times saving information about allocation from all runs in one place. (Pisarchyk and Lee, §7)
