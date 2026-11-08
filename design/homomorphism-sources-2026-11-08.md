# The homomorphisms of the asynchronous consumer: sources, quoted verbatim

Retrieved 2026-11-08, for design/torch-source-lowering-change-request.md item 1 (the asynchronous, incomplete,
noncontiguous consumer of design/plan-to-bridge-change-request.md R4). Each excerpt below was copied by script from
the cited lines of the text file named; nothing in a block is retyped. This file quotes the sources and adds no
argument or recommendation.

- Each PDF was fetched with curl into its own new directory under the session's temporary folder `scratchpad/i1src/`,
  which is not kept in this repository, and its SHA-256 is given below.
- Text was extracted with `pdftotext -layout` (poppler, pdftotext version 26.09.0). The excerpts are lines of that
  extraction, `<name>.layout.txt` beside each PDF, in fenced blocks so that the layout of formulas is kept. A
  formula's super- and subscripts are where the extraction puts them. Trailing spaces and form feeds are removed.
- The PDF page of each excerpt is counted from the extraction's form feeds.
- The text layers of the JFP and CWI PDFs are of scanned pages, and their symbols are as those layers give them: in
  Gibbons, "-H-" and "-W-" for ++, "O", "G", "Q" or "©" for the operator ⊙, and "horn" for hom; in Meertens, "+" for the
  construction operation and "X" or "x" for ×.

## 1. M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax"

- **Bibliography.** M. Milakov, N. Gimelshein. "Online normalizer calculation for softmax." arXiv:1805.02867v2, 2018.
- **Source:** https://arxiv.org/pdf/1805.02867v2, SHA-256 722747b3a32eb90f5f0255d6b59f8bb0e1bc2257702ab7915e684c15275d44d7 → `scratchpad/i1src/milakov/1805.02867v2.pdf`; text `scratchpad/i1src/milakov/1805.02867v2.layout.txt`.

### Algorithm 3

Lines 122–132, PDF page 3.

```
Algorithm 3 Safe softmax with online normalizer calculation
 1: m0 ← −∞
 2: d0 ← 0
 3: for j ← 1, V do
 4:     mj ← max (mj−1 , xj )
 5:     dj ← dj−1 × emj−1 −mj + exj −mj
 6: end for
 7: for i ← 1, V do
               xi −mV
 8:     yi ← e d V
 9: end for
```

### §3.1, Parallel online normalizer calculation

Lines 201–220, PDF page 4.

```
3.1 Parallel online normalizer calculation
The lines 1-6 of the algorithm 3 define a sequential way of calculating the normalization term in
a single pass over input vector. Modern computing devices allow running multiple threads concur-
rently; We need to have a parallel version of the algorithm to fully utilize devices. We define a
generalized version of the online normalizer calculation:
                                                          
                                    mV         x1    x2         x
                                             =     ⊕    ⊕ ... ⊕ V                                  (3)
                                    dV         1     1           1

where xi , mV , dV ∈ R. The binary operation ⊕ : R2 × R2 → R2 is defined as:
                                                                       
                  mi   mj                  max (mi , mj )
                     ⊕    =                                                                        (4)
                  di   dj   di × emi −max(mi ,mj ) + dj × emj −max(mi ,mj )

Applying (3) sequentially from left to right is equivalent to running lines 1-6 of the algorithm 3. The
operation ⊕ is associative, which enables parallel evaluation of (3). It is also commutative, which
provides the flexibility needed to make parallel implementations more efficient. We omit the proofs
for these two statements for brevity.
```

## 2. T. Dao et al., "FlashAttention" (NeurIPS 2022)

- **Bibliography.** T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 16344–16359.
- **Source:** https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf, SHA-256 315f252a8c791dd051ae84b531e4549465279c2a7264606a9cfe1f3495d9dfc8 → `scratchpad/i1src/flash/neurips2022.pdf`; text `scratchpad/i1src/flash/neurips2022.layout.txt`.

### §3.1, Tiling

Lines 235–256, PDF page 4.

```
Tiling. We compute attention by blocks. Softmax couples columns of K, so we decompose the large
softmax with scaling [53, 62, 69]. For numerical stability, the softmax of vector 𝑥 ∈ R 𝐵 is computed:
                                                                           ∑︁                          𝑓 (𝑥)
  𝑚(𝑥) := max 𝑥𝑖 , 𝑓 (𝑥) := 𝑒 𝑥1 −𝑚( 𝑥) ... 𝑒 𝑥𝐵 −𝑚( 𝑥) , ℓ(𝑥) :=               𝑓 (𝑥)𝑖 , softmax(𝑥) :=         .
             𝑖
                                                                              𝑖
                                                                                                         ℓ(𝑥)
For vectors 𝑥 (1) ,𝑥 (2) ∈ R 𝐵 , we can decompose the softmax of the concatenated 𝑥 = 𝑥 (1) 𝑥 (2) ∈ R2𝐵 as:
                                                                                                      

𝑚(𝑥) = 𝑚( 𝑥 (1) 𝑥 (2) ) = max(𝑚(𝑥 (1) ),𝑚(𝑥 (2) )), 𝑓 (𝑥) = 𝑒 𝑚( 𝑥 )−𝑚( 𝑥) 𝑓 (𝑥 (1) ) 𝑒 𝑚( 𝑥 )−𝑚( 𝑥) 𝑓 (𝑥 (2) ) ,
                                                                      (1)                       (2)           

                                 (1)                      (2)                                 𝑓 (𝑥)
ℓ(𝑥) = ℓ( 𝑥 (1) 𝑥 (2) ) = 𝑒 𝑚( 𝑥 )−𝑚( 𝑥) ℓ(𝑥 (1) ) +𝑒 𝑚( 𝑥 )−𝑚( 𝑥) ℓ(𝑥 (2) ), softmax(𝑥) =
                     
                                                                                                    .
                                                                                              ℓ(𝑥)
Therefore if we keep track of some extra statistics (𝑚(𝑥),ℓ(𝑥)), we can compute softmax one block
at a time.2 We thus split the inputs Q,K,V into blocks (Algorithm 1 line 1), compute the softmax
values along with extra statistics (Algorithm 1 line 1), and combine the results (Algorithm 1 line 1).
Recomputation. One of our goals is to not store 𝑂 (𝑁 2 ) intermediate values for the backward pass.
```

### Algorithm 1

Lines 275–300, PDF page 5.

```
Algorithm 1 F LASH ATTENTION
                                      𝑁 ×𝑑 in HBM, on-chip SRAM of size 𝑀.
Require: Matrices Q,K,V      𝑀∈R              𝑀 
 1: Set block sizes 𝐵 𝑐 = 4𝑑 ,𝐵𝑟 = min 4𝑑           ,𝑑 .
 2: Initialize O = (0) 𝑁 ×𝑑   ∈   R 𝑁 ×𝑑 ,ℓ = (0) ∈ R 𝑁 ,𝑚 = (−∞) ∈ R 𝑁 in HBM.
                           l m                   𝑁                 𝑁                                            l m
 3: Divide Q into 𝑇𝑟 = 𝐵𝑁 blocks Q1 ,...,Q𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, and divide K,V in to 𝑇𝑐 = 𝐵𝑁
                                𝑟                                                                                 𝑐
    blocks K1 ,...,K𝑇𝑐 and V1 ,...,V𝑇𝑐 , of size 𝐵𝑐 ×𝑑 each.
 4: Divide O into 𝑇𝑟 blocks O𝑖 ,...,O𝑇𝑟 of size 𝐵𝑟 ×𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖 ,...,ℓ𝑇𝑟 of size
    𝐵𝑟 each, divide 𝑚 into 𝑇𝑟 blocks 𝑚 1 ,...,𝑚𝑇𝑟 of size 𝐵𝑟 each.
 5: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do
 6:    Load K 𝑗 ,V 𝑗 from HBM to on-chip SRAM.
 7:    for 1 ≤ 𝑖 ≤ 𝑇𝑟 do
 8:       Load Q𝑖 ,O𝑖 ,ℓ𝑖 ,𝑚 𝑖 from HBM to on-chip SRAM.
 9:       On chip, compute S𝑖 𝑗 = Q𝑖 K𝑇𝑗 ∈ R 𝐵𝑟 ×𝐵𝑐 .
10:       On chip, compute 𝑚˜ 𝑖 𝑗 = rowmax(S𝑖 𝑗 ) ∈ R 𝐵𝑟 , P̃𝑖 𝑗 = exp(S𝑖 𝑗 − 𝑚˜ 𝑖 𝑗 ) ∈ R 𝐵𝑟 ×𝐵𝑐 (pointwise),
          ℓ˜𝑖 𝑗 = rowsum( P̃𝑖 𝑗 ) ∈ R 𝐵𝑟 .
                                                                              new              new
11:       On chip, compute 𝑚 𝑖new = max(𝑚 𝑖 ,𝑚˜ 𝑖 𝑗 ) ∈ R 𝐵𝑟 , ℓ𝑖new = 𝑒 𝑚𝑖 −𝑚𝑖 ℓ𝑖 +𝑒 𝑚˜ 𝑖 𝑗 −𝑚𝑖 ℓ˜𝑖 𝑗 ∈ R 𝐵𝑟 .
                                                          new            new
12:       Write O𝑖 ← diag(ℓ𝑖new ) −1 (diag(ℓ𝑖 )𝑒 𝑚𝑖 −𝑚𝑖 O𝑖 +𝑒 𝑚˜ 𝑖 𝑗 −𝑚𝑖 P̃𝑖 𝑗 V 𝑗 ) to HBM.
13:       Write ℓ𝑖 ← ℓ𝑖new , 𝑚 𝑖 ← 𝑚 𝑖new to HBM.
14:    end for
15: end for
16: Return O.
```

## 3. J. Gibbons, "The third homomorphism theorem" (J. Functional Programming, 1996)

- **Bibliography.** J. Gibbons. "Functional pearls: The third homomorphism theorem." Journal of Functional Programming 6(4), 657–665, 1996. DOI 10.1017/S0956796800001908, open access.
- **Source:** https://www.cambridge.org/core/services/aop-cambridge-core/content/view/8DB1295DC8EB3C4942FB635C439B016E/S0956796800001908a.pdf/functional-pearls-the-third-homomorphism-theorem.pdf, SHA-256 9695414297e1b207cca71849f7a110804332159c2d05956caf45ce282a1cb072 → `scratchpad/i1src/jfp/thirdht-jfp.pdf`; text `scratchpad/i1src/jfp/thirdht-jfp.layout.txt`.

### §2, Notation: homomorphisms

Lines 92–110, PDF pages 2–3.

```
           Homomorphisms: For a binary operator O, the list function h is O-homomorphic iff,
                 for all lists x and y,
                                                         h(x-H-y)           = h x Q hy

https://doi.org/10.1017/S0956796800001908 Published online by Cambridge University Press
                                                            Functional pearls                                   659

                    For example, the functions length and sum are both +-homomorphic, since
                                                  sum (x -W-y) =               sum x + sum y
                                               length (x -H- y)         =      length x + length y
                    Note that O is necessarily associative on the range of h, because -H- is
                    associative. Moreover, h [ ] is necessarily the unit of G on the range of h (if it
                    exists), because [] is the unit of -H-. If G has no unit, then h [] is not defined.
                    For example, head is -C-homomorphic where a <C b = a, but because <C has
                    no unit, head [] is undefined.
                    For associative operator G with unit e, we write 'horn ( O ) / e' for the (unique)
                    G-homomorphic function h for which h ° [•] = / . For example,
                                                      sum       =    horn (+) id 0
                                                   length       =    length = horn (+) one 0
```

### §3, Definition 3.1 to Theorem 3.3

Lines 149–159, PDF page 4.

```
            For the sake of completeness, we state without proof the First and Second Homo-
            morphism Theorems.
            Definition 3.1
            A function of the form horn (O) id e for some O is called a reduction.

            Definition 3.2
            For given/, the function horn (-H-) ([] °f) [] is written 'mapf                      and called a map.
            Theorem 3.3 (First Homomorphism Theorem)
            Every homomorphism can be written as the composition of a reduction and a map:
                                            horn ( O ) / e       =     horn (O) id e ° map f
            Conversely, every such composition is a homomorphism.
```

## 4. L. Meertens, "Algorithmics" (CWI Monographs 1, 1986)

- **Bibliography.** L. Meertens. "Algorithmics: towards programming as a mathematical activity." In J. W. de Bakker, M. Hazewinkel and J. K. Lenstra (eds.), Mathematics and Computer Science, CWI Monographs 1, North-Holland, 1986, pp. 289–334.
- **Source:** https://ir.cwi.nl/pub/2686/2686D.pdf, SHA-256 370df4a7a4882a9ed1e1501ecb270c10243c7fdb2b3b062a411653088741b668 → `scratchpad/i1src/meertens/2686D.pdf`; text `scratchpad/i1src/meertens/2686D.layout.txt`.

### The laws of associativity, commutativity and idempotency

Lines 735–746, PDF page 16.

```
   We can go further and consider structures on which other algebraic laws are
imposed. Of particular interest are the laws of associativity: s +(t +u) =
(s +t)+u; of commutativity: s +t = t +s; and finally of idempotency: s +s =
s. The interesting thing now is that the structures obtained correspond to fam-
iliar data structures: we get, successively, sequences, bags,1 and sets. For sets,           A



is the function Ax: {x} and + is the set union U. The identity law gives us
the empty sequence, bag or set. This relationship between familiar algebraic
laws and familiar data structures has been pointed out by BOOM [ 5].
Sequences correspond to what are known in algebra as monoids (or semi-
```

### Inserted-in, and the homomorphism lemma

Lines 810–837, PDF page 17.

```
          Inserted-in. Let X be an operation in D XD--'> D. Then X/
          stands for the function in SD--"' D satisfying
          (i) if x has an identity e (so that e xx = x x e = x), then
                  X/0 = e,
          (ii)       x
                  X/ = x , and
          (iii)   X/ s + t = X/ s; X X/ t .
So if X stands for the conventional multiplication operation, Ilx Es x is a
more familiar notation for X/s. However, inserting an operator x in a struc-
ture s is only meaningful if x has at least the same algebraic richness as the
operation + used to construct the structure. This means that if x is multipli-
cation, then the notation Xjs is not allowed if s is a set, for (in gen~ral)
x x x =f x. Otherwise, we would obtain contradictions like 2 = x/ 2 =
                                       =
x/2+2 = x/2; x x/2 = 2X2 4. (Alternatively, we could define the
insertion as an indeterminate expression, depending on the choice of represen·
tatives from the congruence classes induced by the laws of + .)
   The classes of functions f• and x/ are special cases of the homomorphisms
definable on Sv. By combining them in the form X/f•, all such homomor·
phisms can be expressed. 1his can be stated in the form of another lemma:
HOMOMORPHISM LEMMA. Let the function gE SD --'>-D' be a homomorphism, i.e.,
let there exist a function fE D--'>-D' and an operation X E D'XD' --'>-D' with
identity X/ 0, satisfying, for all x E D and s and t E SD:
(i) gO = X/0,
        x
(ii) g = fx,
(iii) gs +t = gs; Xgt.
Then g = X/f•.
```
