# Program specialization, and the host state the capture specializes on: sources, quoted verbatim

Retrieved 2026-11-07. Each block below was copied by script from the cited lines of the file named; nothing in a block is retyped.

## 1. Jones, Gomard and Sestoft (1993)

- **Source:** Neil D. Jones, Carsten K. Gomard and Peter Sestoft, *Partial Evaluation and Automatic Program Generation*, Prentice Hall International, 1993, ISBN 0-13-020249-5; the authors' A4 PDF, copyright transferred back to the authors in 1999 (title page, below).
- **Where read:** `https://www.itu.dk/people/sestoft/pebook/jonesgomardsestoft-a4.pdf` (404 today) through the Internet Archive, `https://web.archive.org/web/2021id_/https://www.itu.dk/people/sestoft/pebook/jonesgomardsestoft-a4.pdf`, saved as `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit6/pe/pebook.pdf`; text by `pdftotext -layout`, `pe/pebook.txt`. The text extraction drops the "fi" and "ff" ligatures ("De nition", "rst", "di erent"); the blocks keep the extraction as it is. Page numbers are the book's, from the running heads.

`pebook.txt` L25–27, title page:

```text
Originally published by Prentice Hall International 1993; ISBN 0-13-020249-5.
In 1999, the copyright has been transferred back to the authors.
Copyright c 1999 Neil D. Jones, Carsten K. Gomard, and Peter Sestoft;
```

`pebook.txt` L2828–2846, §4.2.2 "Residual programs and program specialization", p. 71:

```text
4.2.2 Residual programs and program specialization
De nition 4.1 Let p be an L-program taking as input a two-element list, and let
d1 2 D. Then an L-program r is a residual program for p with respect to d1 i

     [[p]]L [d1, d2] = [[r]]L d2
for all d2 2 D.                                                              2
  The equation expresses that running r on d2 yields the same result as running p
on both d1 and d2. Intuitively, the input d1 is already incorporated in the residual
program r. We now de ne a (correct) program specializer to be a program which
given p and d1 yields as result a residual program r, such that the above equation
holds.
De nition 4.2 An L-program mix is a program specializer i for every p, d1 2 D,
the program r = [[p]]L d1 is a residual L-program for p with respect to d1. Expressed
symbolically we get the mix equation:
      [[p]]L [d1, d2] = [[([[mix]]L [p, d1] )]]L d2
for all d2 2 D.                                                                    2
   The program p is called the subject program. Note that the right hand side of
the mix equation involves two program executions. First, a residual program r =
[[mix]]L [p, d1] is generated. Second, the residual program r is run on data d2.
```

`pebook.txt` L3123–3132, L3151–3164, §4.4.2 "Program point specialization", pp. 78–79:

```text
4.4.2 Program point specialization
The idea of program point specialization is to incorporate the values of the static
variables into the control point. Suppose that at program specialization time we
discover that if the program had been executed normally with all input data sup-
plied, the computation might eventually be in a state with control at point pp and
with vs as the values of the static variables. Then the pair (pp, vs) is made
a program point in the residual program. The code that (pp, vs) labels in the
residual program is an optimized version of the code at pp in the subject program.
The potential for optimization is because we know the values of the static variables.
As hinted by the examples, this means that a program point pp may appear in
De nition 4.3 A specialized program point is a tuple (pp, vs) where pp is a pro-
gram point from the subject program and vs is a set of values of the static
variables.                                                                        2
A specialized program point (pp, vs) represents a set of states of the subject
program's computation | all those with control point pp and static variable values
                                              Program specialization techniques 79
vs. In the residual program the value of vs is `built in' to the specialized program
point (pp, vs), and not explicit as it is at program specialization time.

De nition 4.4 The set of all specialized program points (pp, vs) that are reach-
able during program execution is called poly.                                 2

  Note that poly thus represents the set of points of control in the residual pro-
gram; in our example:
```

`pebook.txt` L3210–3215, "Computing poly", p. 80:

```text
Computing poly
Let pp0 be the rst label in the program and let vs0 be the initial values of the static
data. It is clear that (pp0 , vs0 ) should be in poly. Moreover, if any specialized
program point (pp, vs) is in poly, all specialized program points reachable from
(pp, vs) should be there. That is, poly is the closure of vs0 under the relation
`reachable from'.
```

`pebook.txt` L3229–3240, Figure 4.6, p. 80:

```text
(Figure 4.6).
  poly :=    f
            (pp0 , vs0 )     g;
  while poly     contains an unmarked (pp, vs) do
  begin
        mark
          (pp, vs)     ;
        generate code for the basic block starting at pp using the values in vs;
        poly := poly   [ successors(pp, vs)
  end

Figure 4.6: A simple specialization algorithm.
```

`pebook.txt` L3339–3346, the infinite poly and the finite division, p. 83:

```text
and so forth ad nauseam. The problem is that poly becomes in nite:
     poly = f(iterate, 0), (iterate, 1), (iterate, 2), . . . g

This happens because the value of X, though known, is unbounded since its val-
ues are computed under dynamic control. The problem did not arise when we
specialized the example programs of this chapter (an interpreter and mix itself).
The problem is handled by classifying the unbounded variable(s) as dynamic. A
division which ensures niteness of poly is itself called nite.
```

## 2. The static variable of the subject program: Dawn as e2b links it

- **Source:** metal-microbench `.build/Dawn`, commit `466163dc7bf94146ac9a921b86cc3ed08184ebb8 (2026-07-21)`. License: BSD 3-Clause, as design/capture-replay-sources-2026-11-07.md §5.

`src/dawn/native/Buffer.h` L268:

```cpp
    bool mIsDataInitialized = false;
```

`src/dawn/native/Buffer.cpp` L1072–1074, L1132–1134:

```cpp
bool BufferBase::NeedsInitialization() const {
    return !mIsDataInitialized && GetDevice()->IsToggleEnabled(Toggle::LazyClearResourceOnFirstUse);
}
void BufferBase::SetInitialized(bool initialized) {
    mIsDataInitialized = initialized;
}
```

`src/dawn/native/metal/BufferMTL.mm` L245–251, L261–264, L285–292, L302–304:

```objc
bool Buffer::EnsureDataInitialized(CommandRecordingContext* commandContext) {
    if (!NeedsInitialization()) {
        return false;
    }

    InitializeToZero(commandContext);
    return true;
    if (IsFullBufferRange(offset, size)) {
        SetInitialized(true);
        return false;
    }
void Buffer::InitializeToZero(CommandRecordingContext* commandContext) {
    DAWN_ASSERT(NeedsInitialization());

    ClearBuffer(commandContext, uint8_t(0u));

    SetInitialized(true);
    GetDevice()->IncrementLazyClearCountForTesting();
}
    [commandContext->EnsureBlit() fillBuffer:mMtlBuffer.Get()
                                       range:NSMakeRange(offset, size)
                                       value:clearValue];
```

`src/dawn/native/metal/CommandBufferMTL.mm` L1127–1129, every buffer of a dispatch scope:

```objc
        for (BufferBase* bufferBase : scope.buffers) {
            ToBackend(bufferBase)->EnsureDataInitialized(commandContext);
        }
```
