# Mealy (1955): sources, quoted verbatim

Retrieved 2026-11-07. Each block below was copied by script from the cited lines of the file named; nothing in a block is retyped.

- **Source:** George H. Mealy, "A Method for Synthesizing Sequential Circuits", *The Bell System Technical Journal* 34(5):1045–1079, September 1955.
- **Where read:** the Internet Archive's BSTJ item `bstj34-5-1045` ("BSTJ 34: 5. September 1955: A Method for Synthesizing Sequential Circuits. (Mealy, George H.)"), its OCR text `https://archive.org/download/bstj34-5-1045/bstj34-5-1045_djvu.txt`, saved as `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit7/mealy/mealy.txt`. The text is OCR of the printed pages and keeps the OCR's misreadings (for example "staies" for "states", "infernal" for "internal", "caitse" for "cause", "outpid" for "output"); subscripts are lost. Page numbers are the journal's, from the running heads in the text.

`mealy.txt` L1–12, title and abstract, p. 1045:

```text
A Method for 
Synthesizing Sequential Circuits 

By GEORGE H. MEALY 

(Mamiscript received May 6, 1955) 

The theoretical basis of sequential circuit synthesis is developed, with 
■particular reference to ike work of D. A. Huffman and E. F. Moore. A new 
method of synthesis is developed which emphasizes formal procedures rather 
than the more familiar intuitive ones. Famdliarity is assumed with the use 
of switching algebra in the synthesis of combinational circuits. 
```

`mealy.txt` L356–358, L464–467, §2.1 "The Model": the definition, pp. 1049–1050 (a table and the page break fall inside the sentence):

```text
We begin by giving an abstract definition of a switching circuit: 

A switching circuit is a circuit with a finite number of inputs, outputs. 
and (internal) staies. Its present output combination and next state are 
determined uniquely by the present input combination and the present state. 
If the circuit has one infernal state, we call it a combinational circuit; 
otherwise, we call it a sequential circuit. 
```

`mealy.txt` L496–503, §2.1: the general equations, p. 1050:

```text
Now, according to the definition given above, to each circuit we must 
be able to assign some set of equations relating the 9, and y; to the Xi 
and Qi . These equations will have the general form: 

Q^ = Q{Q< , Xi) 

Yt = Y{Q, , Xj) 

```

`mealy.txt` L526–532, §2.1: the delay lines hold the state, p. 1051:

```text
have assumed (SI), {S2), and (D) and know that, physically, the delay 
unit provides storage. We assign the qi , the excitation variables, to the 
inputs of delay lines, and we assign the Qj , the state variables, to delay 
line outputs. The present state of the circuit is the combination of con- 
ditions on the delay line outputs. The next state is the combination of 
conditions on the delay line inputs, since one time unit later this com- 
bination will be present on the outputs. 
```

`mealy.txt` L797–800, §2.2: a transition is labeled with its input and its output, p. 1052:

```text
associated circuit, and both the arrows and the changes of state are 
called transitions. A transition begins at a present state and ends at the 
next state. The transition is labeled X/ Y. X is an input combination and 
Y is the corresponding output combination. 
```

`mealy.txt` L939–941, L951–954, §3.1 "Moore's Theory": equivalence, p. 1054:

```text
Roughly speaking, we call two circuits equivalent if we cannot tell 
them apart by performing experiments involving only their inputs and 
outputs. Once we have solved the first problem of synthesis by finding 
(1) Two states, Q,- in circuit S and Qj in circuit T, are called equivalent 
if, given S initially in state Qi and T initially in state Qj , there is no 
sequence of input combinations which, when presented to both S and T, 
will caitse S and T to produce different sequences of outpid combinations. 
```

`mealy.txt` L978–981, §3.1: Theorem 1 (Moore), p. 1054:

```text
Theorem 1 (Moore). Corresponding to each circuit, S, is a circuit T 
which has the properties: (1) T is equivalent to S, (2) T has a minimal 
number of states, (S) no two states in T are equivalent, and (4) T is unique, 
except for circuits that residt from T by relabeling its states. T is called the 
```
