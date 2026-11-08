# Amdahl, Little, the α–β cost model, and HiGHS: sources, quoted verbatim

Retrieved 2026-11-07. Each block below was copied by script from the cited lines of the file named; nothing in a block is retyped. PDFs were fetched with `curl` into `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit11/`, text by `pdftotext` in reading order; two-column pages interleave the next column's words into a line, which the blocks keep as extracted.

## 1. Amdahl (1967)

- **Source:** G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", AFIPS Spring Joint Computer Conference, 1967, pp. 483–485. DOI 10.1145/1465482.1465560. Read through the Internet Archive snapshot of `https://dl.acm.org/doi/pdf/10.1145/1465482.1465560` (`web/2020id_`), saved as `amdahl.pdf`.

`amdahl.txt` L1–6, title:

```text
Validity of the single processor
approach to achieving large scale
computing capabilities
by DR. GENE M. AMDAHL
International Business Machines Corporation
Sunnyvale, California
```

`amdahl.txt` L9–14, L44–50:

```text
of the problem could exploit at most a processor of
performance three to four times the performance of
the housekeeping processor. A fairly obvious conclusion which can be drawn at this point is that the
effort expended on achieving high parallel processing
rates is wasted unless it is accompanied by achievements in sequential processing rates of very nearly
the same magnitude.
purpose environment this might be reduced by a
factor of two, but it is highly improbably that it could
be reduced by a factor of three. The nature of this
overhead appears to be sequential so that it is unlikely
to be amenable to parallel processing techniques.
Overhead alone would then place an upper limit on
throughput of five to seven times the sequential pro-
```

## 2. Little (2011): L = λW

- **Source:** J. D. C. Little, "Little's Law as Viewed on Its 50th Anniversary", Operations Research 59(3):536–549, 2011, DOI 10.1287/opre.1110.0940, which restates J. D. C. Little, "A proof for the queuing formula: L = λW", Operations Research 9(3):383–387, 1961. Read from `https://people.cs.umass.edu/~emery/classes/cmpsci691st/readings/OS/Littles-Law-50-Years-Later.pdf`, saved as `little.pdf`. The text extraction renders λ as "ã".

`little.txt` L1–5, header:

```text
OPERATIONS RESEARCH
Vol. 59, No. 3, May–June 2011, pp. 536–549
issn 0030-364X ó eissn 1526-5463 ó 11 ó 5903 ó 0536

doi 10.1287/opre.1110.0940
```

`little.txt` L15–16, L78–82:

```text
L = ãW ” [Little, J. D. C. 1961. A proof for the queuing formula: L = ãW . Oper. Res. 9(3) 383–387]. Over the years,
L = ãW has become widely known as “Little’s Law.” Basically, it is a theorem in queuing theory. It has become well
Little’s Law says that the average number of items in a
queuing system, denoted L, equals the average arrival rate
of items to the system, ã, multiplied by the average waiting
time of an item in the system, W . Thus,
L = ãW 0
```

## 3. Thakur, Rabenseifner and Gropp (2005): the α + nβ cost model

- **Source:** R. Thakur, R. Rabenseifner and W. Gropp, "Optimization of Collective Communication Operations in MPICH", International Journal of High Performance Computing Applications 19(1):49–66, 2005. Read through the Internet Archive snapshot of `http://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf` (`web/2015id_`), saved as `thakur.pdf`.

`thakur.txt` L1–4, L59, title and authors:

```text
Optimization of Collective Communication Operations in MPICH
Rajeev Thakur∗

Rolf Rabenseifner†
William Gropp∗
```

`thakur.txt` L140–153, §3, "Cost Model":

```text
Cost Model

We use a simple model to estimate the cost of the
collective communication algorithms in terms of latency and bandwidth use, and to guide the selection
of algorithms for a particular collective communication operation. This model is similar to the one used
by Van de Geijn [2, 16, 24], Hockney [11], and others.
Although more sophisticated models such as LogP [9]
and LogGP [1] exist, this model is sufficient for our
needs.
We assume that the time taken to send a message
4.1 Allgather
between any two nodes can be modeled as α + nβ,
where α is the latency (or startup time) per message, MPI Allgather is a gather operation in which the
independent of message size, β is the transfer time per data contributed by each process is gathered on
```

## 4. HiGHS

- **Source:** the HiGHS 1.15.1 source tree that mesh builds against, `rdma/.build/highs/HiGHS-1.15.1/README.md`.

`README.md` L46–47, L53:

```text
HiGHS is a high performance serial and parallel solver for large scale sparse
linear optimization problems of the form
HiGHS has primal and dual revised simplex solvers, originally written by Qi Huangfu and further developed by Julian Hall. It also has an interior point solver for LP written by Lukas Schork, an active set solver for QP written by Michael Feldmeier, and a MIP solver written by Leona Gottwald. Other features have been added by Julian Hall and Ivet Galabova, who manages the software engineering of HiGHS and interfaces to C, C#, FORTRAN, Julia and Python.
```
