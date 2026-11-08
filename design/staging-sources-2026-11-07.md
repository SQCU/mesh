# Jørring and Scherlis (1986): staging transformations; sources, transcribed

Retrieved 2026-11-07.

- **Source:** Ulrik Jørring and William L. Scherlis, "Compilers and Staging Transformations", in *Proceedings of the 13th ACM SIGACT-SIGPLAN Symposium on Principles of Programming Languages (POPL '86)*, pp. 86–96, 1986. DOI 10.1145/512644.512652 (Crossref). ACM copyright line on p. 86: "© 1986 ACM-0-89791-175-X-1/86-0086 $0.75".
- **Where read:** ACM's scanned PDF, from the Internet Archive snapshot `https://web.archive.org/web/2020id_/https://dl.acm.org/doi/pdf/10.1145/512644.512652`, saved as `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit7/staging/js1986.pdf` (11 pages).
- **How quoted:** the PDF's embedded OCR is unusable (letter-spaced and garbled, e.g. "S t a g i n g IDransfbrmations"). The passages below were therefore **transcribed by reading the page images**, `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit7/staging/p-01.png` (p. 86) and `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit7/staging/p-08.png` (p. 93), rendered with `pdftoppm -r 100`. They are not copied by script, unlike the other records. Italics in the original are marked with asterisks.

## p. 86, Abstract

> Computations can generally be separated into stages, which are distinguished from one another by either frequency of execution or availability of data. *Precomputation* and *frequency reduction* involve moving computation among a collection of stages so that work is done as early as possible (so less time is required in later steps) and as infrequently as possible (to reduce overall time).

## p. 86, §1 "Introduction", left column

> Among the multitude of program improvement methods used by programmers (manually) and by compilers (automatically), a small number of themes seem to predominate. Two of these, usually called *frequency reduction* and *precomputation*, refer to classes of optimizations for detecting computations that can be shifted, respectively, to contexts in which they will be carried out less frequently or to contexts in which efficiency is less critical. We refer to these two techniques as *staging transformations*.

## p. 86, §1, right column

> Precomputation and frequency reduction, taken broadly, encompass activities ranging from code motion optimizations, in which invariant code within a loop is moved out of the loop, to compiling and linking and, indeed, configuration management.

## p. 93, §5 "Comparing the two approaches"

> In both of our approaches we have used program transformation techniques to separate evaluation of a program into multiple stages.

> The pass separation approach introduces explicit data structure, using data transformations, for communication between stages. Indeed, many uses of data structure in programs can be explained in terms of pass separation.

> The meta transformation approach, on the other hand, separates a computation by classifying code into object and meta levels. The communication medium between phases, therefore, is the object-level code itself. With this approach, the later stage is the implicit interpretation of the object-level code. Pass separation, by contrast, provides means for deriving an explicit run-time system to interpret the intermediate data structure.
