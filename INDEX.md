# Index

This repo makes a set of Macs work as one machine. Each Mac runs a bridge that sends data over Thunderbolt cables by
RDMA. On top of the bridges is a library with the NCCL calls, bindings for torch and NumPy, and a few example programs.
This index lists each document that tells how the system works now, with a short note on each. The repo also holds
other projects, such as a game bot, and old dated records; they are not listed here.

## Start here

- [design/rubric-actionable.md](design/rubric-actionable.md): the short form of the rubrics. It gives the review
  procedure, what crosses each interface between mesh, collective and data code, and one line for each dimension of
  rubrics M, C and D, with that dimension's line in design/rubrics.md.
- [design/rubrics.md](design/rubrics.md): the full rubrics for mesh (M), collective (C) and data (D) code. It defines
  the three kinds of source, how a change is ranked against the form it replaces, and each dimension with its poles and
  the operator's ranked pairs and quotes. Later sections cover the inversions between kinds, the two channels, failure
  across interfaces, strings as messages and how the operator reviews.

## Change requests

- [design/change-requests.md](design/change-requests.md): what a change request is and how one is written. A change
  request says what code to write, and for each part it names the paper, book or program that says just how to write it.

## The bridge and transport

- [design/data-model.md](design/data-model.md): the hardware and the paths bytes take through it.
  These facts explain why writes use pages and masks, and why their work is compiled before a call.
  The later sections derive the limits on bridge code, routes, and collectives in plain prose near grades 4-6.
