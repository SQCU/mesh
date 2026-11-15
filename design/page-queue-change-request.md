# Completed change request: compiled page transfers consumed by the resident bridge

## Collective size

The compiler consumes the supplied collective membership and Thunderbolt-connected topology. It routes the
program's complete transfer set through Garg–Könemann concurrent flow (FOCS 1998), including intermediate nodes
as relays, and orders their computation and transfers through [plan-to-bridge R5](plan-to-bridge-change-request.md).
Every live collective member runs its assigned rank (operator, 3 November 2026, this file at `cdd44406`).

## The page queue

Compilation writes page-transfer entries naming region page, adjacent node, direction, completion word and
position. Each send consumes its producer's publication; each receive publishes its completion for downstream
work. At an intermediate node, the incoming receive's page supplies the outgoing send. The lists reference the
registered region's storage and are reused on repeated executions (operator's page-queue instruction,
3 November; MPI 4.1 §3.9 persistent requests).

## The functions

The compiler consumes declared operand ranges, destination ranks and the topology/capacity operand, and produces
source, relay and destination entries. The bridge consumes those entries through its SEND/RECV interface
(Apple TN3205; [plan-to-bridge R1/R6/R7](plan-to-bridge-change-request.md)). Physical cable and queue-pair objects
remain bridge-owned; programs supply page views. The compiled transfer order supplies the bridge's ordered work.

The bridge reads the device-reported queue-pair budget and applies the provider's reserved count, taking the
smaller capacity at each cable's endpoints (`mesh-verbs.h:device_up/max_pairs`, source at `pre-return-20261103`).
Port children and their supervisor use that revision's stop, reap, flush, close and holder-table lifecycle,
as the operator's 3 November lifecycle instruction specifies. Completion exposes success or an error value
within the invocation's bound, following the operator's bounded-completion instruction and collective-values
items 7/11/13.

## Program consumption

A program supplies its declared operands, placement and rank count to compilation. Each rank consumes the
compiled page-transfer result through the resident bridge's page interface (plan-to-bridge R1/R6/R7).
[Integration item 1](integration-finishing-change-request.md) assigns the domain-wide compiler consumers,
following the operator's [program-domain instruction](documentary-review-2026-11-13.md#program-domain-correction-14-november).
