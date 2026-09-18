# Constructed TX path

| Instruction from the fixed-cell read to native SEND acceptance | D0 row | Source |
| --- | --- | --- |
| `ldapr x8,[x23]`: load the next cell. `x23` is the slot's register cursor. | M04, M16 | `rdma/mesh-flow.c:199` |
| `ldp x9,x0,[x8,#-32]`: native entry and QP, after affine `records + 32*event`. | M05 | `rdma/mesh-flow.c:204` |
| `ldp x1,x2,[x8,#-16]`: prepared WR and native failure-output address. | M05, M06, M15 | `rdma/mesh-flow.c:204` |
| `blr x9`: enter the native post implementation with those three arguments. | M05, M06, M07, M15 | `rdma/mesh-flow.c:205` |
| `stlr xzr,[x23]`: clear the accepted cell. Register add/compare/select advances only this slot. | M04, M16 | `rdma/mesh-flow.c:210` |

| Removed TX source load site | Replacement |
| --- | --- |
| `reader->inputs` | M16 register cursor |
| `reader->count` | M16 register end |
| `reader->cursor` | M16 register cursor |
| `input->slots` | M16 register cursor |
| `input->position` | M16 register cursor |
| `input->mask` | M16 register end |
| `send_edges[first].end` | One M04 cell per native request |
| `source->span.addr` for a wire tag store | Deleted wire tag |
| `source->tag_row` | Deleted wire tag |
| `source->queue` | Native QP already in M05 |
| Retry `ready->head` | Deleted retry queue |
| Retry `ready->tail` | Deleted retry queue |
| Retry `ready->first` | Deleted retry queue |
| Retry `ready->mask` | Deleted retry queue |
| Retry range `first` | M16 cursor stays on an unaccepted request |
| Retry range `end` | Deleted retry range |
| Retry range `invocation` | Deleted wire tag |

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 3 AArch64 instructions on the constructed M04 → M05 → native SEND path; one dependent address edge. |
| Deleted | 17 TX source load sites; this is not an inferred instruction count for the old compiler output. |
| Native ownership boundary | M06/M07 are native API inputs. The native driver's internal instructions are not attributed to Mesh. |
| Unconstructed transition portions | Producer M13 and the complete receive/consumer transition are not included in the 3-load claim. |
