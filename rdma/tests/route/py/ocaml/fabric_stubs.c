/* The OCaml plugin's C face (fabric.ml): route_ml_init starts the OCaml runtime once; route_ml_fabric_stats hands
   fabric.ml n contiguous struct route_fabric (route.h) where they lie, and out [n][4] doubles, as external Bigarrays */
#include <stddef.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include "../../route.h"

static const value *stats;
int route_ml_init(void) {
  static char *argv[] = {"route_ml", NULL};
  if (!stats) { caml_startup(argv); stats = caml_named_value("fabric_stats"); }
  return stats ? 0 : -1;
}
size_t route_ml_fabric_size(void) { return sizeof(struct route_fabric); }
int route_ml_fabric_stats(const void *fabrics, int n, double *out) {
  CAMLparam0();
  CAMLlocal4(ints, dbls, off, res);
  _Static_assert(sizeof(struct route_fabric) % 8 == 0, "a fabric a whole number of doubles");
  const size_t S = sizeof(struct route_fabric);
  const long field[9] = {offsetof(struct route_fabric, nodes) / 4, offsetof(struct route_fabric, ncompute) / 4, offsetof(struct route_fabric, lanes) / 4,
                         offsetof(struct route_fabric, compute) / 4, offsetof(struct route_fabric, to) / 4, offsetof(struct route_fabric, rate) / 8,
                         offsetof(struct route_fabric, latency) / 8, (long)(S / 4), (long)(S / 8)};
  ints = caml_ba_alloc_dims(CAML_BA_INT32 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1, (void *)fabrics, (intnat)(n * S / 4));
  dbls = caml_ba_alloc_dims(CAML_BA_FLOAT64 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1, (void *)fabrics, (intnat)(n * S / 8));
  off = caml_alloc(9, 0);
  for (int k = 0; k < 9; k++) Store_field(off, k, Val_long(field[k]));
  value args[4] = {ints, dbls, off, caml_ba_alloc_dims(CAML_BA_FLOAT64 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 2, out, (intnat)n, (intnat)4)};
  res = caml_callbackN_exn(*stats, 4, args);
  CAMLreturnT(int, Is_exception_result(res) ? -1 : 0);
}
