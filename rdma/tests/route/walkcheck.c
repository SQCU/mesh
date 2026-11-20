/* split.h: the walk (split_walk_at/next, the CPU sampler's and each GPU lane's chunk) against random access (split_set_token)
   over every destination set, degree 1-5 and 200 masks each, chunks from anywhere. cc -I . -I ../.. -I .. walkcheck.c */
#include "route.h"
#include <stdio.h>
int main(void) {   /* the walk and the random-access tokens agree, every draw shape */
  struct route_draw d = {0}; long bad = 0, n = 0;
  for (uint32_t deg = 1; deg <= 5; deg++) for (uint32_t S = 1; S < 64; S++) for (uint32_t t = 0; t < 200; t++) {
    d.deg = deg; d.S = S; d.valid = 0x3fffffff; d.mask = (t * 2654435761u) ^ (t << 7);
    struct split_set a = split_set_of(&d); struct split_walk w = split_walk_at(&a, 0, a.n);
    for (uint32_t i = 0; w.left; i++, split_walk_next(&w)) { n++; bad += w.token != split_set_token(&a, i); }
    for (uint32_t c = 1; c <= 7; c += 3) for (uint32_t i0 = 0; i0 < a.n; i0 += c) {   /* chunks from anywhere */
      struct split_walk v = split_walk_at(&a, i0, c);
      for (uint32_t i = i0; v.left; i++, split_walk_next(&v)) { n++; bad += v.token != split_set_token(&a, i); }
    }
  }
  printf("walk vs random access: %ld of %ld tokens differ\n", bad, n); return bad != 0;
}
