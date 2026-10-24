#include "mesh-write-order.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  const double capacity[] = {10, 20};
  struct mesh_write_task early[] = {
    {.arc=0,.parent=UINT32_MAX,.bytes=100,.release=0},
    {.arc=1,.parent=0,.bytes=100},
    {.arc=1,.parent=UINT32_MAX,.bytes=20,.release=2}
  };
  assert(!mesh_write_order(early,3,capacity,2,10));
  assert(early[1].order < early[2].order);
  assert(fabs(early[1].start-1)<1e-9 && fabs(early[1].finish-10.5)<1e-9);
  assert(fabs(early[2].start-10.5)<1e-9);
  struct mesh_write_task late[] = {
    {.arc=0,.parent=UINT32_MAX,.bytes=100,.release=0},
    {.arc=0,.parent=UINT32_MAX,.bytes=100,.release=0},
    {.arc=1,.parent=1,.bytes=100},
    {.arc=1,.parent=UINT32_MAX,.bytes=20,.release=2}
  };
  assert(!mesh_write_order(late,4,capacity,2,10));
  assert(late[3].order < late[2].order);
  assert(fabs(late[3].start-2)<1e-9 && fabs(late[2].start-11)<1e-9);
  for (unsigned k=0;k<4;k++) if(late[k].parent!=UINT32_MAX) assert(late[late[k].parent].order<late[k].order);
  struct mesh_write_task cycle[] = {{.arc=0,.parent=1,.bytes=10},{.arc=1,.parent=0,.bytes=10}};
  assert(mesh_write_order(cycle,2,capacity,2,10)==ELOOP);
  puts("streaming relay overlap, queued upstream arrival, local/relay interleaving, and cycle detection passed");
}
