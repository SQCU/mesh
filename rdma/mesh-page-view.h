#ifndef MESH_PAGE_VIEW_H
#define MESH_PAGE_VIEW_H

#include <errno.h>
#include <stdint.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

struct mesh_view { void *address; size_t bytes; };

static inline int mesh_page_view(void *arena, uint64_t minimum_offset, const void *offsets, size_t count, size_t stride,
                               size_t quantum, struct mesh_view *mapping) {
    if (!count || !quantum || quantum % vm_page_size || count > SIZE_MAX / quantum) return EINVAL;
    const size_t span = count * quantum;
    mach_vm_address_t base = 0;
    const mach_port_t task = mach_task_self();
    if (mach_vm_allocate(task, &base, span, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) return ENOMEM;
    int status = 0;
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t *field = (const uintptr_t *)((const char *)offsets + i * stride);
        const uintptr_t at = __atomic_load_n(field, __ATOMIC_RELAXED);
        const uintptr_t source = (uintptr_t)arena + at;
        if (at < minimum_offset || source % vm_page_size) { status = EFAULT; break; }
        mach_vm_address_t destination = base + i * quantum;
        vm_prot_t current, maximum;
        if (mach_vm_remap(task, &destination, quantum, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                          task, source, FALSE, &current, &maximum, VM_INHERIT_NONE) != KERN_SUCCESS ||
            !(current & VM_PROT_WRITE)) { status = EACCES; break; }
    }
    if (status) {
        mach_vm_deallocate(task, base, span);
        return status;
    }
    *mapping = (struct mesh_view){(void *)base, span};
    return 0;
}

static inline void mesh_page_view_unmap(struct mesh_view mapping) {
    mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)mapping.address, mapping.bytes);
}

#endif
