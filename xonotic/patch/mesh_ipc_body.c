
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdatomic.h>

#define MESH_MAX_REGIONS 8

typedef struct meshregion_s
{
	_Atomic unsigned int req;
	_Atomic unsigned int done;
	unsigned int nfloats;
	unsigned int magic;
	float f[1];
}
meshregion_t;

static meshregion_t *mesh_region[MESH_MAX_REGIONS];
static size_t mesh_regionsize[MESH_MAX_REGIONS];
static unsigned int mesh_seq[MESH_MAX_REGIONS];

static meshregion_t *VM_mesh_resolve(prvm_prog_t *prog, int h, unsigned int need, unsigned int idx)
{
	if (h < 0 || h >= MESH_MAX_REGIONS || !mesh_region[h])
	{
		VM_Warning(prog, "mesh: region %i not open in %s\n", h, prog->name);
		return NULL;
	}
	if (idx + need > mesh_region[h]->nfloats)
	{
		VM_Warning(prog, "mesh: range %u+%u exceeds %u floats in %s\n", idx, need, mesh_region[h]->nfloats, prog->name);
		return NULL;
	}
	return mesh_region[h];
}

void VM_mesh_open(prvm_prog_t *prog)
{
	const char *name;
	unsigned int nfloats;
	size_t bytes;
	int fd, h;
	meshregion_t *r;

	VM_SAFEPARMCOUNT(2, VM_mesh_open);
	PRVM_G_FLOAT(OFS_RETURN) = -1;

	name = PRVM_G_STRING(OFS_PARM0);
	nfloats = (unsigned int)PRVM_G_FLOAT(OFS_PARM1);
	bytes = sizeof(meshregion_t) + (size_t)nfloats * sizeof(float);

	for (h = 0; h < MESH_MAX_REGIONS; h++)
		if (!mesh_region[h])
			break;
	if (h == MESH_MAX_REGIONS)
	{
		VM_Warning(prog, "mesh_open: all %i regions in use in %s\n", MESH_MAX_REGIONS, prog->name);
		return;
	}

	fd = shm_open(name, O_CREAT | O_RDWR, 0600);
	if (fd < 0)
	{
		VM_Warning(prog, "mesh_open: shm_open(%s) failed in %s\n", name, prog->name);
		return;
	}
	ftruncate(fd, (off_t)bytes);
	r = (meshregion_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (r == MAP_FAILED)
	{
		VM_Warning(prog, "mesh_open: mmap(%s) failed in %s\n", name, prog->name);
		return;
	}

	r->nfloats = nfloats;
	r->magic = 0x4d455348;
	mesh_region[h] = r;
	mesh_regionsize[h] = bytes;
	mesh_seq[h] = atomic_load_explicit(&r->req, memory_order_relaxed);
	PRVM_G_FLOAT(OFS_RETURN) = h;
}

void VM_mesh_close(prvm_prog_t *prog)
{
	int h;
	VM_SAFEPARMCOUNT(1, VM_mesh_close);
	h = (int)PRVM_G_FLOAT(OFS_PARM0);
	if (h < 0 || h >= MESH_MAX_REGIONS || !mesh_region[h])
		return;
	munmap(mesh_region[h], mesh_regionsize[h]);
	mesh_region[h] = NULL;
	mesh_regionsize[h] = 0;
}

void VM_mesh_set(prvm_prog_t *prog)
{
	meshregion_t *r;
	unsigned int i;
	VM_SAFEPARMCOUNT(3, VM_mesh_set);
	i = (unsigned int)PRVM_G_FLOAT(OFS_PARM1);
	r = VM_mesh_resolve(prog, (int)PRVM_G_FLOAT(OFS_PARM0), 1, i);
	if (!r)
		return;
	r->f[i] = PRVM_G_FLOAT(OFS_PARM2);
}

void VM_mesh_get(prvm_prog_t *prog)
{
	meshregion_t *r;
	unsigned int i;
	VM_SAFEPARMCOUNT(2, VM_mesh_get);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	i = (unsigned int)PRVM_G_FLOAT(OFS_PARM1);
	r = VM_mesh_resolve(prog, (int)PRVM_G_FLOAT(OFS_PARM0), 1, i);
	if (!r)
		return;
	PRVM_G_FLOAT(OFS_RETURN) = r->f[i];
}

void VM_mesh_gather(prvm_prog_t *prog)
{
	meshregion_t *r;
	unsigned int dst, n, e, first;
	int fld, stride;
	VM_SAFEPARMCOUNT(5, VM_mesh_gather);
	dst = (unsigned int)PRVM_G_FLOAT(OFS_PARM1);
	fld = PRVM_G_INT(OFS_PARM2);
	first = (unsigned int)PRVM_G_FLOAT(OFS_PARM3);
	n = (unsigned int)PRVM_G_FLOAT(OFS_PARM4);
	r = VM_mesh_resolve(prog, (int)PRVM_G_FLOAT(OFS_PARM0), n, dst);
	if (!r)
		return;
	stride = prog->entityfields;
	if (fld < 0 || fld >= stride || (first + n) > (unsigned int)prog->max_edicts)
	{
		VM_Warning(prog, "mesh_gather: field %i or edicts %u+%u out of range in %s\n", fld, first, n, prog->name);
		return;
	}
	for (e = 0; e < n; e++)
		r->f[dst + e] = (float)prog->edictsfields[(size_t)(first + e) * stride + fld];
}

void VM_mesh_scatter(prvm_prog_t *prog)
{
	meshregion_t *r;
	unsigned int src, n, e, first;
	int fld, stride;
	VM_SAFEPARMCOUNT(5, VM_mesh_scatter);
	src = (unsigned int)PRVM_G_FLOAT(OFS_PARM1);
	fld = PRVM_G_INT(OFS_PARM2);
	first = (unsigned int)PRVM_G_FLOAT(OFS_PARM3);
	n = (unsigned int)PRVM_G_FLOAT(OFS_PARM4);
	r = VM_mesh_resolve(prog, (int)PRVM_G_FLOAT(OFS_PARM0), n, src);
	if (!r)
		return;
	stride = prog->entityfields;
	if (fld < 0 || fld >= stride || (first + n) > (unsigned int)prog->max_edicts)
	{
		VM_Warning(prog, "mesh_scatter: field %i or edicts %u+%u out of range in %s\n", fld, first, n, prog->name);
		return;
	}
	for (e = 0; e < n; e++)
		prog->edictsfields[(size_t)(first + e) * stride + fld] = (prvm_vec_t)r->f[src + e];
}

void VM_mesh_publish(prvm_prog_t *prog)
{
	int h;
	VM_SAFEPARMCOUNT(1, VM_mesh_publish);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	h = (int)PRVM_G_FLOAT(OFS_PARM0);
	if (!VM_mesh_resolve(prog, h, 0, 0))
		return;
	mesh_seq[h]++;
	atomic_store_explicit(&mesh_region[h]->req, mesh_seq[h], memory_order_release);
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)mesh_seq[h];
}

void VM_mesh_poll(prvm_prog_t *prog)
{
	int h;
	VM_SAFEPARMCOUNT(1, VM_mesh_poll);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	h = (int)PRVM_G_FLOAT(OFS_PARM0);
	if (!VM_mesh_resolve(prog, h, 0, 0))
		return;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)atomic_load_explicit(&mesh_region[h]->done, memory_order_acquire);
}
