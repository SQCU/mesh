#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "mesh_shm.h"

typedef struct meshhandle_s
{
	mesh_hdr_t *hdr;
	size_t bytes;
	float *req;
	float *resp;
	uint32_t nreq;
	uint32_t nresp;
	uint64_t seq;
	uint64_t delivered;
	uint64_t misses;
	char name[64];
}
meshhandle_t;

static meshhandle_t mesh_h[MESH_MAX_REGIONS];

static meshhandle_t *VM_mesh_resolve(prvm_prog_t *prog, const char *who, int h)
{
	if (h < 0 || h >= MESH_MAX_REGIONS || !mesh_h[h].hdr)
	{
		VM_Warning(prog, "%s: region %i not open in %s\n", who, h, prog->name);
		return NULL;
	}
	return &mesh_h[h];
}

static void VM_mesh_drain(meshhandle_t *m)
{
	m->delivered = mesh_get_resp(m->hdr, m->delivered, m->resp, &m->misses);
}

void VM_mesh_open(prvm_prog_t *prog)
{
	const char *name;
	uint32_t nreq, nresp;
	size_t bytes;
	int fd, h;
	mesh_hdr_t *p;

	VM_SAFEPARMCOUNT(3, VM_mesh_open);
	PRVM_G_FLOAT(OFS_RETURN) = -1;

	name = PRVM_G_STRING(OFS_PARM0);
	nreq = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	nresp = (uint32_t)PRVM_G_FLOAT(OFS_PARM2);
	bytes = mesh_bytes(nreq, nresp);

	for (h = 0; h < MESH_MAX_REGIONS; h++)
		if (!mesh_h[h].hdr)
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
	p = (mesh_hdr_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (p == MAP_FAILED)
	{
		VM_Warning(prog, "mesh_open: mmap(%s, %u) failed in %s\n", name, (unsigned)bytes, prog->name);
		return;
	}

	mesh_reset(p, nreq, nresp);

	mesh_h[h].hdr = p;
	mesh_h[h].bytes = bytes;
	mesh_h[h].nreq = nreq;
	mesh_h[h].nresp = nresp;
	mesh_h[h].seq = 0;
	mesh_h[h].delivered = 0;
	mesh_h[h].misses = 0;
	mesh_h[h].req = (float *)Mem_Alloc(prog->progs_mempool, (nreq + 1) * sizeof(float));
	mesh_h[h].resp = (float *)Mem_Alloc(prog->progs_mempool, (nresp + 1) * sizeof(float));
	memset(mesh_h[h].req, 0, (nreq + 1) * sizeof(float));
	memset(mesh_h[h].resp, 0, (nresp + 1) * sizeof(float));
	strlcpy(mesh_h[h].name, name, sizeof(mesh_h[h].name));
	PRVM_G_FLOAT(OFS_RETURN) = h;
}

void VM_mesh_close(prvm_prog_t *prog)
{
	meshhandle_t *m;
	VM_SAFEPARMCOUNT(1, VM_mesh_close);
	m = VM_mesh_resolve(prog, "mesh_close", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	munmap(m->hdr, m->bytes);
	Mem_Free(m->req);
	Mem_Free(m->resp);
	memset(m, 0, sizeof(*m));
}

void VM_mesh_set(prvm_prog_t *prog)
{
	meshhandle_t *m;
	uint32_t i;
	VM_SAFEPARMCOUNT(3, VM_mesh_set);
	m = VM_mesh_resolve(prog, "mesh_set", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	i = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	if (i >= m->nreq)
	{
		VM_Warning(prog, "mesh_set: index %u >= %u in %s\n", i, m->nreq, prog->name);
		return;
	}
	m->req[i] = (float)PRVM_G_FLOAT(OFS_PARM2);
}

void VM_mesh_get(prvm_prog_t *prog)
{
	meshhandle_t *m;
	uint32_t i;
	VM_SAFEPARMCOUNT(2, VM_mesh_get);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_get", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	i = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	if (i >= m->nresp)
	{
		VM_Warning(prog, "mesh_get: index %u >= %u in %s\n", i, m->nresp, prog->name);
		return;
	}
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)m->resp[i];
}

static int VM_mesh_span(prvm_prog_t *prog, const char *who, uint32_t cap, uint32_t off, int *fld, uint32_t *first, uint32_t *n)
{
	*fld = PRVM_G_INT(OFS_PARM2);
	*first = (uint32_t)PRVM_G_FLOAT(OFS_PARM3);
	*n = (uint32_t)PRVM_G_FLOAT(OFS_PARM4);
	if (off > cap || *n > cap - off)
	{
		VM_Warning(prog, "%s: span %u+%u exceeds %u floats in %s\n", who, off, *n, cap, prog->name);
		return 0;
	}
	if (*fld < 0 || *fld >= prog->entityfields || *first > (uint32_t)prog->max_edicts || *n > (uint32_t)prog->max_edicts - *first)
	{
		VM_Warning(prog, "%s: field %i edicts %u+%u out of range in %s\n", who, *fld, *first, *n, prog->name);
		return 0;
	}
	return 1;
}

void VM_mesh_gather(prvm_prog_t *prog)
{
	meshhandle_t *m;
	uint32_t dst, n, e, first;
	int fld;
	size_t stride;
	VM_SAFEPARMCOUNT(5, VM_mesh_gather);
	m = VM_mesh_resolve(prog, "mesh_gather", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	dst = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	if (!VM_mesh_span(prog, "mesh_gather", m->nreq, dst, &fld, &first, &n))
		return;
	stride = (size_t)prog->entityfields;
	for (e = 0; e < n; e++)
		m->req[dst + e] = (float)prog->edictsfields[(size_t)(first + e) * stride + fld];
}

void VM_mesh_scatter(prvm_prog_t *prog)
{
	meshhandle_t *m;
	uint32_t src, n, e, first;
	int fld;
	size_t stride;
	VM_SAFEPARMCOUNT(5, VM_mesh_scatter);
	m = VM_mesh_resolve(prog, "mesh_scatter", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	src = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	if (!VM_mesh_span(prog, "mesh_scatter", m->nresp, src, &fld, &first, &n))
		return;
	stride = (size_t)prog->entityfields;
	for (e = 0; e < n; e++)
		prog->edictsfields[(size_t)(first + e) * stride + fld] = (prvm_vec_t)m->resp[src + e];
}

void VM_mesh_publish(prvm_prog_t *prog)
{
	meshhandle_t *m;
	VM_SAFEPARMCOUNT(1, VM_mesh_publish);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_publish", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	m->seq++;
	mesh_put_req(m->hdr, m->seq, m->req);
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)m->seq;
}

void VM_mesh_poll(prvm_prog_t *prog)
{
	meshhandle_t *m;
	VM_SAFEPARMCOUNT(1, VM_mesh_poll);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_poll", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	VM_mesh_drain(m);
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)m->delivered;
}

void VM_mesh_wait(prvm_prog_t *prog)
{
	meshhandle_t *m;
	double deadline;
	uint64_t want;
	VM_SAFEPARMCOUNT(3, VM_mesh_wait);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_wait", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	want = (uint64_t)PRVM_G_FLOAT(OFS_PARM1);
	deadline = Sys_DirtyTime() + (double)PRVM_G_FLOAT(OFS_PARM2) * 1e-6;
	do
		VM_mesh_drain(m);
	while (m->delivered < want && Sys_DirtyTime() < deadline);
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)m->delivered;
}

void VM_mesh_stat(prvm_prog_t *prog)
{
	meshhandle_t *m;
	uint64_t v[8];
	VM_SAFEPARMCOUNT(2, VM_mesh_stat);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_stat", (int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		return;
	v[0] = m->seq;
	v[1] = m->delivered;
	v[2] = m->seq - m->delivered;
	v[3] = m->nreq;
	v[4] = m->nresp;
	v[5] = atomic_load_explicit(&m->hdr->solver_alive, memory_order_relaxed);
	v[6] = m->misses;
	v[7] = MESH_DEPTH;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)v[(int)PRVM_G_FLOAT(OFS_PARM1) & 7];
}
