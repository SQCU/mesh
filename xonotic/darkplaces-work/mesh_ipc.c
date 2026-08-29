#include "mesh.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#include "mesh-client.c"
#pragma GCC diagnostic pop

#define MESH_XON_MAGIC     0x584d5348u
#define MESH_XON_VERSION   1
#define MESH_XON_HDRBYTES  32u
#define MESH_XON_REQ       1
#define MESH_XON_RESP      2
#define MESH_XON_MAXCHUNK  64u
#define MESH_XON_TXSLOTS   4096u
#define MESH_XON_HANDLES   4
#define MESH_XON_RESPWIDTH 8u

typedef struct meshxwire_s
{
	uint32_t magic;
	uint16_t version;
	uint16_t kind;
	uint32_t req_id;
	uint32_t tick;
	uint16_t width;
	uint16_t rows;
	uint16_t chunk;
	uint16_t chunks;
	uint32_t rows_total;
	uint32_t flags;
}
meshxwire_t;

_Static_assert(sizeof(meshxwire_t) == MESH_XON_HDRBYTES, "mesh wire header is 32 bytes");

typedef struct meshxhandle_s
{
	int used;
	int node;
	uint32_t width;
	uint32_t maxrows;
	uint32_t reqrows;
	uint32_t resprows;
	float *req;
	float *resp;
	uint32_t txbase;
	uint32_t cursor;
	uint32_t req_id;
	uint32_t inflight_id;
	uint32_t inflight_chunks;
	uint64_t mask;
	uint32_t done_id;
	uint32_t done_rows;
	uint32_t done_tick;
	uint32_t dropped;
	uint32_t shortwrites;
	uint32_t respchunks;
}
meshxhandle_t;

static meshxhandle_t meshx_h[MESH_XON_HANDLES];
static unsigned char *meshx_arena;
static size_t meshx_nslots, meshx_stride, meshx_usable;

static int meshx_attach(void)
{
	if (meshx_arena)
		return 0;
	meshx_arena = (unsigned char *)mesh_open(&meshx_nslots, &meshx_stride, &meshx_usable);
	return meshx_arena ? 0 : -1;
}

static meshxhandle_t *meshx_get(int h)
{
	if (h < 0 || h >= MESH_XON_HANDLES || !meshx_h[h].used)
		return NULL;
	return &meshx_h[h];
}

static int meshx_open(int node, uint32_t width, uint32_t maxrows)
{
	uint32_t payload, reqrows, resprows;
	meshxhandle_t *m;
	int h;

	if (width < 1 || width > 256 || maxrows < 1)
		return -1;
	if (meshx_attach())
		return -1;

	payload = (uint32_t)meshx_usable - MESH_XON_HDRBYTES;
	reqrows = payload / (width * 4);
	resprows = payload / (MESH_XON_RESPWIDTH * 4);
	if (!reqrows)
		return -1;
	if (maxrows > reqrows * MESH_XON_MAXCHUNK)
		maxrows = reqrows * MESH_XON_MAXCHUNK;

	for (h = 0; h < MESH_XON_HANDLES; h++)
		if (meshx_h[h].used && meshx_h[h].node == node && meshx_h[h].width == width && meshx_h[h].maxrows == maxrows)
			return h;
	for (h = 0; h < MESH_XON_HANDLES; h++)
		if (!meshx_h[h].used)
			break;
	if (h == MESH_XON_HANDLES)
		return -1;
	if ((size_t)(h + 1) * MESH_XON_TXSLOTS > meshx_nslots)
		return -1;

	m = &meshx_h[h];
	memset(m, 0, sizeof(*m));
	m->req = (float *)calloc((size_t)maxrows * width, sizeof(float));
	m->resp = (float *)calloc((size_t)maxrows * MESH_XON_RESPWIDTH, sizeof(float));
	if (!m->req || !m->resp)
	{
		free(m->req);
		free(m->resp);
		memset(m, 0, sizeof(*m));
		return -1;
	}
	m->used = 1;
	m->node = node;
	m->width = width;
	m->maxrows = maxrows;
	m->reqrows = reqrows;
	m->resprows = resprows;
	m->txbase = (uint32_t)h * MESH_XON_TXSLOTS;
	return h;
}

static uint32_t meshx_publish(meshxhandle_t *m, uint32_t tick, uint32_t nrows)
{
	uint32_t chunks, c, sent;
	unsigned char *base;
	size_t want;

	if (nrows > m->maxrows)
		nrows = m->maxrows;
	if (!nrows)
		return 0;
	chunks = (nrows + m->reqrows - 1) / m->reqrows;
	if (m->cursor + chunks > MESH_XON_TXSLOTS)
		m->cursor = 0;

	m->req_id++;
	base = meshx_arena + (size_t)(m->txbase + m->cursor) * meshx_stride;
	for (c = 0, sent = 0; c < chunks; c++)
	{
		meshxwire_t w;
		unsigned char *p = base + (size_t)c * meshx_stride;
		uint32_t rows = nrows - sent < m->reqrows ? nrows - sent : m->reqrows;

		w.magic = MESH_XON_MAGIC;
		w.version = MESH_XON_VERSION;
		w.kind = MESH_XON_REQ;
		w.req_id = m->req_id;
		w.tick = tick;
		w.width = (uint16_t)m->width;
		w.rows = (uint16_t)rows;
		w.chunk = (uint16_t)c;
		w.chunks = (uint16_t)chunks;
		w.rows_total = nrows;
		w.flags = 0;
		memcpy(p, &w, MESH_XON_HDRBYTES);
		memcpy(p + MESH_XON_HDRBYTES, m->req + (size_t)sent * m->width, (size_t)rows * m->width * sizeof(float));
		sent += rows;
	}

	want = (size_t)chunks * meshx_usable;
	if (mesh_write(base, want, m->node) < want)
		m->shortwrites++;
	m->cursor += chunks;
	return m->req_id;
}

static int meshx_slot(meshxhandle_t *m, const unsigned char *q, size_t b)
{
	meshxwire_t w;
	uint64_t full;

	if (b < (size_t)MESH_XON_HDRBYTES)
		return 0;
	memcpy(&w, q, MESH_XON_HDRBYTES);
	if (w.magic != MESH_XON_MAGIC || w.version != MESH_XON_VERSION || w.kind != MESH_XON_RESP)
		return 0;
	if (w.width != MESH_XON_RESPWIDTH || !w.chunks || (uint32_t)w.chunks > MESH_XON_MAXCHUNK || w.chunk >= w.chunks)
		return 0;
	if ((size_t)w.rows * w.width * sizeof(float) > b - MESH_XON_HDRBYTES)
		return 0;
	if ((uint32_t)w.chunk * m->resprows + w.rows > m->maxrows || w.rows_total > m->maxrows)
		return 0;

	if (w.req_id > m->inflight_id)
	{
		full = m->inflight_chunks >= 64 ? ~(uint64_t)0 : ((uint64_t)1 << m->inflight_chunks) - 1;
		if (m->inflight_id && m->mask != full)
			m->dropped++;
		m->inflight_id = w.req_id;
		m->inflight_chunks = w.chunks;
		m->mask = 0;
	}
	else if (w.req_id < m->inflight_id)
		return 0;

	m->mask |= (uint64_t)1 << w.chunk;
	m->respchunks++;
	memcpy(m->resp + (size_t)w.chunk * m->resprows * MESH_XON_RESPWIDTH,
		q + MESH_XON_HDRBYTES, (size_t)w.rows * w.width * sizeof(float));

	full = m->inflight_chunks >= 64 ? ~(uint64_t)0 : ((uint64_t)1 << m->inflight_chunks) - 1;
	if (m->mask == full)
	{
		m->done_id = w.req_id;
		m->done_rows = w.rows_total;
		m->done_tick = w.tick;
	}
	return 1;
}

static uint32_t meshx_poll(meshxhandle_t *m)
{
	for (;;)
	{
		void *q = NULL;
		int from = 0;
		size_t b = mesh_read(&q, &from);

		if (!b)
			break;
		meshx_slot(m, (const unsigned char *)q, b);
	}
	return m->done_id;
}

static double meshx_stat(meshxhandle_t *m, int sel)
{
	switch (sel & 15)
	{
	case 0: return m->req_id;
	case 1: return m->done_id;
	case 2: return (double)m->req_id - (double)m->done_id;
	case 3: return m->width;
	case 4: return m->maxrows;
	case 5: return meshx_arena ? 1 : 0;
	case 6: return m->dropped;
	case 7: return m->reqrows;
	case 8: return MESH_XON_TXSLOTS;
	case 9: return m->shortwrites;
	case 10: return m->done_rows;
	case 11: return m->respchunks;
	case 12: return m->done_tick;
	case 13: return (double)meshx_nslots;
	case 14: return (double)meshx_usable;
	default: return m->node;
	}
}

#ifndef MESH_XON_CORE_ONLY

static meshxhandle_t *VM_mesh_resolve(prvm_prog_t *prog, const char *who)
{
	meshxhandle_t *m = meshx_get((int)PRVM_G_FLOAT(OFS_PARM0));
	if (!m)
		VM_Warning(prog, "%s: handle %i not open in %s\n", who, (int)PRVM_G_FLOAT(OFS_PARM0), prog->name);
	return m;
}

static int VM_mesh_span(prvm_prog_t *prog, const char *who, meshxhandle_t *m, uint32_t width, uint32_t col, int *fld, uint32_t *first, uint32_t *n)
{
	*fld = PRVM_G_INT(OFS_PARM2);
	*first = (uint32_t)PRVM_G_FLOAT(OFS_PARM3);
	*n = (uint32_t)PRVM_G_FLOAT(OFS_PARM4);
	if (col >= width || *n > m->maxrows)
	{
		VM_Warning(prog, "%s: column %u of %u rows %u of %u out of range in %s\n", who, col, width, *n, m->maxrows, prog->name);
		return 0;
	}
	if (*fld < 0 || *fld >= prog->entityfields || *first > (uint32_t)prog->max_edicts || *n > (uint32_t)prog->max_edicts - *first)
	{
		VM_Warning(prog, "%s: field %i edicts %u+%u out of range in %s\n", who, *fld, *first, *n, prog->name);
		return 0;
	}
	return 1;
}

void VM_mesh_open(prvm_prog_t *prog)
{
	VM_SAFEPARMCOUNT(3, VM_mesh_open);
	PRVM_G_FLOAT(OFS_RETURN) = meshx_open((int)PRVM_G_FLOAT(OFS_PARM0), (uint32_t)PRVM_G_FLOAT(OFS_PARM1), (uint32_t)PRVM_G_FLOAT(OFS_PARM2));
}

void VM_mesh_gather(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	uint32_t col, first, n, row;
	size_t stride;
	int fld;

	VM_SAFEPARMCOUNT(5, VM_mesh_gather);
	m = VM_mesh_resolve(prog, "mesh_gather");
	if (!m)
		return;
	col = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	if (!VM_mesh_span(prog, "mesh_gather", m, m->width, col, &fld, &first, &n))
		return;
	stride = (size_t)prog->entityfields;
	for (row = 0; row < n; row++)
		m->req[(size_t)row * m->width + col] = (float)prog->edictsfields[(size_t)(first + row) * stride + fld];
}

void VM_mesh_scatter(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	uint32_t col, first, n, row;
	size_t stride;
	int fld;

	VM_SAFEPARMCOUNT(5, VM_mesh_scatter);
	m = VM_mesh_resolve(prog, "mesh_scatter");
	if (!m)
		return;
	col = (uint32_t)PRVM_G_FLOAT(OFS_PARM1);
	if (!VM_mesh_span(prog, "mesh_scatter", m, MESH_XON_RESPWIDTH, col, &fld, &first, &n))
		return;
	if (n > m->done_rows)
		n = m->done_rows;
	stride = (size_t)prog->entityfields;
	for (row = 0; row < n; row++)
		prog->edictsfields[(size_t)(first + row) * stride + fld] = (prvm_vec_t)m->resp[(size_t)row * MESH_XON_RESPWIDTH + col];
}

void VM_mesh_publish(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	VM_SAFEPARMCOUNT(3, VM_mesh_publish);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_publish");
	if (!m)
		return;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)meshx_publish(m, (uint32_t)PRVM_G_FLOAT(OFS_PARM1), (uint32_t)PRVM_G_FLOAT(OFS_PARM2));
}

void VM_mesh_poll(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	VM_SAFEPARMCOUNT(1, VM_mesh_poll);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_poll");
	if (!m)
		return;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)meshx_poll(m);
}

void VM_mesh_stat(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	VM_SAFEPARMCOUNT(2, VM_mesh_stat);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_stat");
	if (!m)
		return;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)meshx_stat(m, (int)PRVM_G_FLOAT(OFS_PARM1));
}

#endif
