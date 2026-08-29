#include <stdio.h>
#define MESH_XON_CORE_ONLY
#include "../engine/mesh_ipc.c"

static void fillrows(meshxhandle_t *m, uint32_t nrows, uint32_t tick)
{
	uint32_t r, c;
	for (r = 0; r < nrows; r++)
		for (c = 0; c < m->width; c++)
			m->req[(size_t)r * m->width + c] = (float)((r * 16u + c + tick) % 997u) - 498.0f;
}

static size_t mkresp(unsigned char *p, uint32_t req_id, uint32_t tick, uint32_t rows_total, uint32_t chunk, uint32_t chunks, uint32_t rows, uint32_t resprows)
{
	meshxwire_t w;
	uint32_t i;
	float *v = (float *)(p + MESH_XON_HDRBYTES);

	w.magic = MESH_XON_MAGIC;
	w.version = MESH_XON_VERSION;
	w.kind = MESH_XON_RESP;
	w.req_id = req_id;
	w.tick = tick;
	w.width = MESH_XON_RESPWIDTH;
	w.rows = (uint16_t)rows;
	w.chunk = (uint16_t)chunk;
	w.chunks = (uint16_t)chunks;
	w.rows_total = rows_total;
	w.flags = 0;
	memcpy(p, &w, MESH_XON_HDRBYTES);
	for (i = 0; i < rows * MESH_XON_RESPWIDTH; i++)
		v[i] = (float)(chunk * resprows * MESH_XON_RESPWIDTH + i);
	return MESH_XON_HDRBYTES + (size_t)rows * MESH_XON_RESPWIDTH * sizeof(float);
}

static int reassembly(meshxhandle_t *m, uint32_t nrows)
{
	static unsigned char slotbuf[8192];
	uint32_t chunks = (nrows + m->resprows - 1) / m->resprows;
	uint32_t id = m->req_id + 1000;
	uint32_t c, rows;
	int bad = 0;
	size_t b;

	for (c = chunks; c-- > 0;)
	{
		rows = nrows - c * m->resprows < m->resprows ? nrows - c * m->resprows : m->resprows;
		b = mkresp(slotbuf, id, 77, nrows, c, chunks, rows, m->resprows);
		meshx_slot(m, slotbuf, b);
		if (c)
			meshx_slot(m, slotbuf, b);
	}
	bad += m->done_id != id;
	bad += m->done_rows != nrows;
	bad += m->done_tick != 77;
	for (c = 0; c < nrows * MESH_XON_RESPWIDTH; c++)
		if (m->resp[c] != (float)c)
		{
			bad++;
			break;
		}
	b = mkresp(slotbuf, id - 1, 5, nrows, 0, chunks, m->resprows, m->resprows);
	bad += meshx_slot(m, slotbuf, b) != 0;
	b = mkresp(slotbuf, id + 1, 9, nrows, 0, chunks, m->resprows, m->resprows);
	meshx_slot(m, slotbuf, b);
	bad += m->done_id != id;
	slotbuf[0] = 0;
	bad += meshx_slot(m, slotbuf, b) != 0;
	printf("reassembly %s: %u chunks out of order, duplicate chunks ignored, stale req dropped, incomplete not rendered, bad magic rejected, dropped %u\n",
		bad ? "FAIL" : "ok", chunks, m->dropped);
	return bad;
}

int main(int argc, char **argv)
{
	int node = argc > 1 ? atoi(argv[1]) : 1;
	uint32_t width = argc > 2 ? (uint32_t)atoi(argv[2]) : 16;
	uint32_t nrows = argc > 3 ? (uint32_t)atoi(argv[3]) : 480;
	int ticks = argc > 4 ? atoi(argv[4]) : 8;
	int passes = argc > 5 ? atoi(argv[5]) : 200;
	meshxhandle_t *m;
	uint32_t last = 0;
	int h, t, p;

	h = meshx_open(node, width, nrows);
	if (h < 0)
	{
		printf("mesh_open failed: bridge down or arena unavailable\n");
		return 0;
	}
	m = meshx_get(h);
	printf("attached handle %d node %d arena %zu slots stride %zu usable %zu\n", h, node, meshx_nslots, meshx_stride, meshx_usable);
	printf("width %u maxrows %u reqrows/slot %u resprows/slot %u chunks %u\n",
		m->width, m->maxrows, m->reqrows, m->resprows, (m->maxrows + m->reqrows - 1) / m->reqrows);

	for (t = 0; t < ticks; t++)
	{
		fillrows(m, nrows, (uint32_t)t);
		if (!meshx_publish(m, (uint32_t)t, nrows))
			printf("tick %d published nothing\n", t);
		last = meshx_poll(m);
	}
	for (p = 0; p < passes; p++)
	{
		last = meshx_poll(m);
		usleep(1000);
	}

	printf("published %u complete %u inflight %u dropped %u shortwrites %u respchunks %u lastcompletetick %u\n",
		(unsigned)meshx_stat(m, 0), (unsigned)meshx_stat(m, 1), (unsigned)meshx_stat(m, 2),
		(unsigned)meshx_stat(m, 6), (unsigned)meshx_stat(m, 9), (unsigned)meshx_stat(m, 11), (unsigned)meshx_stat(m, 12));
	printf("last complete req_id %u rows %u attached %u peer %u\n",
		last, (unsigned)meshx_stat(m, 10), (unsigned)meshx_stat(m, 5), (unsigned)meshx_stat(m, 15));
	return reassembly(m, nrows) ? 1 : 0;
}
