#include "../../rdma/mesh.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/un.h>
#include <unistd.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#include "../../rdma/mesh-client.c"
#pragma GCC diagnostic pop

enum
{
#define MESH_WIRE(name, value) MESH_XON_##name = value,
#include "../../rdma/xonwire.def"
#undef MESH_WIRE
};

typedef struct meshxwire_s
{
	uint32_t magic;
	uint16_t version;
	uint16_t kind;
	uint64_t offset;
	uint64_t values_total;
	uint32_t req_id;
	uint32_t tick;
	uint32_t width;
	uint32_t values;
	uint32_t flags;
	uint32_t reserved;
}
meshxwire_t;

typedef struct meshxrelay_s
{
	int32_t node;
	uint32_t framebytes;
	uint32_t count;
}
meshxrelay_t;

	_Static_assert(sizeof(meshxwire_t) == MESH_XON_HDRBYTES, "mesh wire header is 48 bytes");

typedef struct meshxhandle_s
{
	int used;
	int node;
	uint16_t request_kind;
	uint16_t response_kind;
	uint32_t width;
	uint32_t maxrows;
	uint32_t reqrows;
	uint32_t resprows;
	float *req;
	float *resp;
	uint32_t req_id;
	uint32_t inflight_id;
	size_t inflight_frame_mass;
	size_t inflight_received_frame_mass;
	uint64_t inflight_values;
	uint64_t *receipts;
	size_t receipt_word_mass;
	uint32_t done_id;
	uint32_t done_rows;
	uint32_t done_tick;
	uint32_t dropped;
	uint32_t shortwrites;
	uint32_t response_frame_mass;
	uint64_t scatter_barriers;
	uint64_t scatter_rows;
}
meshxhandle_t;

static meshxhandle_t *meshx_h;
static size_t meshx_handles;
static size_t meshx_handle_capacity;
static unsigned char *meshx_arena;
static size_t meshx_nslots, meshx_stride, meshx_usable;
static int meshx_relay_fd = -1;
static struct sockaddr_un meshx_relay_worker;
static int meshx_relay_send_error;

static void meshx_relay_pump(void);

static void meshx_relay_init(void)
{
	struct sockaddr_un local;
	const char *worker_path;
	char path[sizeof(local.sun_path)];
	int socket_buffer = 8 * 1024 * 1024;

	if (meshx_relay_fd >= 0)
		return;
	meshx_relay_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
	if (meshx_relay_fd < 0)
		return;
	setsockopt(meshx_relay_fd, SOL_SOCKET, SO_SNDBUF, &socket_buffer, sizeof(socket_buffer));
	setsockopt(meshx_relay_fd, SOL_SOCKET, SO_RCVBUF, &socket_buffer, sizeof(socket_buffer));
	fcntl(meshx_relay_fd, F_SETFL, fcntl(meshx_relay_fd, F_GETFL, 0) | O_NONBLOCK);
	memset(&local, 0, sizeof(local));
	local.sun_family = AF_UNIX;
	dpsnprintf(path, sizeof(path), "/tmp/mesh-expert-engine-%d.sock", (int)getpid());
	strlcpy(local.sun_path, path, sizeof(local.sun_path));
	unlink(local.sun_path);
	if (bind(meshx_relay_fd, (struct sockaddr *)&local, sizeof(local)) < 0)
	{
		close(meshx_relay_fd);
		meshx_relay_fd = -1;
		return;
	}
	memset(&meshx_relay_worker, 0, sizeof(meshx_relay_worker));
	meshx_relay_worker.sun_family = AF_UNIX;
	worker_path = getenv("MESH_EXPERT_SOCKET");
	strlcpy(meshx_relay_worker.sun_path,
		worker_path && worker_path[0] ? worker_path : "/tmp/mesh-expert-worker.sock",
		sizeof(meshx_relay_worker.sun_path));
}

static int meshx_relay_request(const unsigned char *q, size_t b, int from)
{
	meshxwire_t w;
	meshxrelay_t relay = {from, (uint32_t)b, 1};
	struct iovec iov[2] = {{&relay, sizeof(relay)}, {(void *)q, b}};
	struct msghdr message;
	ssize_t sent;

	if (b < MESH_XON_HDRBYTES || b > meshx_usable)
		return 0;
	memcpy(&w, q, MESH_XON_HDRBYTES);
	if (w.magic != MESH_XON_MAGIC || w.version != MESH_XON_VERSION ||
		(w.kind != MESH_XON_EXPERT_REQ && w.kind != MESH_XON_EXPERT_TRAIN_REQ &&
		 w.kind != MESH_XON_EXPERT_GRAD_REQ && w.kind != MESH_XON_EXPERT_BATCH_BEGIN &&
		 w.kind != MESH_XON_EXPERT_BATCH_COMMIT))
		return 0;
	meshx_relay_init();
	if (meshx_relay_fd >= 0)
	{
		memset(&message, 0, sizeof(message));
		message.msg_name = &meshx_relay_worker;
		message.msg_namelen = sizeof(meshx_relay_worker);
		message.msg_iov = iov;
		message.msg_iovlen = 2;
		do
		{
			sent = sendmsg(meshx_relay_fd, &message, 0);
			if (sent < 0 && (errno == ENOBUFS || errno == EAGAIN))
			{
				meshx_relay_pump();
				Sys_Sleep(50);
			}
		}
		while (sent < 0 && (errno == ENOBUFS || errno == EAGAIN));
		if (sent != (ssize_t)(sizeof(relay) + b))
		{
			int error = sent < 0 ? errno : EIO;
			if (error != meshx_relay_send_error)
				Con_Printf("mesh expert relay unavailable: sendmsg %s failed: %s\n", meshx_relay_worker.sun_path, strerror(error));
			meshx_relay_send_error = error;
		}
		else if (meshx_relay_send_error)
			Con_Printf("mesh expert relay restored: %s\n", meshx_relay_worker.sun_path);
		if (sent == (ssize_t)(sizeof(relay) + b))
			meshx_relay_send_error = 0;
	}
	else if (meshx_relay_send_error != ENOTCONN)
	{
		meshx_relay_send_error = ENOTCONN;
		Con_Printf("mesh expert relay unavailable: local socket initialization failed\n");
	}
	return 1;
}

static void meshx_relay_pump(void)
{
	static unsigned char *frame;
	static size_t capacity;
	meshxwire_t w;
	meshxrelay_t relay;
	struct iovec iov[2];
	struct msghdr message;
	ssize_t n;
	int available;

	meshx_relay_init();
	if (meshx_relay_fd < 0 || !meshx_arena)
		return;
	mesh_pump();
	while (!ioctl(meshx_relay_fd, FIONREAD, &available) && available > (int)sizeof(meshxrelay_t))
	{
		size_t b;

		if (capacity < (size_t)available - sizeof(relay))
		{
			unsigned char *next = (unsigned char *)realloc(frame, (size_t)available - sizeof(relay));
			if (!next)
				Sys_Error("mesh relay allocation failed");
			frame = next;
			capacity = (size_t)available - sizeof(relay);
		}
		iov[0].iov_base = &relay;
		iov[0].iov_len = sizeof(relay);
		iov[1].iov_base = frame;
		iov[1].iov_len = capacity;
		memset(&message, 0, sizeof(message));
		message.msg_iov = iov;
		message.msg_iovlen = 2;
		n = recvmsg(meshx_relay_fd, &message, MSG_DONTWAIT);
		if (n <= (ssize_t)sizeof(relay))
			continue;
		b = (size_t)n - sizeof(relay);
		if (!relay.framebytes || !relay.count || relay.framebytes > meshx_usable ||
			(uint64_t)relay.framebytes * relay.count != b)
			continue;
		for (uint32_t i = 0; i < relay.count; i++)
		{
			unsigned char *q = frame + (size_t)i * relay.framebytes;
			memcpy(&w, q, MESH_XON_HDRBYTES);
			if (w.magic != MESH_XON_MAGIC || w.version != MESH_XON_VERSION ||
				(w.kind != MESH_XON_EXPERT_RESP && w.kind != MESH_XON_EXPERT_META &&
				 w.kind != MESH_XON_EXPERT_GRAD_RESP && w.kind != MESH_XON_EXPERT_GRAD_META &&
				 w.kind != MESH_XON_EXPERT_BATCH_RESP) ||
				!w.values || !w.values_total ||
				w.offset >= w.values_total || w.values > w.values_total - w.offset ||
				relay.framebytes < MESH_XON_HDRBYTES + (size_t)w.values * sizeof(float))
				break;
			if (i + 1 == relay.count)
				mesh_queue_copy(frame, relay.framebytes, relay.framebytes, relay.count, relay.node);
		}
	}
	mesh_pump();
}

static int meshx_attach(void)
{
	if (meshx_arena)
		return 0;
	meshx_arena = (unsigned char *)mesh_open(&meshx_nslots, &meshx_stride, &meshx_usable);
	if (meshx_arena)
		Con_Printf("mesh transport attached: region %s slots %zu stride %zu usable %zu shared-credit scheduling\n",
			rname(NULL), meshx_nslots, meshx_stride, meshx_usable);
	meshx_relay_init();
	return meshx_arena ? 0 : -1;
}

static meshxhandle_t *meshx_get(int h)
{
	if (h < 0 || (size_t)h >= meshx_handles || !meshx_h[h].used)
		return NULL;
	return &meshx_h[h];
}

static int meshx_reserve(meshxhandle_t *m, uint32_t rows)
{
	uint32_t values_per_page;
	uint64_t values_total;
	size_t values, old_values, frame_mass, receipt_word_mass;
	float *req, *resp;
	uint64_t *receipts;

	if (rows <= m->maxrows)
		return 1;
	values_per_page = meshx_usable > MESH_XON_HDRBYTES
		? ((uint32_t)meshx_usable - MESH_XON_HDRBYTES) / sizeof(float) : 0;
	values_total = (uint64_t)rows * m->width;
	if (!values_per_page || values_total > (uint64_t)(SIZE_MAX / sizeof(float)))
	{
		Con_Printf("mesh transport capacity unavailable: width %u rows %u\n", m->width, rows);
		return 0;
	}
	values = (size_t)values_total;
	frame_mass = 1 + (values - 1) / values_per_page;
	receipt_word_mass = 1 + (frame_mass - 1) / 64;
	req = (float *)calloc(values, sizeof(*req));
	resp = (float *)calloc(values, sizeof(*resp));
	receipts = (uint64_t *)calloc(receipt_word_mass, sizeof(*receipts));
	if (!req || !resp || !receipts)
	{
		free(req);
		free(resp);
		free(receipts);
		Con_Printf("mesh transport capacity allocation failed: width %u rows %u\n", m->width, rows);
		return 0;
	}
	old_values = (size_t)m->maxrows * m->width;
	if (old_values)
	{
		memcpy(req, m->req, old_values * sizeof(*req));
		memcpy(resp, m->resp, old_values * sizeof(*resp));
	}
	if (m->receipt_word_mass)
		memcpy(receipts, m->receipts, m->receipt_word_mass * sizeof(*receipts));
	free(m->req);
	free(m->resp);
	free(m->receipts);
	m->req = req;
	m->resp = resp;
	m->receipts = receipts;
	m->receipt_word_mass = receipt_word_mass;
	m->maxrows = rows;
	return 1;
}

static int meshx_open(int node, uint16_t request_kind, uint16_t response_kind, uint32_t width, uint32_t maxrows)
{
	uint32_t values_per_page;
	uint64_t values_total;
	size_t frame_mass;
	meshxhandle_t *m;
	size_t h;

	if (width < 1 || maxrows < 1)
		return -1;
	if (meshx_attach())
		return -1;

	values_per_page = meshx_usable > MESH_XON_HDRBYTES
		? ((uint32_t)meshx_usable - MESH_XON_HDRBYTES) / sizeof(float) : 0;
	if (!values_per_page)
		return -1;

	for (h = 0; h < meshx_handles; h++)
		if (meshx_h[h].used && meshx_h[h].node == node &&
			meshx_h[h].request_kind == request_kind && meshx_h[h].response_kind == response_kind &&
			meshx_h[h].width == width)
			return meshx_reserve(&meshx_h[h], maxrows) ? (int)h : -1;
	h = meshx_handles;
	if (meshx_handles == meshx_handle_capacity)
	{
		size_t capacity = meshx_handle_capacity ? meshx_handle_capacity * 2 : 1;
		meshxhandle_t *handles = (meshxhandle_t *)realloc(meshx_h, capacity * sizeof(*handles));
		if (!handles)
			return -1;
		memset(handles + meshx_handle_capacity, 0,
			(capacity - meshx_handle_capacity) * sizeof(*handles));
		meshx_h = handles;
		meshx_handle_capacity = capacity;
	}
	meshx_handles++;

	m = &meshx_h[h];
	memset(m, 0, sizeof(*m));
	m->node = node;
	m->request_kind = request_kind;
	m->response_kind = response_kind;
	m->width = width;
	if (!meshx_reserve(m, maxrows))
	{
		memset(m, 0, sizeof(*m));
		meshx_handles--;
		return -1;
	}
	m->used = 1;
	m->reqrows = values_per_page / width;
	m->resprows = values_per_page / width;
	values_total = (uint64_t)m->maxrows * width;
	frame_mass = (size_t)(1 + (values_total - 1) / values_per_page);
	Con_Printf("mesh transport handle %zu open: node %d width %u rows %u pages %zu\n",
		h, node, width, maxrows, frame_mass);
	return (int)h;
}

static uint32_t meshx_publish(meshxhandle_t *m, uint32_t tick, uint32_t nrows)
{
	uint32_t values_per_page;
	uint64_t values_total;
	size_t frame_mass;
	unsigned char *frames;

	if (nrows > m->maxrows && !meshx_reserve(m, nrows))
		return 0;
	if (!m->request_kind)
		return 0;
	values_per_page = ((uint32_t)meshx_usable - MESH_XON_HDRBYTES) / sizeof(float);
	values_total = (uint64_t)nrows * m->width;
	frame_mass = values_total ? (size_t)(1 + (values_total - 1) / values_per_page) : 1;
	if (frame_mass > SIZE_MAX / meshx_usable)
		return 0;
	frames = (unsigned char *)calloc(frame_mass, meshx_usable);
	if (!frames)
		return 0;
	m->req_id++;
	for (size_t c = 0; c < frame_mass; c++)
	{
		unsigned char *frame = frames + c * meshx_usable;
		meshxwire_t w;
		uint64_t offset = (uint64_t)c * values_per_page;
		uint32_t values = values_total ? (uint32_t)(values_total - offset < values_per_page
			? values_total - offset : values_per_page) : 0;

		memset(&w, 0, sizeof(w));
		w.magic = MESH_XON_MAGIC;
		w.version = MESH_XON_VERSION;
		w.kind = m->request_kind;
		w.offset = offset;
		w.values_total = values_total;
		w.req_id = m->req_id;
		w.tick = tick;
		w.width = m->width;
		w.values = values;
		memcpy(frame, &w, MESH_XON_HDRBYTES);
		if (values)
			memcpy(frame + MESH_XON_HDRBYTES, m->req + offset,
				(size_t)values * sizeof(float));
	}
	mesh_queue_copy(frames, meshx_usable, meshx_usable, frame_mass, m->node);
	free(frames);
	mesh_pump();
	return m->req_id;
}

static int meshx_slot(meshxhandle_t *m, const unsigned char *q, size_t b, int from)
{
	meshxwire_t w;
	uint32_t values_per_page;
	uint64_t rows;
	size_t frame_index;
	size_t frame_mass;
	uint64_t receipt_bit;

	if (b < (size_t)MESH_XON_HDRBYTES)
		return 0;
	memcpy(&w, q, MESH_XON_HDRBYTES);
	if (w.magic != MESH_XON_MAGIC || w.version != MESH_XON_VERSION ||
		!m->response_kind || w.kind != m->response_kind || from != m->node)
		return 0;
	if (w.width != m->width || !w.values || !w.values_total ||
		w.values_total % w.width || w.offset >= w.values_total ||
		w.values > w.values_total - w.offset)
		return 0;
	if ((size_t)w.values * sizeof(float) > b - MESH_XON_HDRBYTES)
		return 0;
	rows = w.values_total / w.width;
	if (rows > UINT32_MAX || (rows > m->maxrows && !meshx_reserve(m, (uint32_t)rows)))
		return 0;
	values_per_page = ((uint32_t)meshx_usable - MESH_XON_HDRBYTES) / sizeof(float);
	if (w.offset % values_per_page)
		return 0;
	frame_index = (size_t)(w.offset / values_per_page);
	frame_mass = (size_t)(1 + (w.values_total - 1) / values_per_page);
	if (w.values != (uint32_t)(w.values_total - w.offset < values_per_page
		? w.values_total - w.offset : values_per_page) || frame_index >= frame_mass ||
		1 + (frame_mass - 1) / 64 > m->receipt_word_mass)
		return 0;

	if (w.req_id != m->inflight_id)
	{
		if (m->inflight_id && m->inflight_received_frame_mass != m->inflight_frame_mass && w.offset)
		{
			m->dropped++;
			return 0;
		}
		if (m->inflight_id && m->inflight_received_frame_mass != m->inflight_frame_mass)
			m->dropped++;
		m->inflight_id = w.req_id;
		m->inflight_frame_mass = frame_mass;
		m->inflight_received_frame_mass = 0;
		m->inflight_values = w.values_total;
		memset(m->receipts, 0, m->receipt_word_mass * sizeof(*m->receipts));
	}
	else if (frame_mass != m->inflight_frame_mass || w.values_total != m->inflight_values)
		return 0;

	receipt_bit = (uint64_t)1 << (frame_index & 63);
	if (!(m->receipts[frame_index >> 6] & receipt_bit))
	{
		m->receipts[frame_index >> 6] |= receipt_bit;
		m->inflight_received_frame_mass++;
	}
	m->response_frame_mass++;
	memcpy(m->resp + w.offset, q + MESH_XON_HDRBYTES,
		(size_t)w.values * sizeof(float));

	if (m->inflight_received_frame_mass == m->inflight_frame_mass)
	{
		m->done_id = w.req_id;
		m->done_rows = (uint32_t)rows;
		m->done_tick = w.tick;
	}
	return 1;
}

static void meshx_dispatch(void)
{
	size_t h;
	if (!meshx_arena)
		return;
	mesh_pump();
	meshx_relay_pump();
	for (;;)
	{
		void *q = NULL;
		int from = 0;
		size_t b = mesh_read(&q, &from);

		if (!b)
			break;
		if (meshx_relay_request((const unsigned char *)q, b, from))
			continue;
		for (h = 0; h < meshx_handles; h++)
			if (meshx_h[h].used && meshx_slot(&meshx_h[h], (const unsigned char *)q, b, from))
				break;
	}
	mesh_pump();
}

void MeshX_Pump(void)
{
	meshx_dispatch();
}

static uint32_t meshx_poll(meshxhandle_t *m)
{
	meshx_dispatch();
	return m->done_id;
}

static double meshx_stat(meshxhandle_t *m, int sel)
{
	switch (sel)
	{
	case 0: return m->req_id;
	case 1: return m->done_id;
	case 2: return (double)m->req_id - (double)m->done_id;
	case 3: return m->width;
	case 4: return m->maxrows;
	case 5: return meshx_arena ? 1 : 0;
	case 6: return m->dropped;
	case 7: return m->reqrows;
	case 8: return (double)meshx_nslots;
	case 9: return m->shortwrites;
	case 10: return m->done_rows;
	case 11: return m->response_frame_mass;
	case 12: return m->done_tick;
	case 13: return (double)meshx_nslots;
	case 14: return (double)meshx_usable;
	case 15: return m->node;
	case 16: return (double)m->scatter_barriers;
	case 17: return (double)m->scatter_rows;
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
	if (col >= width)
	{
		VM_Warning(prog, "%s: column %u of %u out of range in %s\n", who, col, width, prog->name);
		return 0;
	}
	if (*n > m->maxrows && !meshx_reserve(m, *n))
	{
		VM_Warning(prog, "%s: rows %u unavailable in %s\n", who, *n, prog->name);
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
	VM_SAFEPARMCOUNT(5, VM_mesh_open);
	PRVM_G_FLOAT(OFS_RETURN) = meshx_open(
		(int)PRVM_G_FLOAT(OFS_PARM0),
		(uint16_t)PRVM_G_FLOAT(OFS_PARM1),
		(uint16_t)PRVM_G_FLOAT(OFS_PARM2),
		(uint32_t)PRVM_G_FLOAT(OFS_PARM3),
		(uint32_t)PRVM_G_FLOAT(OFS_PARM4));
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
	if (!VM_mesh_span(prog, "mesh_scatter", m, m->width, col, &fld, &first, &n))
		return;
	if (n > m->done_rows)
		n = m->done_rows;
	stride = (size_t)prog->entityfields;
	for (row = 0; row < n; row++)
		prog->edictsfields[(size_t)(first + row) * stride + fld] = (prvm_vec_t)m->resp[(size_t)row * m->width + col];
}

static int VM_mesh_rows(prvm_prog_t *prog, const char *who, meshxhandle_t *m, int *fld, uint32_t *first, uint32_t *n, uint32_t *cols)
{
	*fld = PRVM_G_INT(OFS_PARM1);
	*first = (uint32_t)PRVM_G_FLOAT(OFS_PARM2);
	*n = (uint32_t)PRVM_G_FLOAT(OFS_PARM3);
	*cols = (uint32_t)PRVM_G_FLOAT(OFS_PARM4);
	if (*n > m->maxrows && !meshx_reserve(m, *n))
	{
		VM_Warning(prog, "%s: rows %u unavailable in %s\n", who, *n, prog->name);
		return 0;
	}
	if (!*cols || *cols > m->width || *fld < 0 || (uint32_t)*fld + *cols > (uint32_t)prog->entityfields || *first > (uint32_t)prog->max_edicts || *n > (uint32_t)prog->max_edicts - *first)
	{
		VM_Warning(prog, "%s: fields %i+%u edicts %u+%u width %u rows %u out of range in %s\n", who, *fld, *cols, *first, *n, m->width, m->maxrows, prog->name);
		return 0;
	}
	return 1;
}

void VM_mesh_gather_rows(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	uint32_t first, n, cols, row, col;
	size_t stride;
	int fld;

	VM_SAFEPARMCOUNT(5, VM_mesh_gather_rows);
	m = VM_mesh_resolve(prog, "mesh_gather_rows");
	if (!m || !VM_mesh_rows(prog, "mesh_gather_rows", m, &fld, &first, &n, &cols))
		return;
	stride = (size_t)prog->entityfields;
	if (sizeof(prvm_vec_t) == sizeof(float))
	{
		for (row = 0; row < n; row++)
			memcpy(m->req + (size_t)row * m->width, prog->edictsfields + (size_t)(first + row) * stride + fld, (size_t)cols * sizeof(float));
		return;
	}
	for (row = 0; row < n; row++)
		for (col = 0; col < cols; col++)
			m->req[(size_t)row * m->width + col] = (float)prog->edictsfields[(size_t)(first + row) * stride + fld + col];
}

void VM_mesh_gather_list(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	prvm_edict_t *row;
	uint32_t n, cols, gathered = 0;
	int fld, nextfld;

	VM_SAFEPARMCOUNT(6, VM_mesh_gather_list);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_gather_list");
	if (!m)
		return;
	fld = PRVM_G_INT(OFS_PARM1);
	row = PRVM_G_EDICT(OFS_PARM2);
	nextfld = PRVM_G_INT(OFS_PARM3);
	n = (uint32_t)PRVM_G_FLOAT(OFS_PARM4);
	cols = (uint32_t)PRVM_G_FLOAT(OFS_PARM5);
	if (n > m->maxrows && !meshx_reserve(m, n))
	{
		VM_Warning(prog, "mesh_gather_list: rows %u unavailable in %s\n", n, prog->name);
		return;
	}
	if (!cols || cols > m->width || fld < 0 || (uint32_t)fld + cols > (uint32_t)prog->entityfields ||
		nextfld < 0 || nextfld >= prog->entityfields)
	{
		VM_Warning(prog, "mesh_gather_list: fields %i+%u next %i width %u rows %u out of range in %s\n", fld, cols, nextfld, m->width, m->maxrows, prog->name);
		return;
	}
	while (gathered < n && row != prog->edicts)
	{
		for (uint32_t col = 0; col < cols; col++)
			m->req[(size_t)gathered * m->width + col] = (float)row->fields.fp[fld + col];
		row = PRVM_PROG_TO_EDICT(PRVM_EDICTFIELDEDICT(row, nextfld));
		gathered++;
	}
	if (gathered != n)
		VM_Warning(prog, "mesh_gather_list: extent ended at %u of %u rows in %s\n", gathered, n, prog->name);
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)gathered;
}

void VM_mesh_scatter_rows(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	uint32_t first, n, cols, row, col;
	size_t stride;
	int fld;

	VM_SAFEPARMCOUNT(5, VM_mesh_scatter_rows);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	m = VM_mesh_resolve(prog, "mesh_scatter_rows");
	if (!m || !VM_mesh_rows(prog, "mesh_scatter_rows", m, &fld, &first, &n, &cols))
		return;
	if (n != m->done_rows)
	{
		VM_Warning(prog, "mesh_scatter_rows: response has %u rows for %u destinations in %s\n", m->done_rows, n, prog->name);
		return;
	}
	stride = (size_t)prog->entityfields;
	if (sizeof(prvm_vec_t) == sizeof(float))
	{
		for (row = 0; row < n; row++)
			memcpy(prog->edictsfields + (size_t)(first + row) * stride + fld, m->resp + (size_t)row * m->width, (size_t)cols * sizeof(float));
	}
	else
	{
		for (row = 0; row < n; row++)
			for (col = 0; col < cols; col++)
				prog->edictsfields[(size_t)(first + row) * stride + fld + col] = (prvm_vec_t)m->resp[(size_t)row * m->width + col];
	}
	m->scatter_barriers++;
	m->scatter_rows += n;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)n;
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
