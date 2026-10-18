#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/un.h>
#include <unistd.h>

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
	float *incoming;
	uint32_t req_id;
	uint32_t inflight_id;
	uint64_t session;
	uint64_t inflight_session;
	uint32_t inflight_tick;
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
typedef struct meshxpacket_s
{
	struct meshxpacket_s *next;
	unsigned char *frames;
	size_t capacity, count, first, framebytes;
	int node;
}
meshxpacket_t;

static meshxpacket_t *meshx_pending, *meshx_pending_tail, *meshx_spare;
static size_t meshx_pending_frames;
static size_t meshx_local_slots = 64;
static size_t meshx_usable;
static int meshx_relay_fd = -1;
static struct sockaddr_un meshx_relay_worker;
static int meshx_relay_send_error;

static int meshx_slot(meshxhandle_t *m, const unsigned char *q, size_t b, int from);

int MeshX_Descriptor(void)
{
	return meshx_relay_fd;
}

static int meshx_relay_init(void)
{
	struct sockaddr_un local;
	const char *worker_path = getenv("MESH_EXPERT_SOCKET"), *framebytes = getenv("MESH_EXPERT_FRAMEBYTES");
	int socket_buffer = 8 * 1024 * 1024;

	if (meshx_relay_fd >= 0)
		return 0;
	if (!worker_path || !worker_path[0] || !framebytes || strtoul(framebytes, NULL, 10) <= MESH_XON_HDRBYTES)
	{
		Con_Printf("mesh runtime needs MESH_EXPERT_SOCKET and MESH_EXPERT_FRAMEBYTES (more than %d bytes)\n", MESH_XON_HDRBYTES);
		return -1;
	}
	meshx_usable = strtoul(framebytes, NULL, 10);
	meshx_relay_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
	if (meshx_relay_fd < 0)
	{
		Con_Printf("mesh runtime socket failed: %s\n", strerror(errno));
		return -1;
	}
	setsockopt(meshx_relay_fd, SOL_SOCKET, SO_SNDBUF, &socket_buffer, sizeof(socket_buffer));
	setsockopt(meshx_relay_fd, SOL_SOCKET, SO_RCVBUF, &socket_buffer, sizeof(socket_buffer));
	fcntl(meshx_relay_fd, F_SETFL, fcntl(meshx_relay_fd, F_GETFL, 0) | O_NONBLOCK);
	memset(&local, 0, sizeof(local));
	local.sun_family = AF_UNIX;
	dpsnprintf(local.sun_path, sizeof(local.sun_path), "/tmp/mesh-engine-%d.sock", (int)getpid());
	unlink(local.sun_path);
	if (bind(meshx_relay_fd, (struct sockaddr *)&local, sizeof(local)) < 0)
	{
		Con_Printf("mesh runtime bind failed: %s\n", strerror(errno));
		close(meshx_relay_fd);
		meshx_relay_fd = -1;
		return -1;
	}
	memset(&meshx_relay_worker, 0, sizeof(meshx_relay_worker));
	meshx_relay_worker.sun_family = AF_UNIX;
	strlcpy(meshx_relay_worker.sun_path, worker_path, sizeof(meshx_relay_worker.sun_path));
	Con_Printf("mesh runtime %s: frames of %zu bytes\n", meshx_relay_worker.sun_path, meshx_usable);
	return 0;
}

static void meshx_reframe(meshxpacket_t *packet)
{
	meshxwire_t header;
	memcpy(&header, packet->frames, sizeof(header));
	size_t previous = (packet->framebytes - MESH_XON_HDRBYTES) / sizeof(float), values = max(1, previous / 2);
	size_t count = max(1, (header.values_total + values - 1) / values);
	size_t bytes = MESH_XON_HDRBYTES + values * sizeof(float);
	unsigned char *frames = calloc(count, bytes);
	if (!frames)
		Sys_Error("mesh local reframing allocation failed");
	for (size_t i = 0; i < count; ++i)
	{
		header.offset = i * values;
		header.values = min(values, header.values_total - header.offset);
		memcpy(frames + i * bytes, &header, sizeof(header));
		for (size_t copied = 0; copied < header.values;)
		{
			size_t offset = header.offset + copied, part = min(header.values - copied, previous - offset % previous);
			memcpy(frames + i * bytes + MESH_XON_HDRBYTES + copied * sizeof(float),
				packet->frames + offset / previous * packet->framebytes + MESH_XON_HDRBYTES + offset % previous * sizeof(float), part * sizeof(float));
			copied += part;
		}
	}
	meshx_pending_frames = meshx_pending_frames - (packet->count - packet->first) + count;
	free(packet->frames);
	packet->frames = frames;
	packet->count = count;
	packet->first = 0;
	packet->capacity = count * bytes;
	packet->framebytes = bytes;
	Con_Printf("mesh local packet reframed into %zu frames of %zu bytes\n", count, bytes);
}

static int meshx_local_send(const meshxrelay_t *relay, const void *data, size_t bytes)
{
	struct iovec iov[2] = {{(void *)relay, sizeof(*relay)}, {(void *)data, bytes}};
	struct msghdr message;
	ssize_t sent;
	int error;

	memset(&message, 0, sizeof(message));
	message.msg_name = &meshx_relay_worker;
	message.msg_namelen = sizeof(meshx_relay_worker);
	message.msg_iov = iov;
	message.msg_iovlen = bytes ? 2 : 1;
	sent = sendmsg(meshx_relay_fd, &message, MSG_DONTWAIT);
	error = sent == (ssize_t)(sizeof(*relay) + bytes) ? 0 : sent < 0 ? errno : EIO;
	if (error != meshx_relay_send_error)
		Con_Printf("mesh runtime %s: %s\n", meshx_relay_worker.sun_path, error ? strerror(error) : "restored");
	meshx_relay_send_error = error;
	return !error;
}

static void meshx_pending_release(void)
{
	meshxpacket_t *packet = meshx_pending;
	meshx_pending = packet->next;
	if (!meshx_pending)
		meshx_pending_tail = NULL;
	packet->next = meshx_spare;
	meshx_spare = packet;
}

static void meshx_relay_pump(void)
{
	static unsigned char *frame;
	static size_t capacity;
	meshxrelay_t relay;
	struct iovec iov[2];
	struct msghdr message;
	int available;

	if (meshx_relay_fd < 0)
		return;
	while (!ioctl(meshx_relay_fd, FIONREAD, &available) && available > 0)
	{
		ssize_t n;
		size_t b;

		if (capacity < (size_t)available)
		{
			unsigned char *next = (unsigned char *)realloc(frame, (size_t)available);
			if (!next)
				Sys_Error("mesh local receive allocation failed");
			frame = next;
			capacity = (size_t)available;
		}
		iov[0].iov_base = &relay;
		iov[0].iov_len = sizeof(relay);
		iov[1].iov_base = frame;
		iov[1].iov_len = capacity;
		memset(&message, 0, sizeof(message));
		message.msg_iov = iov;
		message.msg_iovlen = 2;
		n = recvmsg(meshx_relay_fd, &message, MSG_DONTWAIT);
		if (n < (ssize_t)sizeof(relay))
		{
			Con_Printf("mesh local receive incomplete: %zd bytes\n", n);
			break;
		}
		b = (size_t)n - sizeof(relay);
		if (!relay.framebytes || !relay.count || (uint64_t)relay.framebytes * relay.count != b)
		{
			Con_Printf("mesh local framing mismatch: %zu bytes, %u frames of %u\n", b, relay.count, relay.framebytes);
			continue;
		}
		for (uint32_t i = 0; i < relay.count; i++)
			for (size_t h = 0; h < meshx_handles; h++)
				if (meshx_h[h].used && meshx_slot(&meshx_h[h], frame + (size_t)i * relay.framebytes, relay.framebytes, relay.node))
					break;
	}
	while (meshx_pending)
	{
		meshxpacket_t *packet = meshx_pending;
		size_t bytes = packet->framebytes;
		size_t count = packet->count - packet->first < meshx_local_slots ? packet->count - packet->first : meshx_local_slots;
		meshxrelay_t data = {packet->node, (uint32_t)bytes, (uint32_t)count};
		if (!meshx_local_send(&data, packet->frames + packet->first * bytes, bytes * count))
		{
			if (meshx_relay_send_error == EMSGSIZE)
			{
				if (count > 1)
					meshx_local_slots = (count + 1) / 2;
				else
					meshx_reframe(packet);
				continue;
			}
			if (meshx_relay_send_error == ENOENT || meshx_relay_send_error == ECONNREFUSED)
				while (meshx_pending)
				{
					meshx_pending_frames -= meshx_pending->count - meshx_pending->first;
					meshx_pending_release();
				}
			break;
		}
		packet->first += count;
		meshx_pending_frames -= count;
		if (packet->first == packet->count)
			meshx_pending_release();
	}
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
	float *req, *resp, *incoming;
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
	if (receipt_word_mass < m->receipt_word_mass)
		receipt_word_mass = m->receipt_word_mass;
	req = (float *)calloc(values, sizeof(*req));
	resp = (float *)calloc(values, sizeof(*resp));
	incoming = (float *)calloc(values, sizeof(*incoming));
	receipts = (uint64_t *)calloc(receipt_word_mass, sizeof(*receipts));
	if (!req || !resp || !incoming || !receipts)
	{
		free(req);
		free(resp);
		free(incoming);
		free(receipts);
		Con_Printf("mesh transport capacity allocation failed: width %u rows %u\n", m->width, rows);
		return 0;
	}
	old_values = (size_t)m->maxrows * m->width;
	if (old_values)
	{
		memcpy(req, m->req, old_values * sizeof(*req));
		memcpy(resp, m->resp, old_values * sizeof(*resp));
		memcpy(incoming, m->incoming, old_values * sizeof(*incoming));
	}
	if (m->receipt_word_mass)
		memcpy(receipts, m->receipts, m->receipt_word_mass * sizeof(*receipts));
	free(m->req);
	free(m->resp);
	free(m->incoming);
	free(m->receipts);
	m->req = req;
	m->resp = resp;
	m->incoming = incoming;
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
	if (meshx_relay_init())
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
	meshxpacket_t *packet;

	if (nrows > m->maxrows && !meshx_reserve(m, nrows))
		return 0;
	if (!m->request_kind)
		return 0;
	values_per_page = ((uint32_t)meshx_usable - MESH_XON_HDRBYTES) / sizeof(float);
	values_total = (uint64_t)nrows * m->width;
	frame_mass = values_total ? (size_t)(1 + (values_total - 1) / values_per_page) : 1;
	if (frame_mass > SIZE_MAX / meshx_usable)
		return 0;
	packet = meshx_spare;
	if (packet)
		meshx_spare = packet->next;
	else
		packet = (meshxpacket_t *)calloc(1, sizeof(*packet));
	if (!packet)
		Sys_Error("mesh local publication allocation failed");
	if (packet->capacity < frame_mass * meshx_usable)
	{
		unsigned char *next = (unsigned char *)realloc(packet->frames, frame_mass * meshx_usable);
		if (!next)
			Sys_Error("mesh local publication storage failed");
		packet->frames = next;
		packet->capacity = frame_mass * meshx_usable;
	}
	packet->count = frame_mass;
	packet->framebytes = meshx_usable;
	packet->first = 0;
	packet->node = m->node;
	packet->next = NULL;
	frames = packet->frames;
	memset(frames, 0, frame_mass * meshx_usable);
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
		w.flags = m->session;
		w.reserved = m->session >> 32;
		memcpy(frame, &w, MESH_XON_HDRBYTES);
		if (values)
			memcpy(frame + MESH_XON_HDRBYTES, m->req + offset,
				(size_t)values * sizeof(float));
	}
	if (meshx_pending_tail)
		meshx_pending_tail->next = packet;
	else
		meshx_pending = packet;
	meshx_pending_tail = packet;
	meshx_pending_frames += frame_mass;
	meshx_relay_pump();
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
	uint64_t session;

	if (b < (size_t)MESH_XON_HDRBYTES)
		return 0;
	memcpy(&w, q, MESH_XON_HDRBYTES);
	session = (uint64_t)w.flags | ((uint64_t)w.reserved << 32);
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
	values_per_page = ((uint32_t)b - MESH_XON_HDRBYTES) / sizeof(float);
	if (!values_per_page || w.offset % values_per_page)
		return 0;
	frame_index = (size_t)(w.offset / values_per_page);
	frame_mass = (size_t)(1 + (w.values_total - 1) / values_per_page);
	if (w.values != (uint32_t)(w.values_total - w.offset < values_per_page
		? w.values_total - w.offset : values_per_page) || frame_index >= frame_mass)
		return 0;
	if (1 + (frame_mass - 1) / 64 > m->receipt_word_mass)
	{
		size_t words = 1 + (frame_mass - 1) / 64;
		uint64_t *next = (uint64_t *)realloc(m->receipts, words * sizeof(*next));
		if (!next)
			Sys_Error("mesh response receipt storage failed");
		memset(next + m->receipt_word_mass, 0, (words - m->receipt_word_mass) * sizeof(*next));
		m->receipts = next;
		m->receipt_word_mass = words;
	}

	if (w.req_id != m->inflight_id || w.tick != m->inflight_tick || session != m->inflight_session)
	{
		if (m->inflight_id && m->inflight_received_frame_mass != m->inflight_frame_mass && w.offset)
		{
			m->dropped++;
			return 0;
		}
		if (m->inflight_id && m->inflight_received_frame_mass != m->inflight_frame_mass)
			m->dropped++;
		m->inflight_id = w.req_id;
		m->inflight_tick = w.tick;
		m->inflight_session = session;
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
	memcpy(m->incoming + w.offset, q + MESH_XON_HDRBYTES,
		(size_t)w.values * sizeof(float));

	if (m->inflight_received_frame_mass == m->inflight_frame_mass)
	{
		float *previous = m->resp;
		m->resp = m->incoming;
		m->incoming = previous;
		m->done_id = w.req_id;
		m->done_rows = (uint32_t)rows;
		m->done_tick = w.tick;
	}
	return 1;
}

#ifndef MESH_XON_CORE_ONLY
static void MeshX_ViewConsume(prvm_prog_t *prog);
#endif

void MeshX_Pump(void)
{
	meshx_relay_pump();
#ifndef MESH_XON_CORE_ONLY
	if (SVVM_prog->loaded)
		MeshX_ViewConsume(SVVM_prog);
#endif
}

static uint32_t meshx_poll(meshxhandle_t *m)
{
	meshx_relay_pump();
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
	case 6: return m->dropped;
	case 7: return m->reqrows;
	case 9: return m->shortwrites;
	case 10: return m->done_rows;
	case 11: return m->response_frame_mass;
	case 12: return m->done_tick;
	case 14: return (double)meshx_usable;
	case 15: return m->node;
	case 16: return (double)m->scatter_barriers;
	case 17: return (double)m->scatter_rows;
	case 18: return (double)meshx_pending_frames;
	default: return m->node;
	}
}

#ifndef MESH_XON_CORE_ONLY

static meshxhandle_t *VM_mesh_resolve(prvm_prog_t *prog, const char *who)
{
	meshxhandle_t *m = meshx_get((int)PRVM_G_READFLOAT(OFS_PARM0));
	if (!m)
		VM_Warning(prog, "%s: handle %i not open in %s\n", who, (int)PRVM_G_READFLOAT(OFS_PARM0), prog->name);
	return m;
}

static int VM_mesh_span(prvm_prog_t *prog, const char *who, meshxhandle_t *m, uint32_t width, uint32_t col, int *fld, uint32_t *first, uint32_t *n)
{
	*fld = PRVM_G_READINT(OFS_PARM2);
	*first = (uint32_t)PRVM_G_READFLOAT(OFS_PARM3);
	*n = (uint32_t)PRVM_G_READFLOAT(OFS_PARM4);
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
		(int)PRVM_G_READFLOAT(OFS_PARM0),
		(uint16_t)PRVM_G_READFLOAT(OFS_PARM1),
		(uint16_t)PRVM_G_READFLOAT(OFS_PARM2),
		(uint32_t)PRVM_G_READFLOAT(OFS_PARM3),
		(uint32_t)PRVM_G_READFLOAT(OFS_PARM4));
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
	col = (uint32_t)PRVM_G_READFLOAT(OFS_PARM1);
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
	col = (uint32_t)PRVM_G_READFLOAT(OFS_PARM1);
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
	*fld = PRVM_G_READINT(OFS_PARM1);
	*first = (uint32_t)PRVM_G_READFLOAT(OFS_PARM2);
	*n = (uint32_t)PRVM_G_READFLOAT(OFS_PARM3);
	*cols = (uint32_t)PRVM_G_READFLOAT(OFS_PARM4);
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
	fld = PRVM_G_READINT(OFS_PARM1);
	row = PRVM_G_EDICT(OFS_PARM2);
	nextfld = PRVM_G_READINT(OFS_PARM3);
	n = (uint32_t)PRVM_G_READFLOAT(OFS_PARM4);
	cols = (uint32_t)PRVM_G_READFLOAT(OFS_PARM5);
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
	PRVM_ViewFor(prog, 0);
	m->session = prog->view_session;
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)meshx_publish(m, (uint32_t)PRVM_G_READFLOAT(OFS_PARM1), (uint32_t)PRVM_G_READFLOAT(OFS_PARM2));
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
	PRVM_G_FLOAT(OFS_RETURN) = (prvm_vec_t)meshx_stat(m, (int)PRVM_G_READFLOAT(OFS_PARM1));
}


static unsigned int MeshX_ViewSchema(prvm_prog_t *prog)
{
	uint32_t value = (uint32_t)prog->filecrc;
	value = (value * 16777619u) ^ prog->numglobals;
	value = (value * 16777619u) ^ prog->entityfields;
	value = (value * 16777619u) ^ sizeof(prvm_vec_t);
	value = (value * 16777619u) ^ PRVM_VIEW_STATE_WIDTH;
	value = (value * 16777619u) ^ PRVM_VIEW_RESPONSE_WIDTH;
	value = (value * 16777619u) ^ PRVM_VIEW_READONLY;
	return value & 0xffffff;
}

static void MeshX_ViewConsume(prvm_prog_t *prog)
{
	meshxhandle_t *m = meshx_get(prog->view_response_handle - 1);
	const float **headers;
	uint32_t row, stale = 0;
	if (!m || m->done_id <= prog->view_response_sequence)
		return;
	prog->view_response_sequence = m->done_id;
	headers = Mem_Alloc(tempmempool, prog->num_edicts * sizeof(*headers));
	for (row = 0; row < m->done_rows; ++row)
	{
		float *data = m->resp + (size_t)row * m->width;
		int owner, entity, offset, width;
		prvm_view_t *view;
		prvm_view_page_t *page;
		for (int i = 0; i < m->width; ++i)
			if (!isfinite(data[i]))
				goto invalid;
		for (int i = 0; i < PRVM_VIEW_RESPONSE_HEADER; ++i)
			if ((i < 2 || i >= 5) && data[i] != floorf(data[i]))
				goto invalid;
		if (data[0] < 0 || data[0] >= prog->num_edicts || data[5] < -1 || data[5] >= prog->num_edicts ||
			data[6] < 0 || data[6] >= (data[5] < 0 ? prog->numglobals : prog->entityfields) ||
			data[1] != MeshX_ViewSchema(prog))
			goto invalid;
		owner = data[0]; entity = data[5]; offset = data[6];
		width = entity < 0 ? prog->numglobals : prog->entityfields;
		if (offset % PRVM_VIEW_PAGE)
			goto invalid;
		if (headers[owner])
		{
			for (int i = 0; i < PRVM_VIEW_RESPONSE_HEADER; ++i)
				if ((i < 5 || i >= 8) && data[i] != headers[owner][i])
					goto invalid;
		}
		headers[owner] = data;
		view = PRVM_ViewFor(prog, owner);
		page = PRVM_ViewPage(prog, view, entity, offset);
		if (page->checked_sequence == m->done_id)
			goto invalid;
		page->checked_sequence = m->done_id;
		for (int i = max(0, width - offset); i < PRVM_VIEW_PAGE; ++i)
			if (data[PRVM_VIEW_RESPONSE_HEADER + i] || data[PRVM_VIEW_RESPONSE_HEADER + PRVM_VIEW_PAGE + i])
				goto invalid;
	}
	for (row = 0; row < m->done_rows; ++row)
	{
		float *data = m->resp + (size_t)row * m->width;
		prvm_view_t *view = PRVM_ViewFor(prog, (int)data[0]);
		prvm_view_page_t *page = PRVM_ViewPage(prog, view, (int)data[5], (int)data[6]);
		if (prog->edicts[view->owner].priv.required->free || data[8] != view->generation || data[7] != page->generation)
		{
			++stale;
			continue;
		}
		memcpy(page->residual, data + PRVM_VIEW_RESPONSE_HEADER, sizeof(page->residual));
		memcpy(page->velocity, data + PRVM_VIEW_RESPONSE_HEADER + PRVM_VIEW_PAGE, sizeof(page->velocity));
		page->forcing = 0;
		memset(page->applied_epoch, 0, sizeof(page->applied_epoch));
		for (int i = 0; i < PRVM_VIEW_PAGE; ++i)
		{
			PRVM_ViewMask(view, page, i);
			page->forcing |= page->residual[i] != 0 || page->velocity[i] != 0;
		}
		if (!page->forcing)
		{
			memset(page->applied, 0, sizeof(page->applied));
			memset(page->integer_applied, 0, sizeof(page->integer_applied));
		}
		view->source_time = data[2];
		view->duration = data[3];
		view->tau = data[4];
		view->sequence = m->done_id;
	}
	if (stale)
		Con_Printf("mesh view response %u applied %u pages; %u pages belonged to departed owners or replaced entities\n", m->done_id, m->done_rows - stale, stale);
	Mem_Free(headers);
	return;
invalid:
	Con_Printf("mesh view response %u row %u has an invalid numeric, memory or snapshot extent; entire previous view continues\n", m->done_id, row);
	Mem_Free(headers);
}

void VM_mesh_round_outcome(prvm_prog_t *prog)
{
	meshxhandle_t *m;
	VM_SAFEPARMCOUNT(3, VM_mesh_round_outcome);
	PRVM_ViewFor(prog, 0);
	m = meshx_get(meshx_open((int)PRVM_G_READFLOAT(OFS_PARM0), MESH_XON_OUTCOME, 0, 5, 1));
	if (!m)
		return;
	m->session = prog->view_session;
	m->req[0] = prog->view_session >> 24;
	m->req[1] = prog->view_session & 0xffffff;
	m->req[2] = PRVM_G_READFLOAT(OFS_PARM1);
	m->req[3] = PRVM_G_READFLOAT(OFS_PARM2);
	m->req[4] = PRVM_serverglobalfloat(time);
	meshx_publish(m, 0, 1);
}

void VM_mesh_view_publish(prvm_prog_t *prog)
{
	int node = PRVM_G_READFLOAT(OFS_PARM0), tick = PRVM_G_READFLOAT(OFS_PARM1);
	prvm_view_t *view;
	prvm_view_page_t *page;
	meshxhandle_t *m;
	uint32_t rows = 0;
	VM_SAFEPARMCOUNT(2, VM_mesh_view_publish);
	PRVM_ViewFor(prog, 0);
	if (!prog->view_state_handle)
		prog->view_state_handle = meshx_open(node, MESH_XON_STATE, 0, PRVM_VIEW_STATE_WIDTH, 1) + 1;
	if (!prog->view_response_handle)
		prog->view_response_handle = meshx_open(node, 0, MESH_XON_STRATEGY, PRVM_VIEW_RESPONSE_WIDTH, 1) + 1;
	m = meshx_get(prog->view_state_handle - 1);
	if (!m)
		return;
	MeshX_ViewConsume(prog);
	m->session = prog->view_session;
	for (view = prog->views; view; view = view->next)
	{
		PRVM_ViewFor(prog, view->owner);
		for (page = view->pages; page; page = page->next)
		{
			PRVM_ViewPage(prog, view, page->entity, page->offset);
			++rows;
		}
	}
	if (!meshx_reserve(m, max(1, rows)))
		return;
	rows = 0;
	for (view = prog->views; view; view = view->next)
	{
		PRVM_ViewTime(view, PRVM_serverglobalfloat(time));
		for (page = view->pages; page; page = page->next)
		{
			float *data = m->req + (size_t)rows++ * m->width;
			data[0] = view->owner;
			data[1] = MeshX_ViewSchema(prog);
			data[2] = PRVM_serverglobalfloat(time);
			data[3] = view->applied_sequence;
			data[4] = page->entity;
			data[5] = page->offset;
			data[6] = page->generation;
			data[7] = view->generation;
			data[8] = prog->view_session >> 24;
			data[9] = prog->view_session & 0xffffff;
			data[10] = view->source_time;
			data[11] = view->duration;
			data[12] = view->tau;
			data[13] = view->sequence;
			data[14] = page->forcing;
			for (int i = 0; i < PRVM_VIEW_PAGE; ++i)
			{
				int type = page->present[i] ? (page->type[i] ? page->type[i] : ev_pointer) : 0;
				uint32_t bits = (uint32_t)page->integer_state[i];
				int encoded = type && (type != ev_float || !isfinite(page->state[i]));
				PRVM_ViewMask(view, page, i);
				data[PRVM_VIEW_STATE_HEADER + i] = encoded ? bits & 0xffff : page->state[i];
				data[PRVM_VIEW_STATE_HEADER + PRVM_VIEW_PAGE + i] = page->forcing ? view->decay * page->residual[i] + view->gain * page->velocity[i] : 0;
				data[PRVM_VIEW_STATE_HEADER + 2 * PRVM_VIEW_PAGE + i] = (encoded ? (1u << 20) | ((bits >> 16) << 3) | type : type) |
					(type && PRVM_ViewReadOnly(view, page->entity, page->offset + i) ? PRVM_VIEW_READONLY : 0);
				data[PRVM_VIEW_STATE_HEADER + 3 * PRVM_VIEW_PAGE + i] = page->read_time[i];
				data[PRVM_VIEW_STATE_HEADER + 4 * PRVM_VIEW_PAGE + i] = page->velocity[i];
				data[PRVM_VIEW_STATE_HEADER + 5 * PRVM_VIEW_PAGE + i] = page->residual[i];
			}
		}
	}
	PRVM_G_FLOAT(OFS_RETURN) = meshx_publish(m, tick, rows);
}

#endif
