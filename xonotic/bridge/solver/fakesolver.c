#include <signal.h>
#include "mesh_attach.h"

static volatile sig_atomic_t running = 1;
static void onsig(int s) { (void)s; running = 0; }

int main(int argc, char **argv)
{
	const char *name = argc > 1 ? argv[1] : "/mesh_bridge_test";
	double runfor = argc > 2 ? atof(argv[2]) : 30.0;
	double work_us = argc > 3 ? atof(argv[3]) : 0.0;
	mesh_hdr_t *h;
	float *in, *out;
	uint64_t last = 0, seq, served = 0, misses = 0, epoch;
	uint32_t i;
	double deadline, spin, tnext, t0;

	signal(SIGINT, onsig);
	signal(SIGTERM, onsig);

	h = mesh_attach(name, 10.0, NULL);
	if (!h)
	{
		fprintf(stderr, "fakesolver: attach %s failed\n", name);
		return 1;
	}
	in = (float *)calloc(h->nreq + 1, sizeof(float));
	out = (float *)calloc(h->nresp + 1, sizeof(float));
	fprintf(stderr, "fakesolver: attached %s nreq=%u nresp=%u depth=%u\n", name, h->nreq, h->nresp, h->depth);

	epoch = atomic_load_explicit(&h->epoch, memory_order_acquire);
	t0 = mesh_now();
	tnext = t0;
	deadline = t0 + runfor;
	while (running && mesh_now() < deadline)
	{
		if (getenv("MESH_TRACE") && mesh_now() > tnext)
		{
			tnext = mesh_now() + 1.0;
			fprintf(stderr, "fakesolver: t=%.1f req_seq=%llu resp_seq=%llu last=%llu served=%llu\n", mesh_now() - t0,
				(unsigned long long)atomic_load_explicit(&h->req_seq, memory_order_acquire),
				(unsigned long long)atomic_load_explicit(&h->resp_seq, memory_order_acquire),
				(unsigned long long)last, (unsigned long long)served);
		}
		if (atomic_load_explicit(&h->epoch, memory_order_acquire) != epoch)
		{
			epoch = atomic_load_explicit(&h->epoch, memory_order_acquire);
			last = 0;
		}
		seq = mesh_get_req(h, last, in, &misses);
		if (seq == last)
			continue;
		last = seq;
		for (i = 0; i < h->nresp; i++)
			out[i] = in[i % h->nreq] * 2.0f + 1.0f;
		spin = mesh_now() + work_us * 1e-6;
		while (mesh_now() < spin)
			;
		mesh_put_resp(h, seq, out);
		atomic_store_explicit(&h->solver_alive, ++served, memory_order_relaxed);
	}
	fprintf(stderr, "fakesolver: served %llu seqmisses %llu\n", (unsigned long long)served, (unsigned long long)misses);
	return 0;
}
