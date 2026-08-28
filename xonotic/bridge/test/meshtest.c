#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include "../solver/mesh_attach.h"

extern char **environ;

typedef struct host_s
{
	mesh_hdr_t *hdr;
	float *req;
	float *resp;
	uint64_t seq;
	uint64_t delivered;
	uint64_t misses;
}
host_t;

static void host_publish(host_t *H)
{
	H->seq++;
	mesh_put_req(H->hdr, H->seq, H->req);
}

static void host_poll(host_t *H)
{
	H->delivered = mesh_get_resp(H->hdr, H->delivered, H->resp, &H->misses);
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

static double pct(double *v, int n, double p)
{
	int i = (int)(p * (n - 1) + 0.5);
	return v[i < 0 ? 0 : (i >= n ? n - 1 : i)];
}

static void fillreq(float *r, uint32_t n, uint64_t seq)
{
	uint32_t i;
	for (i = 0; i < n; i++)
		r[i] = (float)((seq * 1103515245u + i * 12345u) % 1000u) - 500.0f;
	r[0] = 0.0f;
	r[1] = 0.0f;
	r[2] = 1.0f;
	r[3] = 2.0f;
	r[4] = -0.0f;
	r[5] = 65536.0f;
	if (n > 6)
		r[n - 1] = 0.0f;
}

static int verify(const float *in, const float *out, uint32_t nreq, uint32_t nresp)
{
	uint32_t i;
	for (i = 0; i < nresp; i++)
		if (out[i] != in[i % nreq] * 2.0f + 1.0f)
			return (int)i + 1;
	return 0;
}

static int nulhazard(const float *v, uint32_t n)
{
	const unsigned char *b = (const unsigned char *)v;
	size_t total = (size_t)n * sizeof(float), i;
	for (i = 0; i < total; i++)
		if (!b[i])
			return (int)i;
	return -1;
}

int main(int argc, char **argv)
{
	const char *name = argc > 1 ? argv[1] : "/mesh_bridge_test";
	uint32_t nfloats = argc > 2 ? (uint32_t)atoi(argv[2]) : 4096;
	int ticks = argc > 3 ? atoi(argv[3]) : 600;
	double hz = argc > 4 ? atof(argv[4]) : 60.0;
	double work_us = argc > 5 ? atof(argv[5]) : 0.0;
	const char *solverpath = argc > 6 ? argv[6] : "./fakesolver";
	int blocking = argc > 7 ? atoi(argv[7]) : 0;
	double waitcap_us = argc > 8 ? atof(argv[8]) : 4000.0;

	host_t H;
	size_t bytes;
	double *tpub, *bridge_us, *rt_us;
	uint64_t *seq_at_tick;
	int t, nrt = 0, bad = 0, err;
	double t0, t1, tick_dt, start, next, wall;
	pid_t pid;
	char *sargv[5];
	char rf[32], wu[32];
	uint64_t delivered_before;

	shm_unlink(name);
	H.hdr = mesh_create(name, nfloats, nfloats, &bytes);
	if (!H.hdr)
	{
		fprintf(stderr, "mesh_create failed\n");
		return 1;
	}
	H.req = (float *)calloc(nfloats + 1, sizeof(float));
	H.resp = (float *)calloc(nfloats + 1, sizeof(float));
	H.seq = H.delivered = H.misses = 0;

	err = nulhazard(H.req, nfloats);
	printf("region %s  %u floats each way  %zu bytes  depth %u\n", name, nfloats, bytes, MESH_DEPTH);
	printf("nul-hazard: first zero byte in a zeroed fp32 request at offset %d of %zu -> any strlen-based builtin would move %d of %zu bytes\n",
		err, (size_t)nfloats * 4, err, (size_t)nfloats * 4);

	snprintf(rf, sizeof(rf), "%.3f", getenv("MESH_SOLVER_RUNFOR") ? atof(getenv("MESH_SOLVER_RUNFOR")) : ticks / hz + 10.0);
	snprintf(wu, sizeof(wu), "%.3f", work_us);
	sargv[0] = (char *)solverpath;
	sargv[1] = (char *)name;
	sargv[2] = rf;
	sargv[3] = wu;
	sargv[4] = NULL;
	if (posix_spawn(&pid, solverpath, NULL, NULL, sargv, environ) != 0)
	{
		fprintf(stderr, "spawn %s failed\n", solverpath);
		return 1;
	}

	tpub = (double *)calloc(ticks + 2, sizeof(double));
	bridge_us = (double *)calloc(ticks + 2, sizeof(double));
	rt_us = (double *)calloc(ticks + 2, sizeof(double));
	seq_at_tick = (uint64_t *)calloc(ticks + 2, sizeof(uint64_t));

	next = mesh_now() + 10.0;
	fillreq(H.req, nfloats, 1);
	host_publish(&H);
	while (!atomic_load_explicit(&H.hdr->solver_alive, memory_order_relaxed) && mesh_now() < next)
		usleep(200);
	H.delivered = atomic_load_explicit(&H.hdr->resp_seq, memory_order_acquire);

	tick_dt = 1.0 / hz;
	start = mesh_now();
	next = start;
	for (t = 0; t < ticks; t++)
	{
		while (mesh_now() < next)
			;
		next += tick_dt;

		t0 = mesh_now();
		fillreq(H.req, nfloats, H.seq + 1);
		host_publish(&H);
		tpub[H.seq % (ticks + 2)] = t0;
		delivered_before = H.delivered;
		host_poll(&H);
		if (blocking)
		{
			double cap = t0 + waitcap_us * 1e-6;
			while (H.delivered < H.seq && mesh_now() < cap)
				host_poll(&H);
		}
		t1 = mesh_now();
		bridge_us[t] = (t1 - t0) * 1e6;
		seq_at_tick[t] = H.delivered;

		if (H.delivered > delivered_before)
		{
			rt_us[nrt++] = (t1 - tpub[H.delivered % (ticks + 2)]) * 1e6;
			fillreq(H.req, nfloats, H.delivered);
			err = verify(H.req, H.resp, nfloats, nfloats);
			bad += err != 0;
		}
	}
	wall = mesh_now() - start;

	kill(pid, SIGTERM);
	waitpid(pid, NULL, 0);

	qsort(bridge_us, ticks, sizeof(double), cmpd);
	qsort(rt_us, nrt, sizeof(double), cmpd);

	printf("ticks %d at %.1f Hz over %.3f s  published %llu  delivered %llu  responses seen %d  mismatches %d  seqlock retries %llu\n",
		ticks, hz, wall, (unsigned long long)H.seq, (unsigned long long)H.delivered, nrt, bad, (unsigned long long)H.misses);
	printf("mode                      %s\n", blocking ? "in-frame blocking (mesh_wait)" : "pipelined (publish+poll, never blocks)");
	printf("in-frame bridge cost us   med %.2f  p90 %.2f  p99 %.2f  max %.2f\n",
		pct(bridge_us, ticks, 0.5), pct(bridge_us, ticks, 0.9), pct(bridge_us, ticks, 0.99), bridge_us[ticks - 1]);
	if (nrt)
		printf("publish->deliver us       med %.2f  p90 %.2f  p99 %.2f  max %.2f\n",
			pct(rt_us, nrt, 0.5), pct(rt_us, nrt, 0.9), pct(rt_us, nrt, 0.99), rt_us[nrt - 1]);
	printf("payload throughput        %.1f MB/s both ways at this rate (%llu round trips / %.3f s, %zu B each way)\n",
		2.0 * (double)nrt * nfloats * 4.0 / wall / 1e6, (unsigned long long)nrt, wall, (size_t)nfloats * 4);

	free(tpub);
	free(bridge_us);
	free(rt_us);
	free(seq_at_tick);
	munmap(H.hdr, bytes);
	shm_unlink(name);
	return bad != 0;
}
