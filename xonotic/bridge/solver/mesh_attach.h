#ifndef MESH_ATTACH_H
#define MESH_ATTACH_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include "../engine/mesh_shm.h"

#ifdef __APPLE__
static inline double mesh_now(void)
{
	return 1e-9 * (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}
#else
static inline double mesh_now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
#endif

static inline mesh_hdr_t *mesh_create(const char *name, uint32_t nreq, uint32_t nresp, size_t *bytes_out)
{
	size_t bytes = mesh_bytes(nreq, nresp);
	mesh_hdr_t *p;
	int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
	if (fd < 0)
		return NULL;
	ftruncate(fd, (off_t)bytes);
	p = (mesh_hdr_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (p == MAP_FAILED)
		return NULL;
	mesh_reset(p, nreq, nresp);
	if (bytes_out)
		*bytes_out = bytes;
	return p;
}

static inline mesh_hdr_t *mesh_attach(const char *name, double timeout, size_t *bytes_out)
{
	int fd;
	mesh_hdr_t *p;
	uint32_t magic, nreq, nresp, version;
	size_t bytes;
	double deadline = mesh_now() + timeout;

	for (;;)
	{
		magic = 0;
		nreq = nresp = version = 0;
		fd = shm_open(name, O_RDWR, 0600);
		if (fd >= 0)
		{
			p = (mesh_hdr_t *)mmap(NULL, sizeof(mesh_hdr_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if (p != MAP_FAILED)
			{
				magic = p->magic;
				atomic_thread_fence(memory_order_acquire);
				nreq = p->nreq;
				nresp = p->nresp;
				version = p->version;
				munmap(p, sizeof(mesh_hdr_t));
			}
		}
		if (magic == MESH_MAGIC && version == MESH_VERSION)
			break;
		if (fd >= 0)
			close(fd);
		if (mesh_now() >= deadline)
			return NULL;
		usleep(500);
	}

	bytes = mesh_bytes(nreq, nresp);
	p = (mesh_hdr_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (p == MAP_FAILED)
		return NULL;
	if (bytes_out)
		*bytes_out = bytes;
	return p;
}

#endif
