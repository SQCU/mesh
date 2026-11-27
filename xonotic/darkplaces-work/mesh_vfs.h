#ifndef MESH_VFS_H
#define MESH_VFS_H

#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>

#define MESH_VFS_MAGIC 0x4D564653u
#define MESH_VFS_VERSION 1u
#define MESH_VFS_PATH_MAX 192u

typedef struct mesh_vfs_entry_s
{
	char path[MESH_VFS_PATH_MAX];
	uint64_t offset;
	uint64_t size;
	uint32_t crc32;
	uint32_t flags;
	uint64_t reserved;
}
mesh_vfs_entry_t;

typedef struct mesh_vfs_header_s
{
	uint32_t magic;
	uint32_t version;
	uint64_t total_size;
	uint64_t data_offset;
	_Atomic uint64_t generation;
	uint32_t capacity;
	uint32_t count;
	uint64_t write_cursor;
	uint64_t reserved[2];
}
mesh_vfs_header_t;

_Static_assert(sizeof(mesh_vfs_header_t) == 64, "mesh_vfs_header_t must be 64 bytes");
_Static_assert(sizeof(mesh_vfs_entry_t) == 224, "mesh_vfs_entry_t must be 224 bytes");

#ifndef WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <strings.h>
#endif

static const unsigned char *mesh_vfs_base = NULL;
static size_t mesh_vfs_mapped_size = 0;

static void MeshVFS_NormalizePath(const char *in, char *out, size_t out_size)
{
	size_t i = 0;
	while (*in == '/' || *in == '\\')
		in++;
	while (*in && i + 1 < out_size)
	{
		char c = *in++;
		if (c == '\\')
			c = '/';
		out[i++] = c;
	}
	out[i] = '\0';
}

static void MeshVFS_Shutdown(void)
{
#ifndef WIN32
	if (mesh_vfs_base && mesh_vfs_mapped_size > 0)
		munmap((void *)mesh_vfs_base, mesh_vfs_mapped_size);
#endif
	mesh_vfs_base = NULL;
	mesh_vfs_mapped_size = 0;
}

static void MeshVFS_Init(void)
{
#ifndef WIN32
	const char *spec = getenv("MESH_VFS_PATH");
	int fd = -1;
	struct stat st;
	void *addr;
	const mesh_vfs_header_t *hdr;

	if (!spec || !spec[0])
		return;
	if (mesh_vfs_base)
		MeshVFS_Shutdown();

	if (!strncmp(spec, "shm://", 6))
	{
		char shm_name[256];
		const char *raw = spec + 6;
		if (raw[0] == '/')
			strlcpy(shm_name, raw, sizeof(shm_name));
		else
			dpsnprintf(shm_name, sizeof(shm_name), "/%s", raw);
		fd = shm_open(shm_name, O_RDONLY, 0);
	}
	else
	{
		fd = open(spec, O_RDONLY);
	}
	if (fd < 0)
	{
		Con_Printf("MeshVFS_Init: failed to open %s\n", spec);
		return;
	}
	if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(mesh_vfs_header_t))
	{
		close(fd);
		Con_Printf("MeshVFS_Init: invalid size for %s\n", spec);
		return;
	}
	addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (addr == MAP_FAILED)
	{
		Con_Printf("MeshVFS_Init: mmap failed for %s\n", spec);
		return;
	}
	hdr = (const mesh_vfs_header_t *)addr;
	if (hdr->magic != MESH_VFS_MAGIC || hdr->version != MESH_VFS_VERSION || hdr->total_size > (uint64_t)st.st_size)
	{
		munmap(addr, (size_t)st.st_size);
		Con_Printf("MeshVFS_Init: bad header in %s\n", spec);
		return;
	}
	mesh_vfs_base = (const unsigned char *)addr;
	mesh_vfs_mapped_size = (size_t)st.st_size;
	Con_Printf("MeshVFS_Init: mapped %s (%llu MB, capacity %u)\n", spec, (unsigned long long)(hdr->total_size >> 20), hdr->capacity);
#endif
}

static qboolean MeshVFS_Lookup(const char *filepath, const unsigned char **out_ptr, size_t *out_size)
{
	char norm[MESH_VFS_PATH_MAX];
	const mesh_vfs_header_t *hdr;
	const mesh_vfs_entry_t *entries;
	int attempt;

	if (!mesh_vfs_base || !filepath || !filepath[0])
		return false;

	MeshVFS_NormalizePath(filepath, norm, sizeof(norm));
	hdr = (const mesh_vfs_header_t *)mesh_vfs_base;
	entries = (const mesh_vfs_entry_t *)(mesh_vfs_base + sizeof(mesh_vfs_header_t));

	for (attempt = 0; attempt < 64; attempt++)
	{
		uint64_t g1 = atomic_load_explicit(&hdr->generation, memory_order_acquire);
		uint32_t cnt, cap, i;
		qboolean found = false;
		uint64_t f_off = 0, f_sz = 0;

		if (g1 & 1u)
			continue;
		cap = hdr->capacity;
		cnt = hdr->count;
		if (cnt > cap)
			cnt = cap;
		if (sizeof(mesh_vfs_header_t) + (uint64_t)cap * sizeof(mesh_vfs_entry_t) > mesh_vfs_mapped_size)
			return false;

		for (i = 0; i < cnt; i++)
		{
			if ((entries[i].flags & 1u) && !strcasecmp(entries[i].path, norm))
			{
				f_off = entries[i].offset;
				f_sz = entries[i].size;
				found = true;
				break;
			}
		}

		atomic_thread_fence(memory_order_acquire);
		if (atomic_load_explicit(&hdr->generation, memory_order_acquire) != g1)
			continue;

		if (!found)
			return false;
		if (f_off + f_sz > mesh_vfs_mapped_size)
			return false;
		if (out_ptr)
			*out_ptr = mesh_vfs_base + f_off;
		if (out_size)
			*out_size = (size_t)f_sz;
		return true;
	}
	return false;
}

static void MeshVFS_SearchAppend(const char *pattern, stringlist_t *resultlist)
{
	const mesh_vfs_header_t *hdr;
	const mesh_vfs_entry_t *entries;
	uint64_t g1;
	uint32_t cnt, cap, i;

	if (!mesh_vfs_base || !pattern)
		return;
	hdr = (const mesh_vfs_header_t *)mesh_vfs_base;
	entries = (const mesh_vfs_entry_t *)(mesh_vfs_base + sizeof(mesh_vfs_header_t));
	g1 = atomic_load_explicit(&hdr->generation, memory_order_acquire);
	if (g1 & 1u)
		return;
	cap = hdr->capacity;
	cnt = hdr->count;
	if (cnt > cap)
		cnt = cap;
	if (sizeof(mesh_vfs_header_t) + (uint64_t)cap * sizeof(mesh_vfs_entry_t) > mesh_vfs_mapped_size)
		return;
	for (i = 0; i < cnt; i++)
	{
		if (!(entries[i].flags & 1u))
			continue;
		if (matchpattern(entries[i].path, (char *)pattern, true))
		{
			int j;
			for (j = 0; j < resultlist->numstrings; j++)
				if (!strcmp(resultlist->strings[j], entries[i].path))
					break;
			if (j == resultlist->numstrings)
				stringlistappend(resultlist, entries[i].path);
		}
	}
}

#endif
