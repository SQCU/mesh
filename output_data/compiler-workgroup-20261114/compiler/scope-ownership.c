#include "mesh-graph.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void released(void *owner) { ++*(unsigned *)owner; }
static int within(const void *p, const void *bytes, size_t length) {
    return (uintptr_t)p >= (uintptr_t)bytes && (uintptr_t)p - (uintptr_t)bytes < length;
}

int main(void) {
    unsigned payload = 0, realized = 0, original = 0, replacement = 0, rejected = 0;
    int old_source = 1, new_source = 2;
    const void *old_sources[] = {&old_source}, *new_sources[] = {&new_source};
    const struct mesh_graph_scope_source source = {.row = 0, .node = 0, .identity = {7}};
    const struct mesh_graph_completion completion = {0, 1, 0, 1, 1};
    const struct mesh_graph_completion_writer writer = {0, 0, MESH_GRAPH_NONE, 0, 1};
    const struct mesh_graph_release release = {0, 0};
    const size_t first[] = {0, 1};
    const struct mesh_graph_scopes_data data = {1, 1, 1, &completion, &writer, first, &release, &source};
    struct mesh_graph_scopes *scopes = NULL;
    assert(!mesh_graph_scopes_create(1, &data, released, &payload, &scopes));
    assert(!mesh_graph_scopes_bind(scopes, 1, old_sources, released, &original));
    const struct mesh_graph_processor processor = {MESH_GRAPH_BRIDGE, 0};
    const struct mesh_graph_task task = {0, 0, 0, 0};
    const struct mesh_graph_operation read = {0, MESH_GRAPH_BRIDGE_READ, 0, MESH_GRAPH_PLAN_SUPPLIED, 0, 1};
    struct mesh_graph_signature a = {0}, joined = {0}, restored = {0};
    assert(!mesh_graph_signature(1, &processor, 1, &task, 0, NULL, 1, &read, &a));
    const struct mesh_graph_occurrence occurrence = {0, 0, 0, 0, 0, 0, 0};
    const struct mesh_graph_site site = {0, MESH_GRAPH_NONE, 0, MESH_GRAPH_NONE};
    const uint64_t published = 1;
    assert(!mesh_graph_signature_scopes(&a, scopes, 1, &occurrence, 1, &site, 1, &published, released, &realized));
    assert(a.answers == 1 && a.states[MESH_GRAPH_SCOPED] == 1);
    unsigned char digest[32];
    memcpy(digest, a.digest, sizeof digest);
    struct mesh_graph_site invalid = site;
    invalid.first = 1;
    assert(mesh_graph_signature_scopes(&a, scopes, 1, &occurrence, 1, &invalid, 1, &published,
                                       released, &rejected) == EINVAL);
    assert(!rejected && !memcmp(digest, a.digest, sizeof digest));
    assert(!mesh_graph_compose(&a, &a, &joined));
    assert(joined.occurrences == 2 && joined.publications == 2 && joined.answers == 2);
    assert(!mesh_graph_scopes_bind(scopes, 1, new_sources, released, &replacement));
    assert(!original && mesh_graph_scopes_sources(joined.scopes)[0] == &old_source);
    size_t bytes = 0;
    assert(mesh_graph_record(&joined, NULL, 0, &bytes) == ERANGE);
    void *record = malloc(bytes);
    assert(record && !mesh_graph_record(&joined, record, bytes, &bytes));
    assert(!mesh_graph_record_read(record, &restored));
    size_t rows = 0;
    const struct mesh_graph_scopes_data *saved = mesh_graph_scopes_rows(restored.scopes, &rows);
    assert(rows == 1 && saved[0].source && saved[0].source->identity[0] == 7);
    assert(!mesh_graph_scopes_sources(restored.scopes));
    assert(restored.occurrences == 2 && restored.answers == 2);
    mesh_graph_signature_free(&restored);
    free(record);
    assert(mesh_graph_archive(&joined, NULL, 0, &bytes) == ERANGE);
    void *archive = malloc(bytes), *snapshot = malloc(bytes);
    assert(archive && snapshot && !mesh_graph_archive(&joined, archive, bytes, &bytes));
    memcpy(snapshot, archive, bytes);
    unsigned archived = 0;
    assert(mesh_graph_archive_read(archive, bytes - 1, released, &archived, &restored) == EINVAL && !archived);
    assert(!mesh_graph_archive_read(archive, bytes, released, &archived, &restored));
    assert(restored.answers == joined.answers && !memcmp(restored.digest, joined.digest, sizeof joined.digest));
    assert(!memcmp(restored.states, joined.states, sizeof joined.states));
    const void *views[] = {restored.processor, restored.task, restored.operation, restored.dependence,
        restored.edge, restored.position, restored.count, restored.least, restored.previous, restored.waited,
        restored.first, restored.order, restored.unknown, restored.alias, restored.answer,
        restored.occurrence, restored.site, restored.published};
    for (size_t k = 0; k < sizeof views / sizeof *views; k++) assert(within(views[k], archive, bytes));
    struct mesh_graph_signature retained = {0};
    assert(!mesh_graph_compose(&restored, &restored, &retained));
    assert(!mesh_graph_signature_program(&restored, 0, NULL));
    assert(!mesh_graph_signature_scopes(&restored, scopes, 1, &occurrence, 1, &site, 1, &published, NULL, NULL));
    assert(!memcmp(archive, snapshot, bytes));
    mesh_graph_signature_free(&restored);
    assert(!archived);
    mesh_graph_signature_free(&retained);
    assert(archived == 1);
    free(snapshot); free(archive);
    struct mesh_graph_signature plain = {0};
    assert(!mesh_graph_signature(1, &processor, 1, &task, 0, NULL, 1, &read, &plain));
    assert(mesh_graph_archive(&plain, NULL, 0, &bytes) == ERANGE);
    archive = malloc(bytes);
    assert(archive && !mesh_graph_archive(&plain, archive, bytes, &bytes));
    mesh_graph_signature_free(&plain);
    assert(!mesh_graph_archive_read(archive, bytes, released, &archived, &restored) && archived == 1);
    assert(within(restored.operation, archive, bytes));
    mesh_graph_signature_free(&restored);
    assert(archived == 2);
    free(archive);
    mesh_graph_signature_free(&a);
    mesh_graph_scopes_free(scopes);
    assert(!payload && !realized && !original);
    mesh_graph_signature_free(&joined);
    assert(payload == 1 && realized == 1 && original == 1 && replacement == 1 && !rejected);
    puts("scope ownership, full saved views, composition and source references: PASS");
    return 0;
}
