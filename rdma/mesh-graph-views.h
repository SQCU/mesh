#pragma once
#include "mesh-graph.h"
#include <memory>

struct GraphDomain {
    mesh_graph_domain *handle;
    explicit GraphDomain(mesh_graph_domain *value) : handle(value) {}
    GraphDomain(const GraphDomain &) = delete;
    ~GraphDomain() { mesh_graph_domain_free(handle); }
};

struct GraphIndex {
    mesh_graph_index *handle;
    explicit GraphIndex(mesh_graph_index *value) : handle(value) {}
    GraphIndex(const GraphIndex &) = delete;
    ~GraphIndex() { mesh_graph_index_free(handle); }
};

struct GraphAccessMap {
    const mesh_graph_access_map *handle;
    explicit GraphAccessMap(const mesh_graph_access_map *value) : handle(value) {}
    GraphAccessMap(const GraphAccessMap &) = delete;
    ~GraphAccessMap() { mesh_graph_access_map_free(handle); }
};
