#ifndef MESH_CURVE_RECONCILE_H
#define MESH_CURVE_RECONCILE_H

#include <stddef.h>

typedef struct
{
	double anchor;
	double strain;
	double bend;
	double cusp;
	double tangent_point;
	double thickness;
	double length_scale;
	double thickness_scale;
	double tangent_power;
	double thickness_power;
	double cusp_epsilon;
	double spatial_epsilon;
}
mesh_curve_weights_t;

typedef struct
{
	double anchor_energy;
	double strain_energy;
	double bend_energy;
	double cusp_energy;
	double tangent_point_energy;
	double thickness_energy;
	double total_energy;
	double minimum_turn_cosine;
	double minimum_nonneighbor_segment_distance;
	size_t turn_atom_mass;
	size_t directed_tangent_point_pair_mass;
	size_t nonneighbor_segment_pair_mass;
}
mesh_curve_measures_t;

void MeshCurveAccumulate(const double *points, const double *reference,
	size_t point_mass, int closed, const mesh_curve_weights_t *weights,
	double *gradient, mesh_curve_measures_t *measures);

#endif
