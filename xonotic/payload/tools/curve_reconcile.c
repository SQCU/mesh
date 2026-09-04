#include "curve_reconcile.h"

#include <float.h>
#include <math.h>
#include <string.h>

static double Dot3(const double *a, const double *b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void Sub3(const double *a, const double *b, double *out)
{
	out[0] = a[0] - b[0];
	out[1] = a[1] - b[1];
	out[2] = a[2] - b[2];
}

static double Length3(const double *a)
{
	return sqrt(Dot3(a, a));
}

static void AddGradient(double *gradient, size_t index, double scale,
	const double *value)
{
	gradient[index * 3 + 0] += scale * value[0];
	gradient[index * 3 + 1] += scale * value[1];
	gradient[index * 3 + 2] += scale * value[2];
}

static size_t Previous(size_t index, size_t point_mass, int closed)
{
	return index ? index - 1 : closed ? point_mass - 1 : 0;
}

static size_t Next(size_t index, size_t point_mass, int closed)
{
	return index + 1 < point_mass ? index + 1 : closed ? 0 : point_mass - 1;
}

static int Neighbor(size_t left, size_t right, size_t point_mass, int closed)
{
	if (left == right || left + 1 == right || right + 1 == left)
		return 1;
	return closed && ((left == 0 && right + 1 == point_mass)
		|| (right == 0 && left + 1 == point_mass));
}

static void ClosestSegments(const double *a, const double *b,
	const double *c, const double *d, double epsilon, double *left,
	double *right, double *delta)
{
	double u[3], v[3], w[3];
	double aa, bb, cc, dd, ee, determinant, s, t;
	Sub3(b, a, u);
	Sub3(d, c, v);
	Sub3(a, c, w);
	aa = Dot3(u, u);
	bb = Dot3(u, v);
	cc = Dot3(v, v);
	dd = Dot3(u, w);
	ee = Dot3(v, w);
	determinant = aa * cc - bb * bb;
	s = determinant > epsilon * epsilon ? (bb * ee - cc * dd) / determinant : 0.0;
	s = fmin(1.0, fmax(0.0, s));
	t = cc > epsilon * epsilon ? (bb * s + ee) / cc : 0.0;
	if (t < 0.0)
	{
		t = 0.0;
		s = aa > epsilon * epsilon ? fmin(1.0, fmax(0.0, -dd / aa)) : 0.0;
	}
	else if (t > 1.0)
	{
		t = 1.0;
		s = aa > epsilon * epsilon ? fmin(1.0, fmax(0.0, (bb - dd) / aa)) : 0.0;
	}
	left[0] = s;
	right[0] = t;
	delta[0] = a[0] + s * u[0] - c[0] - t * v[0];
	delta[1] = a[1] + s * u[1] - c[1] - t * v[1];
	delta[2] = a[2] + s * u[2] - c[2] - t * v[2];
}

void MeshCurveAccumulate(const double *points, const double *reference,
	size_t point_mass, int closed, const mesh_curve_weights_t *weights,
	double *gradient, mesh_curve_measures_t *measures)
{
	size_t i, j, edge_mass;
	double length_scale = fmax(weights->length_scale, DBL_EPSILON);
	double thickness_scale = fmax(weights->thickness_scale, DBL_EPSILON);
	double spatial_epsilon = fmax(weights->spatial_epsilon, DBL_EPSILON);
	double cusp_epsilon = fmax(weights->cusp_epsilon, DBL_EPSILON);
	double length_scale2 = length_scale * length_scale;
	double thickness_scale2 = thickness_scale * thickness_scale;
	double spatial_epsilon2 = spatial_epsilon * spatial_epsilon;
	double tangent_alpha = fmax(weights->tangent_power, DBL_EPSILON) * 0.5;
	double thickness_alpha = fmax(weights->thickness_power, DBL_EPSILON) * 0.5;
	memset(gradient, 0, sizeof(*gradient) * point_mass * 3);
	memset(measures, 0, sizeof(*measures));
	measures->minimum_turn_cosine = 1.0;
	measures->minimum_nonneighbor_segment_distance = INFINITY;
	edge_mass = closed && point_mass > 2 ? point_mass : point_mass > 1 ? point_mass - 1 : 0;
	for (i = 0; i < point_mass; ++i)
	{
		double delta[3];
		Sub3(points + i * 3, reference + i * 3, delta);
		measures->anchor_energy += weights->anchor * Dot3(delta, delta) / length_scale2;
		AddGradient(gradient, i, 2.0 * weights->anchor / length_scale2, delta);
	}
	for (i = 0; i < edge_mass; ++i)
	{
		size_t a = i;
		size_t b = Next(i, point_mass, closed);
		double current[3], initial[3];
		double current_length, initial_length, residual, scale;
		Sub3(points + b * 3, points + a * 3, current);
		Sub3(reference + b * 3, reference + a * 3, initial);
		current_length = Length3(current);
		initial_length = Length3(initial);
		residual = current_length - initial_length;
		measures->strain_energy += weights->strain * residual * residual / length_scale2;
		if (current_length > spatial_epsilon)
		{
			scale = 2.0 * weights->strain * residual
				/ (length_scale2 * current_length);
			AddGradient(gradient, a, -scale, current);
			AddGradient(gradient, b, scale, current);
		}
	}
	for (i = closed ? 0 : 1; i < (closed ? point_mass : point_mass > 0 ? point_mass - 1 : 0); ++i)
	{
		size_t a = Previous(i, point_mass, closed);
		size_t b = Next(i, point_mass, closed);
		double left[3], right[3], second[3], u[3], v[3];
		double left_length, right_length, cosine, denominator, derivative;
		Sub3(points + i * 3, points + a * 3, left);
		Sub3(points + b * 3, points + i * 3, right);
		second[0] = right[0] - left[0];
		second[1] = right[1] - left[1];
		second[2] = right[2] - left[2];
		measures->bend_energy += weights->bend * Dot3(second, second) / length_scale2;
		AddGradient(gradient, a, 2.0 * weights->bend / length_scale2, second);
		AddGradient(gradient, i, -4.0 * weights->bend / length_scale2, second);
		AddGradient(gradient, b, 2.0 * weights->bend / length_scale2, second);
		left_length = Length3(left);
		right_length = Length3(right);
		if (left_length <= spatial_epsilon || right_length <= spatial_epsilon)
			continue;
		for (j = 0; j < 3; ++j)
		{
			u[j] = left[j] / left_length;
			v[j] = right[j] / right_length;
		}
		cosine = fmin(1.0, fmax(-1.0, Dot3(u, v)));
		denominator = 1.0 + cosine + cusp_epsilon;
		measures->cusp_energy += weights->cusp
			* (1.0 / denominator - 1.0 / (2.0 + cusp_epsilon));
		measures->minimum_turn_cosine = fmin(measures->minimum_turn_cosine, cosine);
		measures->turn_atom_mass += 1;
		derivative = -weights->cusp / (denominator * denominator);
		{
			double dl[3], dr[3];
			for (j = 0; j < 3; ++j)
			{
				dl[j] = derivative * (v[j] - cosine * u[j]) / left_length;
				dr[j] = derivative * (u[j] - cosine * v[j]) / right_length;
			}
			AddGradient(gradient, a, -1.0, dl);
			AddGradient(gradient, i, 1.0, dl);
			AddGradient(gradient, i, -1.0, dr);
			AddGradient(gradient, b, 1.0, dr);
		}
	}
	for (i = closed ? 0 : 1; i < (closed ? point_mass : point_mass > 0 ? point_mass - 1 : 0); ++i)
	{
		size_t previous = Previous(i, point_mass, closed);
		size_t next = Next(i, point_mass, closed);
		double tangent_raw[3], tangent[3], tangent_length;
		Sub3(points + next * 3, points + previous * 3, tangent_raw);
		tangent_length = Length3(tangent_raw);
		if (tangent_length <= spatial_epsilon)
			continue;
		for (j = 0; j < 3; ++j)
			tangent[j] = tangent_raw[j] / tangent_length;
		for (j = 0; j < point_mass; ++j)
		{
			double delta[3], delta2, axial, normal2, denominator, rho, base, energy;
			double factor, gradient_delta[3], gradient_tangent[3], projected[3];
			size_t axis;
			if (Neighbor(i, j, point_mass, closed))
				continue;
			Sub3(points + j * 3, points + i * 3, delta);
			delta2 = Dot3(delta, delta);
			axial = Dot3(delta, tangent);
			normal2 = fmax(0.0, delta2 - axial * axial);
			denominator = delta2 + spatial_epsilon2;
			rho = 4.0 * normal2 / (denominator * denominator);
			base = length_scale2 * rho;
			energy = weights->tangent_point * pow(base, tangent_alpha);
			measures->tangent_point_energy += energy;
			measures->directed_tangent_point_pair_mass += 1;
			if (rho <= 0.0 || weights->tangent_point == 0.0)
				continue;
			factor = weights->tangent_point * tangent_alpha * length_scale2
				* pow(base, tangent_alpha - 1.0);
			for (axis = 0; axis < 3; ++axis)
			{
				gradient_delta[axis] = factor * 4.0
					* ((2.0 * delta[axis] - 2.0 * axial * tangent[axis])
					/ (denominator * denominator)
					- 4.0 * normal2 * delta[axis]
					/ (denominator * denominator * denominator));
				gradient_tangent[axis] = factor * -8.0 * axial * delta[axis]
					/ (denominator * denominator);
			}
			factor = Dot3(gradient_tangent, tangent);
			for (axis = 0; axis < 3; ++axis)
				projected[axis] = (gradient_tangent[axis] - factor * tangent[axis])
					/ tangent_length;
			AddGradient(gradient, i, -1.0, gradient_delta);
			AddGradient(gradient, j, 1.0, gradient_delta);
			AddGradient(gradient, previous, -1.0, projected);
			AddGradient(gradient, next, 1.0, projected);
		}
	}
	for (i = 0; i < edge_mass; ++i)
	{
		size_t a = i;
		size_t b = Next(i, point_mass, closed);
		for (j = i + 1; j < edge_mass; ++j)
		{
			size_t c = j;
			size_t d = Next(j, point_mass, closed);
			double left[1], right[1], delta[3], distance2, denominator, base, energy, scale;
			if (a == c || a == d || b == c || b == d)
				continue;
			ClosestSegments(points + a * 3, points + b * 3, points + c * 3,
				points + d * 3, spatial_epsilon, left, right, delta);
			distance2 = Dot3(delta, delta);
			measures->minimum_nonneighbor_segment_distance = fmin(
				measures->minimum_nonneighbor_segment_distance, sqrt(distance2));
			measures->nonneighbor_segment_pair_mass += 1;
			denominator = distance2 + spatial_epsilon2;
			base = thickness_scale2 / denominator;
			energy = weights->thickness * pow(base, thickness_alpha);
			measures->thickness_energy += energy;
			if (weights->thickness == 0.0)
				continue;
			scale = -2.0 * thickness_alpha * energy / denominator;
			AddGradient(gradient, a, scale * (1.0 - left[0]), delta);
			AddGradient(gradient, b, scale * left[0], delta);
			AddGradient(gradient, c, -scale * (1.0 - right[0]), delta);
			AddGradient(gradient, d, -scale * right[0], delta);
		}
	}
	if (!isfinite(measures->minimum_nonneighbor_segment_distance))
		measures->minimum_nonneighbor_segment_distance = 0.0;
	measures->total_energy = measures->anchor_energy + measures->strain_energy
		+ measures->bend_energy + measures->cusp_energy
		+ measures->tangent_point_energy + measures->thickness_energy;
}
