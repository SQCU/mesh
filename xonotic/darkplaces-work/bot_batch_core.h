#ifndef BOT_BATCH_CORE_H
#define BOT_BATCH_CORE_H

#include <math.h>
#include <stddef.h>

enum
{
	BOT_BATCH_MOVE_X,
	BOT_BATCH_MOVE_Y,
	BOT_BATCH_MOVE_Z,
	BOT_BATCH_ORIGIN_X,
	BOT_BATCH_ORIGIN_Y,
	BOT_BATCH_ORIGIN_Z,
	BOT_BATCH_DEST_X,
	BOT_BATCH_DEST_Y,
	BOT_BATCH_DEST_Z,
	BOT_BATCH_KEYBOARD_TIME,
	BOT_BATCH_MOVE_SKILL,
	BOT_BATCH_KEYBOARD_SKILL,
	BOT_BATCH_DUCK_TIME,
	BOT_BATCH_RANDOM,
	BOT_BATCH_KEYBOARD_X,
	BOT_BATCH_KEYBOARD_Y,
	BOT_BATCH_KEYBOARD_Z,
	BOT_BATCH_CROUCH,
	BOT_BATCH_FIELDS
};

static void BotBatchKernel(float *data, size_t stride, size_t rows, float now,
	float skill, float maxspeed, float trigger, float distance)
{
	float *mx = data + BOT_BATCH_MOVE_X * stride;
	float *my = data + BOT_BATCH_MOVE_Y * stride;
	float *mz = data + BOT_BATCH_MOVE_Z * stride;
	float *ox = data + BOT_BATCH_ORIGIN_X * stride;
	float *oy = data + BOT_BATCH_ORIGIN_Y * stride;
	float *oz = data + BOT_BATCH_ORIGIN_Z * stride;
	float *dx = data + BOT_BATCH_DEST_X * stride;
	float *dy = data + BOT_BATCH_DEST_Y * stride;
	float *dz = data + BOT_BATCH_DEST_Z * stride;
	float *kt = data + BOT_BATCH_KEYBOARD_TIME * stride;
	float *ms = data + BOT_BATCH_MOVE_SKILL * stride;
	float *ks = data + BOT_BATCH_KEYBOARD_SKILL * stride;
	float *duck = data + BOT_BATCH_DUCK_TIME * stride;
	float *noise = data + BOT_BATCH_RANDOM * stride;
	float *kx = data + BOT_BATCH_KEYBOARD_X * stride;
	float *ky = data + BOT_BATCH_KEYBOARD_Y * stride;
	float *kz = data + BOT_BATCH_KEYBOARD_Z * stride;
	float *crouch = data + BOT_BATCH_CROUCH * stride;
	size_t i;

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
	for (i = 0; i < rows; i++)
	{
		float sk = skill + ms[i];
		float x = mx[i] / maxspeed;
		float y = my[i] / maxspeed;
		float z = mz[i] / maxspeed;
		float blend;
		float span;

		kt[i] = fmaxf(kt[i] + 0.05f / fmaxf(1.0f, sk + ks[i])
			+ noise[i] * 0.025f / fmaxf(0.00025f, skill + ks[i]), now);
		if (x > trigger)
		{
			x = 1.0f;
			if (sk < 2.5f)
				y = 0.0f;
		}
		else if (x < -trigger && sk > 1.5f)
		{
			x = -1.0f;
			if (sk < 4.5f)
				y = 0.0f;
		}
		else
		{
			x = 0.0f;
			if (sk < 1.5f)
				y = 0.0f;
		}
		if (sk < 4.5f)
			z = 0.0f;
		y = y > trigger ? 1.0f : y < -trigger ? -1.0f : 0.0f;
		z = z > trigger ? 1.0f : z < -trigger ? -1.0f : 0.0f;
		if (x == 0.0f && y == 0.0f && z == 0.0f)
			kt[i] = fminf(kt[i], now + 0.2f);
		kx[i] = x * maxspeed;
		ky[i] = y * maxspeed;
		kz[i] = z * maxspeed;
		span = sqrtf((dx[i] - ox[i]) * (dx[i] - ox[i])
			+ (dy[i] - oy[i]) * (dy[i] - oy[i])
			+ (dz[i] - oz[i]) * (dz[i] - oz[i]));
		blend = fminf(fmaxf(span / distance, 0.0f), 1.0f);
		mx[i] += (kx[i] - mx[i]) * blend;
		my[i] += (ky[i] - my[i]) * blend;
		mz[i] += (kz[i] - mz[i]) * blend;
		crouch[i] = duck[i] > now;
	}
}

#endif
