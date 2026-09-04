#include "bot_batch_core.h"

typedef struct botbatchstats_s
{
	uint64_t rows;
	uint64_t kernel_rows;
	uint64_t waves;
	uint64_t cells;
	uint64_t working_sets;
	uint64_t peak_rows;
	uint64_t barriers;
	uint64_t total_rows;
	uint64_t total_kernel_rows;
	uint64_t wave_span;
	uint64_t input_coordinates;
	uint64_t output_coordinates;
	uint64_t buffer_bytes;
	float cell_extent;
	uint32_t causal_cells;
}
botbatchstats_t;

typedef struct vcellplanrow_s
{
	prvm_edict_t *actor;
	prvm_edict_t *client;
	uint32_t stable;
	uint32_t packed;
	int gx;
	int gy;
	int gz;
	int wave;
}
vcellplanrow_t;

typedef struct vcellplan_s
{
	vcellplanrow_t *rows;
	uint32_t count;
	uint32_t wave_span;
	uint32_t causal_cells;
	float cell_extent;
}
vcellplan_t;

static botbatchstats_t botbatchstats;

static int VCellPlanCompare(const void *first, const void *second)
{
	const vcellplanrow_t *a = (const vcellplanrow_t *)first;
	const vcellplanrow_t *b = (const vcellplanrow_t *)second;
	if (a->wave != b->wave)
		return a->wave < b->wave ? -1 : 1;
	if (a->gx != b->gx)
		return a->gx < b->gx ? -1 : 1;
	if (a->gy != b->gy)
		return a->gy < b->gy ? -1 : 1;
	if (a->gz != b->gz)
		return a->gz < b->gz ? -1 : 1;
	return a->stable < b->stable ? -1 : a->stable > b->stable;
}

static int BotBatchStableCompare(const void *first, const void *second)
{
	const vcellplanrow_t *a = (const vcellplanrow_t *)first;
	const vcellplanrow_t *b = (const vcellplanrow_t *)second;
	return a->stable < b->stable ? -1 : a->stable > b->stable;
}

static int VCellResidue(int value, int period)
{
	int residue = value % period;
	return residue < 0 ? residue + period : residue;
}

static int VCellWave(int gx, int gy, int gz, int period)
{
	return (VCellResidue(gx, period) * period + VCellResidue(gy, period)) * period + VCellResidue(gz, period);
}

static int BotBatchField(prvm_prog_t *prog, const char *name)
{
	int field = PRVM_ED_FindFieldOffset(prog, name);
	if (field < 0)
		VM_Warning(prog, "bot_controller_batch: field %s is unavailable in %s\n", name, prog->name);
	return field;
}

void VM_bot_controller_batch(prvm_prog_t *prog)
{
	uint32_t first, count, row, active, padded, wave_start;
	int pending, destination, randomfield, isbot, wavefield, csfield, movement, button5, keyboardtime, moveskill;
	int keyboardskill, ducktime, keyboard, period, raw_wave, color_span, cell_rank;
	float now, dt, skill, maxspeed, trigger, distance, motion, causal_distance;
	float *data;
	vcellplan_t plan;
	vcellplanrow_t *kernelrows;

	VM_SAFEPARMCOUNT(6, VM_bot_controller_batch);
	PRVM_G_FLOAT(OFS_RETURN) = 0;
	botbatchstats.rows = 0;
	botbatchstats.kernel_rows = 0;
	botbatchstats.waves = 0;
	botbatchstats.cells = 0;
	botbatchstats.working_sets = 0;
	botbatchstats.peak_rows = 0;
	botbatchstats.wave_span = 0;
	botbatchstats.input_coordinates = 0;
	botbatchstats.output_coordinates = 0;
	botbatchstats.buffer_bytes = 0;
	botbatchstats.cell_extent = 0;
	botbatchstats.causal_cells = 0;
	first = 1;
	count = (uint32_t)svs.maxclients;
	now = PRVM_G_FLOAT(OFS_PARM0);
	dt = PRVM_G_FLOAT(OFS_PARM1);
	skill = PRVM_G_FLOAT(OFS_PARM2);
	maxspeed = PRVM_G_FLOAT(OFS_PARM3);
	trigger = PRVM_G_FLOAT(OFS_PARM4);
	distance = PRVM_G_FLOAT(OFS_PARM5);
	if (count > (uint32_t)prog->max_edicts - first)
	{
		VM_Warning(prog, "bot_controller_batch: edicts %u+%u out of range in %s\n", first, count, prog->name);
		return;
	}
	pending = BotBatchField(prog, "bot_batch_pending");
	destination = BotBatchField(prog, "bot_batch_destination");
	randomfield = BotBatchField(prog, "bot_batch_random");
	isbot = BotBatchField(prog, "isbot");
	wavefield = BotBatchField(prog, "bot_batch_wave");
	csfield = BotBatchField(prog, "_cs");
	movement = BotBatchField(prog, "movement");
	button5 = BotBatchField(prog, "button5");
	keyboardtime = BotBatchField(prog, "havocbot_keyboardtime");
	moveskill = BotBatchField(prog, "bot_moveskill");
	keyboardskill = BotBatchField(prog, "havocbot_keyboardskill");
	ducktime = BotBatchField(prog, "havocbot_ducktime");
	keyboard = BotBatchField(prog, "havocbot_keyboard");
	if (pending < 0 || destination < 0 || randomfield < 0 || isbot < 0 || wavefield < 0 || csfield < 0 || movement < 0 || button5 < 0 || keyboardtime < 0 || moveskill < 0 || keyboardskill < 0 || ducktime < 0 || keyboard < 0)
		return;
	if (!count)
		return;
	plan.rows = (vcellplanrow_t *)Mem_Alloc(tempmempool, (size_t)count * sizeof(*plan.rows));
	plan.count = 0;
	plan.cell_extent = 1.0f;
	for (row = 0; row < count; row++)
	{
		prvm_edict_t *actor = PRVM_EDICT_NUM(first + row);
		float sx, sy, sz;
		if (!PRVM_EDICTFIELDFLOAT(actor, isbot))
			continue;
		plan.rows[plan.count].actor = actor;
		plan.rows[plan.count].client = NULL;
		plan.rows[plan.count].stable = row;
		sx = PRVM_serveredictvector(actor, maxs)[0] - PRVM_serveredictvector(actor, mins)[0];
		sy = PRVM_serveredictvector(actor, maxs)[1] - PRVM_serveredictvector(actor, mins)[1];
		sz = PRVM_serveredictvector(actor, maxs)[2] - PRVM_serveredictvector(actor, mins)[2];
		plan.cell_extent = fmaxf(plan.cell_extent, fmaxf(sx, fmaxf(sy, sz)));
		plan.count++;
	}
	if (!plan.count)
	{
		Mem_Free(plan.rows);
		return;
	}
	motion = fabsf(maxspeed * dt);
	causal_distance = plan.cell_extent + 2.0f * motion;
	plan.causal_cells = (uint32_t)ceilf(causal_distance / plan.cell_extent);
	period = (int)plan.causal_cells + 1;
	color_span = period * period * period;
	for (row = 0; row < plan.count; row++)
	{
		prvm_edict_t *actor = plan.rows[row].actor;
		plan.rows[row].gx = (int)floorf(PRVM_serveredictvector(actor, origin)[0] / plan.cell_extent);
		plan.rows[row].gy = (int)floorf(PRVM_serveredictvector(actor, origin)[1] / plan.cell_extent);
		plan.rows[row].gz = (int)floorf(PRVM_serveredictvector(actor, origin)[2] / plan.cell_extent);
		plan.rows[row].wave = VCellWave(plan.rows[row].gx, plan.rows[row].gy, plan.rows[row].gz, period);
	}
	qsort(plan.rows, plan.count, sizeof(*plan.rows), VCellPlanCompare);
	cell_rank = 0;
	for (row = 0; row < plan.count; row++)
	{
		if (row && plan.rows[row].gx == plan.rows[row - 1].gx && plan.rows[row].gy == plan.rows[row - 1].gy && plan.rows[row].gz == plan.rows[row - 1].gz)
			cell_rank++;
		else
			cell_rank = 0;
		plan.rows[row].wave += cell_rank * color_span;
	}
	qsort(plan.rows, plan.count, sizeof(*plan.rows), VCellPlanCompare);
	plan.wave_span = 0;
	raw_wave = -1;
	for (row = 0; row < plan.count; row++)
	{
		if (!row || plan.rows[row].wave != raw_wave)
		{
			raw_wave = plan.rows[row].wave;
			plan.wave_span++;
		}
		plan.rows[row].wave = (int)plan.wave_span - 1;
	}
	active = 0;
	for (row = 0; row < plan.count; row++)
	{
		PRVM_EDICTFIELDFLOAT(plan.rows[row].actor, wavefield) = plan.rows[row].wave;
	}
	for (row = 0; row < plan.count; row++)
		if (PRVM_EDICTFIELDFLOAT(plan.rows[row].actor, pending))
			active++;
	padded = active;
	data = active ? (float *)Mem_Alloc(tempmempool, (size_t)BOT_BATCH_FIELDS * padded * sizeof(float)) : NULL;
	kernelrows = active ? (vcellplanrow_t *)Mem_Alloc(tempmempool, (size_t)active * sizeof(*kernelrows)) : NULL;
	active = 0;
	for (row = 0; row < plan.count; row++)
	{
		prvm_edict_t *actor = plan.rows[row].actor;
		uint32_t csnum;
		prvm_edict_t *client;
		if (!PRVM_EDICTFIELDFLOAT(actor, pending))
			continue;
		csnum = (uint32_t)PRVM_EDICTFIELDEDICT(actor, csfield);
		if (!csnum || csnum >= (uint32_t)prog->max_edicts)
		{
			VM_Warning(prog, "bot_controller_batch: pending edict %u has invalid client state %u in %s; scalar fallback remains pending\n", first + plan.rows[row].stable, csnum, prog->name);
			continue;
		}
		client = PRVM_EDICT_NUM(csnum);
		kernelrows[active] = plan.rows[row];
		kernelrows[active].client = client;
		kernelrows[active].packed = active;
		data[BOT_BATCH_MOVE_X * padded + active] = PRVM_EDICTFIELDVECTOR(client, movement)[0];
		data[BOT_BATCH_MOVE_Y * padded + active] = PRVM_EDICTFIELDVECTOR(client, movement)[1];
		data[BOT_BATCH_MOVE_Z * padded + active] = PRVM_EDICTFIELDVECTOR(client, movement)[2];
		data[BOT_BATCH_ORIGIN_X * padded + active] = PRVM_serveredictvector(actor, origin)[0];
		data[BOT_BATCH_ORIGIN_Y * padded + active] = PRVM_serveredictvector(actor, origin)[1];
		data[BOT_BATCH_ORIGIN_Z * padded + active] = PRVM_serveredictvector(actor, origin)[2];
		data[BOT_BATCH_DEST_X * padded + active] = PRVM_EDICTFIELDVECTOR(actor, destination)[0];
		data[BOT_BATCH_DEST_Y * padded + active] = PRVM_EDICTFIELDVECTOR(actor, destination)[1];
		data[BOT_BATCH_DEST_Z * padded + active] = PRVM_EDICTFIELDVECTOR(actor, destination)[2];
		data[BOT_BATCH_KEYBOARD_TIME * padded + active] = PRVM_EDICTFIELDFLOAT(actor, keyboardtime);
		data[BOT_BATCH_MOVE_SKILL * padded + active] = PRVM_EDICTFIELDFLOAT(actor, moveskill);
		data[BOT_BATCH_KEYBOARD_SKILL * padded + active] = PRVM_EDICTFIELDFLOAT(actor, keyboardskill);
		data[BOT_BATCH_DUCK_TIME * padded + active] = PRVM_EDICTFIELDFLOAT(actor, ducktime);
		data[BOT_BATCH_RANDOM * padded + active] = PRVM_EDICTFIELDFLOAT(actor, randomfield);
		active++;
	}
	botbatchstats.input_coordinates = (uint64_t)active * 14;
	botbatchstats.output_coordinates = (uint64_t)active * 9;
	botbatchstats.buffer_bytes = (uint64_t)BOT_BATCH_FIELDS * padded * sizeof(float);
	wave_start = 0;
	while (wave_start < active)
	{
		uint32_t wave_end = wave_start + 1;
		uint32_t wave_rows;
		while (wave_end < active && kernelrows[wave_end].wave == kernelrows[wave_start].wave)
			wave_end++;
		wave_rows = wave_end - wave_start;
		BotBatchKernel(data + wave_start, padded, wave_rows, now, skill, maxspeed, trigger, distance);
		qsort(kernelrows + wave_start, wave_rows, sizeof(*kernelrows), BotBatchStableCompare);
		for (row = wave_start; row < wave_end; row++)
		{
			prvm_edict_t *actor = kernelrows[row].actor;
			prvm_edict_t *client = kernelrows[row].client;
			uint32_t packed = kernelrows[row].packed;
			PRVM_EDICTFIELDVECTOR(client, movement)[0] = data[BOT_BATCH_MOVE_X * padded + packed];
			PRVM_EDICTFIELDVECTOR(client, movement)[1] = data[BOT_BATCH_MOVE_Y * padded + packed];
			PRVM_EDICTFIELDVECTOR(client, movement)[2] = data[BOT_BATCH_MOVE_Z * padded + packed];
			PRVM_EDICTFIELDFLOAT(actor, keyboardtime) = data[BOT_BATCH_KEYBOARD_TIME * padded + packed];
			PRVM_EDICTFIELDVECTOR(actor, keyboard)[0] = data[BOT_BATCH_KEYBOARD_X * padded + packed];
			PRVM_EDICTFIELDVECTOR(actor, keyboard)[1] = data[BOT_BATCH_KEYBOARD_Y * padded + packed];
			PRVM_EDICTFIELDVECTOR(actor, keyboard)[2] = data[BOT_BATCH_KEYBOARD_Z * padded + packed];
			if (data[BOT_BATCH_CROUCH * padded + packed])
				PRVM_EDICTFIELDFLOAT(client, button5) = 1;
			PRVM_EDICTFIELDFLOAT(actor, pending) = 0;
		}
		botbatchstats.barriers++;
		botbatchstats.working_sets++;
		botbatchstats.peak_rows = max(botbatchstats.peak_rows, wave_rows);
		wave_start = wave_end;
	}
	for (row = 0; row < plan.count; row++)
	{
		if (!row || plan.rows[row].gx != plan.rows[row - 1].gx || plan.rows[row].gy != plan.rows[row - 1].gy || plan.rows[row].gz != plan.rows[row - 1].gz)
			botbatchstats.cells++;
		if (!row || plan.rows[row].wave != plan.rows[row - 1].wave)
		{
			uint32_t wave_end = row + 1;
			while (wave_end < plan.count && plan.rows[wave_end].wave == plan.rows[row].wave)
				wave_end++;
			botbatchstats.peak_rows = max(botbatchstats.peak_rows, wave_end - row);
			botbatchstats.waves++;
		}
	}
	if (kernelrows)
		Mem_Free(kernelrows);
	if (data)
		Mem_Free(data);
	botbatchstats.rows = plan.count;
	botbatchstats.kernel_rows = active;
	botbatchstats.total_rows += plan.count;
	botbatchstats.total_kernel_rows += active;
	botbatchstats.wave_span = plan.wave_span;
	botbatchstats.cell_extent = plan.cell_extent;
	botbatchstats.causal_cells = plan.causal_cells;
	Mem_Free(plan.rows);
	PRVM_G_FLOAT(OFS_RETURN) = plan.wave_span;
}

void VM_bot_controller_stat(prvm_prog_t *prog)
{
	int selector;
	VM_SAFEPARMCOUNT(1, VM_bot_controller_stat);
	selector = (int)PRVM_G_FLOAT(OFS_PARM0);
	switch (selector)
	{
	case 0: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.rows; break;
	case 1: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.kernel_rows; break;
	case 2: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.waves; break;
	case 3: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.cells; break;
	case 4: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.barriers; break;
	case 5: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.total_rows; break;
	case 6: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.working_sets; break;
	case 7: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.peak_rows; break;
	case 8: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.wave_span; break;
	case 9: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.cell_extent; break;
	case 10: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.causal_cells; break;
	case 11: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.total_kernel_rows; break;
	case 12: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.input_coordinates; break;
	case 13: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.output_coordinates; break;
	case 14: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.buffer_bytes; break;
	default: PRVM_G_FLOAT(OFS_RETURN) = botbatchstats.total_rows; break;
	}
}
