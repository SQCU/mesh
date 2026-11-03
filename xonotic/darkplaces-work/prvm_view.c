#include <setjmp.h>

static int PRVM_LeaveFunction(prvm_prog_t *prog);
static jmp_buf *prvm_view_recovery;

static int PRVM_ViewSkill(const char *name)
{
	static const char *fields[] = {"havocbot_keyboardskill", "bot_moveskill", "bot_dodgeskill",
		"bot_pingskill", "bot_weaponskill", "bot_aggresskill", "bot_rangepreference",
		"bot_aimskill", "bot_offsetskill", "bot_mouseskill", "bot_thinkskill", "bot_aiskill",
		"skill", "autocvar_skill", "bot_global_skill", "skill_save", "sk"};
	if (!strncmp(name, "plc_ob_f_", 9))
		name += 9;
	for (size_t i = 0; i < sizeof(fields) / sizeof(*fields); ++i)
		if (!strcmp(name, fields[i]))
			return 1;
	return 0;
}

int PRVM_ViewReadOnly(prvm_view_t *view, int entity, int offset)
{
	return entity < 0 ? view->global_readonly[offset] : view->field_readonly[offset];
}

void PRVM_ViewMask(prvm_view_t *view, prvm_view_page_t *page, int slot)
{
	if (PRVM_ViewReadOnly(view, page->entity, page->offset + slot))
	{
		page->residual[slot] = page->velocity[slot] = page->applied[slot] = 0;
		page->integer_applied[slot] = 0;
	}
}

static inline void PRVM_ViewAddress(prvm_prog_t *prog, int entity, int offset, int count)
{
	if (entity < -1 || entity >= prog->num_edicts || offset < 0 ||
		offset > (entity < 0 ? prog->numglobals : prog->entityfields) - count)
		prog->error_cmd("bot view address %i:%i+%i outside VM memory", entity, offset, count);
}

static void PRVM_ViewError(const char *format, ...)
{
	char message[1024];
	va_list args;
	va_start(args, format);
	dpvsnprintf(message, sizeof(message), format, args);
	va_end(args);
	Con_Printf("mesh bot view fault: %s; invocation unwinds and the next bot invocation continues\n", message);
	longjmp(*prvm_view_recovery, 1);
}

prvm_view_t *PRVM_ViewFor(prvm_prog_t *prog, int owner)
{
	prvm_view_t *view;
	if (owner < 0 || owner >= prog->num_edicts)
		prog->error_cmd("bot view owner %i outside %i edicts", owner, prog->num_edicts);
	if (!prog->view_session)
		prog->view_session = ((uint64_t)(Sys_DirtyTime() * 1000000) & 0xffffffffffffULL) | 1;
	for (view = prog->views; view && view->owner != owner; view = view->next) {}
	if (!view)
	{
		view = Mem_Alloc(prog->progs_mempool, sizeof(*view));
		view->owner = owner;
		view->generation = prog->edicts[owner].priv.required->generation;
		view->float_view = Mem_Alloc(prog->progs_mempool, prog->numglobals * sizeof(prvm_vec_t));
		view->integer_view = Mem_Alloc(prog->progs_mempool, prog->numglobals * sizeof(prvm_int_t));
		view->global_types = Mem_Alloc(prog->progs_mempool, prog->numglobals);
		view->global_readonly = Mem_Alloc(prog->progs_mempool, prog->numglobals + PRVM_VIEW_PAGE);
		view->global_invariant = Mem_Alloc(prog->progs_mempool, prog->numglobals + PRVM_VIEW_PAGE);
		view->field_readonly = Mem_Alloc(prog->progs_mempool, prog->entityfields + PRVM_VIEW_PAGE);
		memset(view->global_types, ev_float, prog->numglobals);
		for (int i = 0; i < prog->numglobaldefs; ++i)
		{
			ddef_t *d = &prog->globaldefs[i];
			int type = d->type & ~DEF_SAVEGLOBAL;
			for (int j = 0; j < prvm_type_size[type]; ++j)
			{
				view->global_types[d->ofs + j] = type == ev_vector ? ev_float : type;
				view->global_readonly[d->ofs + j] |= PRVM_ViewSkill(PRVM_GetString(prog, d->s_name));
			}
		}
		memcpy(view->global_invariant, view->global_readonly, prog->numglobals);
		for (int i = 0; i < prog->numfielddefs; ++i)
		{
			ddef_t *d = &prog->fielddefs[i];
			for (int j = 0; j < prvm_type_size[d->type & ~DEF_SAVEGLOBAL]; ++j)
				view->field_readonly[d->ofs + j] |= PRVM_ViewSkill(PRVM_GetString(prog, d->s_name));
		}
		view->tau = 1;
		view->records = Mem_Alloc(prog->progs_mempool, (prog->limit_edicts + 1) * sizeof(*view->records));
		view->tail = &view->pages;
		view->next = prog->views;
		prog->views = view;
	}
	if (view->generation != prog->edicts[owner].priv.required->generation)
	{
		for (prvm_view_page_t *page = view->pages; page; page = page->next)
		{
			memset(page->present, 0, offsetof(prvm_view_page_t, next) - offsetof(prvm_view_page_t, present));
			page->forcing = 0;
		}
		view->generation = prog->edicts[owner].priv.required->generation;
		view->sequence = view->applied_sequence = 0;
	}
	return view;
}

prvm_view_page_t *PRVM_ViewPage(prvm_prog_t *prog, prvm_view_t *view, int entity, int offset)
{
	PRVM_ViewAddress(prog, entity, offset, 1);
	prvm_view_page_t **record = view->records[entity + 1], *page;
	int width = entity < 0 ? prog->numglobals : prog->entityfields;
	if (!record)
	{
		record = Mem_Alloc(prog->progs_mempool, ((width + PRVM_VIEW_PAGE - 1) / PRVM_VIEW_PAGE) * sizeof(*record));
		view->records[entity + 1] = record;
	}
	page = record[offset / PRVM_VIEW_PAGE];
	if (!page)
	{
		page = Mem_Alloc(prog->progs_mempool, sizeof(*page));
		page->entity = entity;
		page->generation = entity < 0 ? 0 : prog->edicts[entity].priv.required->generation;
		page->offset = offset / PRVM_VIEW_PAGE * PRVM_VIEW_PAGE;
		record[offset / PRVM_VIEW_PAGE] = page;
		*view->tail = page;
		view->tail = &page->next;
	}
	if (entity >= 0 && page->generation != prog->edicts[entity].priv.required->generation)
	{
		memset(page->present, 0, offsetof(prvm_view_page_t, next) - offsetof(prvm_view_page_t, present));
		page->generation = prog->edicts[entity].priv.required->generation;
		page->forcing = 0;
	}
	return page;
}

void PRVM_ViewTime(prvm_view_t *view, double now)
{
	double age = max(0, now - view->source_time), held = min(age, view->duration);
	double tail = exp(-(age - held) / view->tau);
	view->decay = exp(-held / view->tau) * tail;
	view->gain = -view->tau * expm1(-held / view->tau) * tail;
}

static prvm_int_t PRVM_ViewInteger(double value)
{
	double magnitude = fmod(fabs(round(value)), ldexp(1, sizeof(prvm_int_t) * 8));
	prvm_uint_t word = (prvm_uint_t)magnitude;
	return value < 0 ? -word : word;
}

static inline void PRVM_ViewApply(prvm_view_t *view, prvm_view_page_t *page, int slot)
{
	PRVM_ViewMask(view, page, slot);
	if (page->forcing && page->applied_epoch[slot] != view->epoch)
	{
		double value = view->decay * page->residual[slot] + view->gain * page->velocity[slot];
		page->applied[slot] = value;
		page->integer_applied[slot] = PRVM_ViewInteger(value);
		page->applied_epoch[slot] = view->epoch;
	}
}
static inline prvm_eval_t PRVM_ViewReadFast(prvm_prog_t *prog, int entity, int offset, int type)
{
	prvm_eval_t result;
	prvm_view_t *view = prog->view;
	PRVM_ViewAddress(prog, entity, offset, type == ev_vector ? 3 : 1);
	prvm_vec_t *base = entity < 0 ? prog->globals.fp : prog->edicts[entity].fields.fp;
	int i, count = type == ev_vector ? 3 : 1;
	memset(&result, 0, sizeof(result));
	for (i = 0; i < count; ++i)
	{
		prvm_view_page_t **record = view->records[entity + 1];
		prvm_view_page_t *page = record ? record[(offset + i) / PRVM_VIEW_PAGE] : NULL;
		if (!page || (entity >= 0 && page->generation != prog->edicts[entity].priv.required->generation))
			page = PRVM_ViewPage(prog, view, entity, offset + i);
		prvm_eval_t *source = (prvm_eval_t *)(base + offset + i);
		int slot = (offset + i) % PRVM_VIEW_PAGE;
		prvm_vec_t residual;
		PRVM_ViewApply(view, page, slot);
		residual = page->applied[slot];
		int scalar_type = type == ev_vector ? ev_float : type;
		if (scalar_type == ev_void)
			scalar_type = entity < 0 ? view->global_types[offset + i] : page->type[slot];
		page->present[slot] = 1;
		page->type[slot] = scalar_type;
		page->state[slot] = scalar_type == ev_float ? source->_float : 0;
		page->integer_state[slot] = source->_int;
		page->read_time[slot] = PRVM_serverglobalfloat(time);
		result.ivector[i] = source->_int;
		if (residual != 0)
		{
			if (scalar_type == ev_float)
				result.vector[i] = source->_float + residual;
			else
				result.ivector[i] = (prvm_uint_t)source->_int + (prvm_uint_t)page->integer_applied[slot];
			++view->reads_modified;
		}
		++view->reads;
	}
	return result;
}

prvm_eval_t PRVM_ViewRead(prvm_prog_t *prog, int entity, int offset, int type)
{
	return PRVM_ViewReadFast(prog, entity, offset, type);
}


static inline void PRVM_ViewWriteTyped(prvm_prog_t *prog, mstatement_t *st, int offset, int count)
{
	int type = st->op == OP_LOAD_ENT || st->op == OP_STORE_ENT ? ev_entity :
		st->op == OP_LOAD_S || st->op == OP_STORE_S ? ev_string :
		st->op == OP_LOAD_FLD || st->op == OP_STORE_FLD ? ev_field :
		st->op == OP_LOAD_FNC || st->op == OP_STORE_FNC ? ev_function :
		st->op == OP_ADDRESS ? ev_pointer : ev_float;
	for (int i = 0; i < count; ++i)
	{
		prog->view->global_types[offset + i] = st->op == OP_RETURN || st->op == OP_DONE ?
			prog->view->global_types[st->operand[0] + i] : type;
		int copied = st->op == OP_RETURN || st->op == OP_DONE || (st->op >= OP_STORE_F && st->op <= OP_STORE_FNC);
		int loaded = st->op >= OP_LOAD_F && st->op <= OP_LOAD_FNC;
		prog->view->global_readonly[offset + i] = prog->view->global_invariant[offset + i] ||
			(copied && prog->view->global_readonly[st->operand[0] + i]) ||
			(loaded && prog->view->global_readonly[st->operand[1]]);
	}
}

static inline void PRVM_ViewStore(prvm_prog_t *prog, int address, const prvm_int_t *value, int count)
{
	for (int i = 0; i < count; ++i)
		if (!prog->view->field_readonly[(address + i) % prog->entityfields])
			((prvm_int_t *)prog->edictsfields)[address + i] = value[i];
}

static inline prvm_eval_t *PRVM_ViewOperand(prvm_prog_t *prog, int offset, int type)
{
	prvm_view_t *view = prog->view;
	int count = type == ev_vector ? 3 : 1, modified = 0;
	PRVM_ViewAddress(prog, -1, offset, count);
	prvm_view_page_t **record = view->records[0];
	prvm_view_page_t *first = record ? record[offset / PRVM_VIEW_PAGE] : NULL;
	if (first && !first->forcing && offset % PRVM_VIEW_PAGE + count <= PRVM_VIEW_PAGE)
	{
		for (int i = 0; i < count; ++i)
		{
			int word = offset + i, slot = word % PRVM_VIEW_PAGE;
			int scalar = type == ev_void ? view->global_types[word] : type == ev_vector ? ev_float : type;
			view->global_types[word] = first->type[slot] = scalar;
			first->present[slot] = 1;
			first->state[slot] = scalar == ev_float ? prog->globals.fp[word] : 0;
			first->integer_state[slot] = prog->globals.ip[word];
			first->read_time[slot] = PRVM_serverglobalfloat(time);
		}
		return (prvm_eval_t *)(prog->globals.fp + offset);
	}
	if (type == ev_void)
		type = view->global_types[offset];
	for (int i = 0; i < count; ++i)
	{
		int word = offset + i, slot = word % PRVM_VIEW_PAGE;
		prvm_view_page_t **record = view->records[0];
		prvm_view_page_t *page = record ? record[word / PRVM_VIEW_PAGE] : NULL;
		if (!page)
			page = PRVM_ViewPage(prog, view, -1, word);
		view->global_types[word] = page->type[slot] = type == ev_vector ? ev_float : type;
		page->present[slot] = 1;
		page->state[slot] = page->type[slot] == ev_float ? prog->globals.fp[word] : 0;
		page->integer_state[slot] = prog->globals.ip[word];
		page->read_time[slot] = PRVM_serverglobalfloat(time);
		PRVM_ViewApply(view, page, slot);
		if (type == ev_float || type == ev_vector)
		{
			((prvm_int_t *)view->float_view)[word] = prog->globals.ip[word];
			if (page->applied[slot] != 0)
			{
				view->float_view[word] = prog->globals.fp[word] + page->applied[slot];
				modified = 1;
			}
		}
		else
		{
			view->integer_view[word] = (prvm_uint_t)prog->globals.ip[word] + (prvm_uint_t)page->integer_applied[slot];
			modified |= page->integer_applied[slot] != 0;
		}
	}
	return (prvm_eval_t *)(!modified ? prog->globals.fp + offset : type == ev_float || type == ev_vector ?
		view->float_view + offset : (prvm_vec_t *)(view->integer_view + offset));
}

prvm_eval_t *PRVM_ViewGlobalRead(prvm_prog_t *prog, int offset, int type)
{
	return PRVM_ViewOperand(prog, offset, type);
}

static inline void PRVM_ViewOperands(prvm_prog_t *prog, mstatement_t *st, prvm_eval_t **a, prvm_eval_t **b)
{
	static const unsigned char types[][2] = {
		{255, 255},
		{ev_float, ev_float}, {ev_vector, ev_vector}, {ev_float, ev_vector}, {ev_vector, ev_float},
		{ev_float, ev_float}, {ev_float, ev_float}, {ev_vector, ev_vector}, {ev_float, ev_float}, {ev_vector, ev_vector},
		{ev_float, ev_float}, {ev_vector, ev_vector}, {ev_string, ev_string}, {ev_entity, ev_entity}, {ev_function, ev_function},
		{ev_float, ev_float}, {ev_vector, ev_vector}, {ev_string, ev_string}, {ev_entity, ev_entity}, {ev_function, ev_function},
		{ev_float, ev_float}, {ev_float, ev_float}, {ev_float, ev_float}, {ev_float, ev_float},
		{ev_entity, ev_field}, {ev_entity, ev_field}, {ev_entity, ev_field}, {ev_entity, ev_field}, {ev_entity, ev_field}, {ev_entity, ev_field},
		{ev_entity, ev_field},
		{ev_float, 255}, {ev_vector, 255}, {ev_string, 255}, {ev_entity, 255}, {ev_field, 255}, {ev_function, 255},
		{ev_float, ev_pointer}, {ev_vector, ev_pointer}, {ev_string, ev_pointer}, {ev_entity, ev_pointer}, {ev_field, ev_pointer}, {ev_function, ev_pointer},
		{255, 255}, {ev_float, 255}, {ev_vector, 255}, {ev_string, 255}, {ev_entity, 255}, {ev_function, 255},
		{ev_void, 255}, {ev_void, 255},
		{ev_function, 255}, {ev_function, 255}, {ev_function, 255}, {ev_function, 255}, {ev_function, 255}, {ev_function, 255}, {ev_function, 255}, {ev_function, 255}, {ev_function, 255},
		{ev_float, ev_function}, {255, 255}, {ev_float, ev_float}, {ev_float, ev_float}, {ev_float, ev_float}, {ev_float, ev_float}
	};
	if (types[st->op][0] != 255)
		*a = PRVM_ViewOperand(prog, st->operand[0], types[st->op][0]);
	if (types[st->op][1] != 255)
		*b = PRVM_ViewOperand(prog, st->operand[1], types[st->op][1]);
}

prvm_view_t *PRVM_ViewBegin(prvm_prog_t *prog, int owner)
{
	prvm_view_t *previous = prog->view, *view = PRVM_ViewFor(prog, owner);
	prog->view = view;
	if (previous != view)
	{
		PRVM_ViewTime(view, PRVM_serverglobalfloat(time));
		++view->calls;
		++view->epoch;
	}
	return previous;
}

void PRVM_ViewEnd(prvm_prog_t *prog, prvm_view_t *previous)
{
	if (prog->view == previous)
		return;

	if (prog->view->applied_sequence != prog->view->sequence)
	{
		prog->view->applied_time = PRVM_serverglobalfloat(time);
		prog->view->applied_source_time = prog->view->source_time;
	}
	prog->view->applied_sequence = prog->view->sequence;
	prog->view = previous;
}

void VM_mesh_view_run(prvm_prog_t *prog)
{
	int owner = PRVM_G_INT(OFS_PARM0);
	func_t function = PRVM_G_INT(OFS_PARM1);
	prvm_view_t *previous;
	prvm_view_t *view;
	jmp_buf recovery;
	jmp_buf *previous_recovery = prvm_view_recovery;
	void (*previous_error)(const char *, ...) = prog->error_cmd;
	int depth = prog->depth, statement = prog->xstatement, strings = prog->tempstringsbuf.cursize;
	prvm_int_t parameters[24];
	VM_SAFEPARMCOUNT(2, VM_mesh_view_run);
	memcpy(parameters, prog->globals.ip + OFS_PARM0, sizeof(parameters));
	previous = PRVM_ViewBegin(prog, owner);
	view = prog->view;
	prvm_view_recovery = &recovery;
	prog->error_cmd = PRVM_ViewError;
	if (!setjmp(recovery))
		prog->ExecuteProgram(prog, function, "mesh_view_run callback");
	else
	{
		++view->faults;
		view->fault_sequence = view->sequence;
		while (prog->depth > depth)
			PRVM_LeaveFunction(prog);
		prog->xstatement = statement;
		prog->tempstringsbuf.cursize = strings;
		memcpy(prog->globals.ip + OFS_PARM0, parameters, sizeof(parameters));
	}
	prog->error_cmd = previous_error;
	prvm_view_recovery = previous_recovery;
	PRVM_ViewEnd(prog, previous);
}

void VM_mesh_view_set(prvm_prog_t *prog)
{
	prvm_view_t *view;
	prvm_view_page_t *page;
	int entity, offset, slot;
	VM_SAFEPARMCOUNT(8, VM_mesh_view_set);
	for (int i = 1; i < 8; ++i)
		if (!isfinite(PRVM_G_FLOAT(OFS_PARM0 + 3 * i)))
		{
			VM_Warning(prog, "mesh_view_set: nonfinite numeric input; current view continues\n");
			return;
		}
	if (PRVM_G_INT(OFS_PARM0) < 0 || PRVM_G_INT(OFS_PARM0) >= prog->num_edicts ||
		PRVM_G_FLOAT(OFS_PARM1) != floor(PRVM_G_FLOAT(OFS_PARM1)) ||
		PRVM_G_FLOAT(OFS_PARM2) != floor(PRVM_G_FLOAT(OFS_PARM2)) ||
		PRVM_G_FLOAT(OFS_PARM1) < -1 || PRVM_G_FLOAT(OFS_PARM1) >= prog->num_edicts ||
		PRVM_G_FLOAT(OFS_PARM2) < 0 || PRVM_G_FLOAT(OFS_PARM2) >=
		(PRVM_G_FLOAT(OFS_PARM1) < 0 ? prog->numglobals : prog->entityfields) ||
		PRVM_G_FLOAT(OFS_PARM6) < 0 || PRVM_G_FLOAT(OFS_PARM7) <= 0)
	{
		VM_Warning(prog, "mesh_view_set: invalid memory extent or interval; current view continues\n");
		return;
	}
	view = PRVM_ViewFor(prog, PRVM_G_INT(OFS_PARM0));
	entity = PRVM_G_FLOAT(OFS_PARM1);
	offset = PRVM_G_FLOAT(OFS_PARM2);
	page = PRVM_ViewPage(prog, view, entity, offset);
	slot = offset % PRVM_VIEW_PAGE;
	page->residual[slot] = PRVM_G_FLOAT(OFS_PARM3);
	page->velocity[slot] = PRVM_G_FLOAT(OFS_PARM4);
	PRVM_ViewMask(view, page, slot);
	memset(page->applied_epoch, 0, sizeof(page->applied_epoch));
	page->forcing = 0;
	for (int i = 0; i < PRVM_VIEW_PAGE; ++i)
		page->forcing |= page->residual[i] != 0 || page->velocity[i] != 0;
	if (!page->forcing)
	{
		memset(page->applied, 0, sizeof(page->applied));
		memset(page->integer_applied, 0, sizeof(page->integer_applied));
	}
	view->source_time = PRVM_G_FLOAT(OFS_PARM5);
	view->duration = PRVM_G_FLOAT(OFS_PARM6);
	view->tau = PRVM_G_FLOAT(OFS_PARM7);
	++view->sequence;
}

void VM_mesh_view_stat(prvm_prog_t *prog)
{
	prvm_view_t *view = PRVM_ViewFor(prog, PRVM_G_INT(OFS_PARM0));
	int selector = PRVM_G_FLOAT(OFS_PARM1);
	VM_SAFEPARMCOUNT(2, VM_mesh_view_stat);
	switch (selector)
	{
	case 0: PRVM_G_FLOAT(OFS_RETURN) = view->applied_sequence; break;
	case 1: PRVM_G_FLOAT(OFS_RETURN) = view->reads; break;
	case 2: PRVM_G_FLOAT(OFS_RETURN) = view->reads_modified; break;
	case 4: PRVM_G_FLOAT(OFS_RETURN) = view->sequence; break;
	case 5: PRVM_G_FLOAT(OFS_RETURN) = view->source_time; break;
	case 6: PRVM_G_FLOAT(OFS_RETURN) = view->applied_time - view->applied_source_time; break;
	case 7: PRVM_G_FLOAT(OFS_RETURN) = prog->view_session >> 24; break;
	case 8: PRVM_G_FLOAT(OFS_RETURN) = prog->view_session & 0xffffff; break;
	case 9: PRVM_G_FLOAT(OFS_RETURN) = view->faults; break;
	case 10: PRVM_G_FLOAT(OFS_RETURN) = view->fault_sequence; break;
	default: PRVM_G_FLOAT(OFS_RETURN) = view->calls; break;
	}
}
