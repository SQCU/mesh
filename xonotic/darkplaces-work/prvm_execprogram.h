#ifdef PRVM_VIEWINTERPRETER
#undef OPA
#undef OPB
#define OPA view_a
#define OPB view_b
#endif
#ifdef PRVM_VIEWINTERPRETER
#define VIEW_GLOBAL_WRITE(offset, count) PRVM_ViewWriteTyped(prog, st, offset, count)
#else
#define VIEW_GLOBAL_WRITE(offset, count) ((void)0)
#endif
#define OPB_WRITE ((prvm_eval_t *)&prog->globals.fp[st->operand[1]])

#define ADVANCE_PROFILE_BEFORE_JUMP() \
	prog->xfunction->profile += (st - startst); \
	if (prvm_statementprofiling.integer || (prvm_coverage.integer & 4)) { \
		                                                         \
		while (++startst <= st) { \
			if (prog->statement_profile[startst - cached_statements]++ == 0 && (prvm_coverage.integer & 4)) \
				PRVM_StatementCoverageEvent(prog, prog->xfunction, startst - cached_statements); \
		} \
		                                                       \
	}

#ifdef PRVMTIMEPROFILING
#define PRE_ERROR() \
	ADVANCE_PROFILE_BEFORE_JUMP(); \
	prog->xstatement = st - cached_statements; \
	tm = Sys_DirtyTime(); \
	prog->xfunction->tprofile += (tm - starttm >= 0 && tm - starttm < 1800) ? (tm - starttm) : 0; \
	startst = st; \
	starttm = tm
#else
#define PRE_ERROR() \
	ADVANCE_PROFILE_BEFORE_JUMP(); \
	prog->xstatement = st - cached_statements; \
	startst = st
#endif

#if HAVE_COMPUTED_GOTOS && !(PRVMSLOWINTERPRETER || PRVMTIMEPROFILING)

# define USE_COMPUTED_GOTOS 1
#endif

#if USE_COMPUTED_GOTOS
#ifdef PRVM_VIEWINTERPRETER
#define HANDLE_OPCODE(opcode) handle_view_##opcode
#else
#define HANDLE_OPCODE(opcode) handle_##opcode
#endif
    const static void *dispatchtable[] = {
	&&HANDLE_OPCODE(OP_DONE),
	&&HANDLE_OPCODE(OP_MUL_F),
	&&HANDLE_OPCODE(OP_MUL_V),
	&&HANDLE_OPCODE(OP_MUL_FV),
	&&HANDLE_OPCODE(OP_MUL_VF),
	&&HANDLE_OPCODE(OP_DIV_F),
	&&HANDLE_OPCODE(OP_ADD_F),
	&&HANDLE_OPCODE(OP_ADD_V),
	&&HANDLE_OPCODE(OP_SUB_F),
	&&HANDLE_OPCODE(OP_SUB_V),

	&&HANDLE_OPCODE(OP_EQ_F),
	&&HANDLE_OPCODE(OP_EQ_V),
	&&HANDLE_OPCODE(OP_EQ_S),
	&&HANDLE_OPCODE(OP_EQ_E),
	&&HANDLE_OPCODE(OP_EQ_FNC),

	&&HANDLE_OPCODE(OP_NE_F),
	&&HANDLE_OPCODE(OP_NE_V),
	&&HANDLE_OPCODE(OP_NE_S),
	&&HANDLE_OPCODE(OP_NE_E),
	&&HANDLE_OPCODE(OP_NE_FNC),

	&&HANDLE_OPCODE(OP_LE),
	&&HANDLE_OPCODE(OP_GE),
	&&HANDLE_OPCODE(OP_LT),
	&&HANDLE_OPCODE(OP_GT),

	&&HANDLE_OPCODE(OP_LOAD_F),
	&&HANDLE_OPCODE(OP_LOAD_V),
	&&HANDLE_OPCODE(OP_LOAD_S),
	&&HANDLE_OPCODE(OP_LOAD_ENT),
	&&HANDLE_OPCODE(OP_LOAD_FLD),
	&&HANDLE_OPCODE(OP_LOAD_FNC),

	&&HANDLE_OPCODE(OP_ADDRESS),

	&&HANDLE_OPCODE(OP_STORE_F),
	&&HANDLE_OPCODE(OP_STORE_V),
	&&HANDLE_OPCODE(OP_STORE_S),
	&&HANDLE_OPCODE(OP_STORE_ENT),
	&&HANDLE_OPCODE(OP_STORE_FLD),
	&&HANDLE_OPCODE(OP_STORE_FNC),

	&&HANDLE_OPCODE(OP_STOREP_F),
	&&HANDLE_OPCODE(OP_STOREP_V),
	&&HANDLE_OPCODE(OP_STOREP_S),
	&&HANDLE_OPCODE(OP_STOREP_ENT),
	&&HANDLE_OPCODE(OP_STOREP_FLD),
	&&HANDLE_OPCODE(OP_STOREP_FNC),

	&&HANDLE_OPCODE(OP_RETURN),
	&&HANDLE_OPCODE(OP_NOT_F),
	&&HANDLE_OPCODE(OP_NOT_V),
	&&HANDLE_OPCODE(OP_NOT_S),
	&&HANDLE_OPCODE(OP_NOT_ENT),
	&&HANDLE_OPCODE(OP_NOT_FNC),
	&&HANDLE_OPCODE(OP_IF),
	&&HANDLE_OPCODE(OP_IFNOT),
	&&HANDLE_OPCODE(OP_CALL0),
	&&HANDLE_OPCODE(OP_CALL1),
	&&HANDLE_OPCODE(OP_CALL2),
	&&HANDLE_OPCODE(OP_CALL3),
	&&HANDLE_OPCODE(OP_CALL4),
	&&HANDLE_OPCODE(OP_CALL5),
	&&HANDLE_OPCODE(OP_CALL6),
	&&HANDLE_OPCODE(OP_CALL7),
	&&HANDLE_OPCODE(OP_CALL8),
	&&HANDLE_OPCODE(OP_STATE),
	&&HANDLE_OPCODE(OP_GOTO),
	&&HANDLE_OPCODE(OP_AND),
	&&HANDLE_OPCODE(OP_OR),

	&&HANDLE_OPCODE(OP_BITAND),
	&&HANDLE_OPCODE(OP_BITOR)
	    };
#ifdef PRVM_VIEWINTERPRETER
#define DISPATCH_OPCODE() do { ++st; PRVM_ViewOperands(prog, st, &view_a, &view_b); goto *dispatchtable[st->op]; } while (0)
#else
#define DISPATCH_OPCODE() \
    goto *dispatchtable[(++st)->op]
#endif

    DISPATCH_OPCODE();
#else
#define DISPATCH_OPCODE() break
#define HANDLE_OPCODE(opcode) case opcode

#if PRVMSLOWINTERPRETER
		{
			if (prog->watch_global_type != ev_void)
			{
				prvm_eval_t *g = PRVM_GLOBALFIELDVALUE(prog->watch_global);
				prog->xstatement = st + 1 - cached_statements;
				PRVM_Watchpoint(prog, 1, "Global watchpoint hit by engine", prog->watch_global_type, &prog->watch_global_value, g);
			}
			if (prog->watch_field_type != ev_void && prog->watch_edict < prog->max_edicts)
			{
				prvm_eval_t *g = PRVM_EDICTFIELDVALUE(prog->edicts + prog->watch_edict, prog->watch_field);
				prog->xstatement = st + 1 - cached_statements;
				PRVM_Watchpoint(prog, 1, "Entityfield watchpoint hit by engine", prog->watch_field_type, &prog->watch_edictfield_value, g);
			}
		}
#endif

		while (1)
		{
			st++;
#ifdef PRVM_VIEWINTERPRETER
			PRVM_ViewOperands(prog, st, &view_a, &view_b);
#endif
#endif

#if !USE_COMPUTED_GOTOS

#if PRVMSLOWINTERPRETER
			if (prog->trace)
				PRVM_PrintStatement(prog, st);
			if (prog->break_statement >= 0)
				if ((st - cached_statements) == prog->break_statement)
				{
					prog->xstatement = st - cached_statements;
					PRVM_Breakpoint(prog, prog->break_stack_index, "Breakpoint hit");
				}
#endif
			switch (st->op)
			{
#endif
			HANDLE_OPCODE(OP_ADD_F):
				OPC->_float = OPA->_float + OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_ADD_V):
				OPC->vector[0] = OPA->vector[0] + OPB->vector[0];
				OPC->vector[1] = OPA->vector[1] + OPB->vector[1];
				OPC->vector[2] = OPA->vector[2] + OPB->vector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 3);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_SUB_F):
				OPC->_float = OPA->_float - OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_SUB_V):
				OPC->vector[0] = OPA->vector[0] - OPB->vector[0];
				OPC->vector[1] = OPA->vector[1] - OPB->vector[1];
				OPC->vector[2] = OPA->vector[2] - OPB->vector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 3);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_MUL_F):
				OPC->_float = OPA->_float * OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_MUL_V):
				OPC->_float = OPA->vector[0]*OPB->vector[0] + OPA->vector[1]*OPB->vector[1] + OPA->vector[2]*OPB->vector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_MUL_FV):
				tempfloat = OPA->_float;
				OPC->vector[0] = tempfloat * OPB->vector[0];
				OPC->vector[1] = tempfloat * OPB->vector[1];
				OPC->vector[2] = tempfloat * OPB->vector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 3);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_MUL_VF):
				tempfloat = OPB->_float;
				OPC->vector[0] = tempfloat * OPA->vector[0];
				OPC->vector[1] = tempfloat * OPA->vector[1];
				OPC->vector[2] = tempfloat * OPA->vector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 3);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_DIV_F):
				if( OPB->_float != 0.0f )
				{
					OPC->_float = OPA->_float / OPB->_float;
				}
				else
				{
					if (developer.integer)
					{
						PRE_ERROR();
						VM_Warning(prog, "Attempted division by zero in %s\n", prog->name );
					}
					OPC->_float = 0.0f;
				}
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_BITAND):
				OPC->_float = (prvm_int_t)OPA->_float & (prvm_int_t)OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_BITOR):
				OPC->_float = (prvm_int_t)OPA->_float | (prvm_int_t)OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_GE):
				OPC->_float = OPA->_float >= OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_LE):
				OPC->_float = OPA->_float <= OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_GT):
				OPC->_float = OPA->_float > OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_LT):
				OPC->_float = OPA->_float < OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_AND):
				OPC->_float = FLOAT_IS_TRUE_FOR_INT(OPA->_int) && FLOAT_IS_TRUE_FOR_INT(OPB->_int);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_OR):
				OPC->_float = FLOAT_IS_TRUE_FOR_INT(OPA->_int) || FLOAT_IS_TRUE_FOR_INT(OPB->_int);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NOT_F):
				OPC->_float = !FLOAT_IS_TRUE_FOR_INT(OPA->_int);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NOT_V):
				OPC->_float = !OPA->vector[0] && !OPA->vector[1] && !OPA->vector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NOT_S):
				OPC->_float = !OPA->string || !*PRVM_GetString(prog, OPA->string);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NOT_FNC):
				OPC->_float = !OPA->function;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NOT_ENT):
				OPC->_float = (OPA->edict == 0);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_EQ_F):
				OPC->_float = OPA->_float == OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_EQ_V):
				OPC->_float = (OPA->vector[0] == OPB->vector[0]) && (OPA->vector[1] == OPB->vector[1]) && (OPA->vector[2] == OPB->vector[2]);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_EQ_S):
				OPC->_float = !strcmp(PRVM_GetString(prog, OPA->string),PRVM_GetString(prog, OPB->string));
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_EQ_E):
				OPC->_float = OPA->_int == OPB->_int;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_EQ_FNC):
				OPC->_float = OPA->function == OPB->function;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NE_F):
				OPC->_float = OPA->_float != OPB->_float;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NE_V):
				OPC->_float = (OPA->vector[0] != OPB->vector[0]) || (OPA->vector[1] != OPB->vector[1]) || (OPA->vector[2] != OPB->vector[2]);
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NE_S):
				OPC->_float = strcmp(PRVM_GetString(prog, OPA->string),PRVM_GetString(prog, OPB->string));
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NE_E):
				OPC->_float = OPA->_int != OPB->_int;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_NE_FNC):
				OPC->_float = OPA->function != OPB->function;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_STORE_F):
			HANDLE_OPCODE(OP_STORE_ENT):
			HANDLE_OPCODE(OP_STORE_FLD):
			HANDLE_OPCODE(OP_STORE_S):
			HANDLE_OPCODE(OP_STORE_FNC):
				OPB_WRITE->_int = OPA->_int;
				VIEW_GLOBAL_WRITE(st->operand[1], 1);
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_STORE_V):
				OPB_WRITE->ivector[0] = OPA->ivector[0];
				OPB_WRITE->ivector[1] = OPA->ivector[1];
				OPB_WRITE->ivector[2] = OPA->ivector[2];
				VIEW_GLOBAL_WRITE(st->operand[1], 3);
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_STOREP_F):
			HANDLE_OPCODE(OP_STOREP_ENT):
			HANDLE_OPCODE(OP_STOREP_FLD):
			HANDLE_OPCODE(OP_STOREP_S):
			HANDLE_OPCODE(OP_STOREP_FNC):
				if ((prvm_uint_t)OPB->_int - cached_entityfields >= cached_entityfieldsarea_entityfields)
				{
					if ((prvm_uint_t)OPB->_int >= cached_entityfieldsarea)
					{
						PRE_ERROR();
						prog->error_cmd("%s attempted to write to an out of bounds edict (%i)", prog->name, (int)OPB->_int);
						goto cleanup;
					}
					if ((prvm_uint_t)OPB->_int < cached_entityfields && !cached_allowworldwrites)
					{
						PRE_ERROR();
						VM_Warning(prog, "assignment to world.%s (field %i) in %s\n", PRVM_GetString(prog, PRVM_ED_FieldAtOfs(prog, OPB->_int)->s_name), (int)OPB->_int, prog->name);
					}
				}
				ptr = (prvm_eval_t *)(cached_edictsfields + OPB->_int);
#ifdef PRVM_VIEWINTERPRETER
				PRVM_ViewStore(prog, OPB->_int, &OPA->_int, 1);
#else
				ptr->_int = OPA->_int;
#endif
				DISPATCH_OPCODE();
			HANDLE_OPCODE(OP_STOREP_V):
				if ((prvm_uint_t)OPB->_int - cached_entityfields > (prvm_uint_t)cached_entityfieldsarea_entityfields_3)
				{
					if ((prvm_uint_t)OPB->_int > cached_entityfieldsarea_3)
					{
						PRE_ERROR();
						prog->error_cmd("%s attempted to write to an out of bounds edict (%i)", prog->name, (int)OPB->_int);
						goto cleanup;
					}
					if ((prvm_uint_t)OPB->_int < cached_entityfields && !cached_allowworldwrites)
					{
						PRE_ERROR();
						VM_Warning(prog, "assignment to world.%s (field %i) in %s\n", PRVM_GetString(prog, PRVM_ED_FieldAtOfs(prog, OPB->_int)->s_name), (int)OPB->_int, prog->name);
					}
				}
				ptr = (prvm_eval_t *)(cached_edictsfields + OPB->_int);
#ifdef PRVM_VIEWINTERPRETER
				PRVM_ViewStore(prog, OPB->_int, OPA->ivector, 3);
#else
				ptr->ivector[0] = OPA->ivector[0];
				ptr->ivector[1] = OPA->ivector[1];
				ptr->ivector[2] = OPA->ivector[2];
#endif
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_ADDRESS):
				if ((prvm_uint_t)OPA->edict >= cached_max_edicts)
				{
					PRE_ERROR();
					prog->error_cmd("%s Progs attempted to address an out of bounds edict number", prog->name);
					goto cleanup;
				}
				if ((prvm_uint_t)OPB->_int >= cached_entityfields)
				{
					PRE_ERROR();
					prog->error_cmd("%s attempted to address an invalid field (%i) in an edict", prog->name, (int)OPB->_int);
					goto cleanup;
				}
#if 0
				if (OPA->edict == 0 && !cached_allowworldwrites)
				{
					PRE_ERROR();
					prog->error_cmd("forbidden assignment to null/world entity in %s", prog->name);
					goto cleanup;
				}
#endif
				OPC->_int = OPA->edict * cached_entityfields + OPB->_int;
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_LOAD_F):
			HANDLE_OPCODE(OP_LOAD_FLD):
			HANDLE_OPCODE(OP_LOAD_ENT):
			HANDLE_OPCODE(OP_LOAD_S):
			HANDLE_OPCODE(OP_LOAD_FNC):
				if ((prvm_uint_t)OPA->edict >= cached_max_edicts)
				{
					PRE_ERROR();
					prog->error_cmd("%s Progs attempted to read an out of bounds edict number", prog->name);
					goto cleanup;
				}
				if ((prvm_uint_t)OPB->_int >= cached_entityfields)
				{
					PRE_ERROR();
					prog->error_cmd("%s attempted to read an invalid field in an edict (%i)", prog->name, (int)OPB->_int);
					goto cleanup;
				}
				ed = PRVM_PROG_TO_EDICT(OPA->edict);
				#ifdef PRVM_VIEWINTERPRETER
				view_field = PRVM_ViewReadFast(prog, OPA->edict, OPB->_int,
					st->op == OP_LOAD_F ? ev_float : st->op == OP_LOAD_S ? ev_string :
					st->op == OP_LOAD_ENT ? ev_entity : st->op == OP_LOAD_FLD ? ev_field : ev_function);
				OPC->_int = view_field._int;
#else
				OPC->_int = ((prvm_eval_t *)(ed->fields.ip + OPB->_int))->_int;
#endif
				VIEW_GLOBAL_WRITE(st->operand[2], 1);
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_LOAD_V):
				if ((prvm_uint_t)OPA->edict >= cached_max_edicts)
				{
					PRE_ERROR();
					prog->error_cmd("%s Progs attempted to read an out of bounds edict number", prog->name);
					goto cleanup;
				}
				if ((prvm_uint_t)OPB->_int > cached_entityfields_3)
				{
					PRE_ERROR();
					prog->error_cmd("%s attempted to read an invalid field in an edict (%i)", prog->name, (int)OPB->_int);
					goto cleanup;
				}
				ed = PRVM_PROG_TO_EDICT(OPA->edict);
				#ifdef PRVM_VIEWINTERPRETER
				view_field = PRVM_ViewReadFast(prog, OPA->edict, OPB->_int, ev_vector);
				ptr = &view_field;
#else
				ptr = (prvm_eval_t *)(ed->fields.ip + OPB->_int);
#endif
				OPC->ivector[0] = ptr->ivector[0];
				OPC->ivector[1] = ptr->ivector[1];
				OPC->ivector[2] = ptr->ivector[2];
				VIEW_GLOBAL_WRITE(st->operand[2], 3);
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_IFNOT):
				if(!FLOAT_IS_TRUE_FOR_INT(OPA->_int))

				{
					ADVANCE_PROFILE_BEFORE_JUMP();
					st = cached_statements + st->jumpabsolute - 1;
					startst = st;

					if (++jumpcount == 10000000 && (prvm_view_recovery || prvm_runawaycheck))
					{
						prog->xstatement = st - cached_statements;
						PRVM_Profile(prog, 1<<30, 1000000, 0);
						prog->error_cmd("%s runaway loop counter hit limit of %d jumps\ntip: read above for list of most-executed functions", prog->name, jumpcount);
					}
				}
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_IF):
				if(FLOAT_IS_TRUE_FOR_INT(OPA->_int))

				{
					ADVANCE_PROFILE_BEFORE_JUMP();
					st = cached_statements + st->jumpabsolute - 1;
					startst = st;

					if (++jumpcount == 10000000 && (prvm_view_recovery || prvm_runawaycheck))
					{
						prog->xstatement = st - cached_statements;
						PRVM_Profile(prog, 1<<30, 0.01, 0);
						prog->error_cmd("%s runaway loop counter hit limit of %d jumps\ntip: read above for list of most-executed functions", prog->name, jumpcount);
					}
				}
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_GOTO):
				ADVANCE_PROFILE_BEFORE_JUMP();
				st = cached_statements + st->jumpabsolute - 1;
				startst = st;

				if (++jumpcount == 10000000 && (prvm_view_recovery || prvm_runawaycheck))
				{
					prog->xstatement = st - cached_statements;
					PRVM_Profile(prog, 1<<30, 0.01, 0);
					prog->error_cmd("%s runaway loop counter hit limit of %d jumps\ntip: read above for list of most-executed functions", prog->name, jumpcount);
				}
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_CALL0):
			HANDLE_OPCODE(OP_CALL1):
			HANDLE_OPCODE(OP_CALL2):
			HANDLE_OPCODE(OP_CALL3):
			HANDLE_OPCODE(OP_CALL4):
			HANDLE_OPCODE(OP_CALL5):
			HANDLE_OPCODE(OP_CALL6):
			HANDLE_OPCODE(OP_CALL7):
			HANDLE_OPCODE(OP_CALL8):
#ifdef PRVMTIMEPROFILING
				tm = Sys_DirtyTime();
				prog->xfunction->tprofile += (tm - starttm >= 0 && tm - starttm < 1800) ? (tm - starttm) : 0;
				starttm = tm;
#endif
				ADVANCE_PROFILE_BEFORE_JUMP();
				startst = st;
				prog->xstatement = st - cached_statements;
				prog->argc = st->op - OP_CALL0;
				if (!OPA->function)
				{
					prog->error_cmd("NULL function in %s", prog->name);
				}

				if(!OPA->function || OPA->function < 0 || OPA->function >= prog->numfunctions)
				{
					PRE_ERROR();
					prog->error_cmd("%s CALL outside the program", prog->name);
					goto cleanup;
				}

				enterfunc = &prog->functions[OPA->function];
				if (enterfunc->callcount++ == 0 && (prvm_coverage.integer & 1))
					PRVM_FunctionCoverageEvent(prog, enterfunc);

				if (enterfunc->first_statement < 0)
				{

					int builtinnumber = -enterfunc->first_statement;
					prog->xfunction->builtinsprofile++;
					if (builtinnumber < prog->numbuiltins && prog->builtins[builtinnumber])
					{
						prog->builtins[builtinnumber](prog);
#ifdef PRVMTIMEPROFILING
						tm = Sys_DirtyTime();
						enterfunc->tprofile += (tm - starttm >= 0 && tm - starttm < 1800) ? (tm - starttm) : 0;
						prog->xfunction->tbprofile += (tm - starttm >= 0 && tm - starttm < 1800) ? (tm - starttm) : 0;
						starttm = tm;
#endif

						cached_edictsfields = prog->edictsfields;
						cached_entityfields = prog->entityfields;
						cached_entityfields_3 = prog->entityfields - 3;
						cached_entityfieldsarea = prog->entityfieldsarea;
						cached_entityfieldsarea_entityfields = prog->entityfieldsarea - prog->entityfields;
						cached_entityfieldsarea_3 = prog->entityfieldsarea - 3;
						cached_entityfieldsarea_entityfields_3 = prog->entityfieldsarea - prog->entityfields - 3;
						cached_max_edicts = prog->max_edicts;

						if (prog->trace != cachedpr_trace)
							goto chooseexecprogram;
					}
					else
						prog->error_cmd("No such builtin #%i in %s; most likely cause: outdated engine build. Try updating!", builtinnumber, prog->name);
				}
				else
					st = cached_statements + PRVM_EnterFunction(prog, enterfunc);
				startst = st;
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_DONE):
			HANDLE_OPCODE(OP_RETURN):
#ifdef PRVMTIMEPROFILING
				tm = Sys_DirtyTime();
				prog->xfunction->tprofile += (tm - starttm >= 0 && tm - starttm < 1800) ? (tm - starttm) : 0;
				starttm = tm;
#endif
				ADVANCE_PROFILE_BEFORE_JUMP();
				prog->xstatement = st - cached_statements;

#ifdef PRVM_VIEWINTERPRETER
				for (int i = 0; i < 3; ++i)
				{
					view_field = *PRVM_ViewOperand(prog, st->operand[0] + i, ev_void);
					prog->globals.ip[OFS_RETURN + i] = view_field._int;
				}
#else
				prog->globals.ip[OFS_RETURN  ] = prog->globals.ip[st->operand[0]  ];
				prog->globals.ip[OFS_RETURN+1] = prog->globals.ip[st->operand[0]+1];
				prog->globals.ip[OFS_RETURN+2] = prog->globals.ip[st->operand[0]+2];
#endif

				VIEW_GLOBAL_WRITE(OFS_RETURN, 3);
				st = cached_statements + PRVM_LeaveFunction(prog);
				startst = st;
				if (prog->depth <= exitdepth)
					goto cleanup;
				DISPATCH_OPCODE();

			HANDLE_OPCODE(OP_STATE):
				if(cached_flag & PRVM_OP_STATE)
				{
					ed = PRVM_PROG_TO_EDICT(PRVM_gameglobaledict(self));
					PRVM_gameedictfloat(ed,nextthink) = PRVM_gameglobalfloat(time) + 0.1;
					PRVM_gameedictfloat(ed,frame) = OPA->_float;
					PRVM_gameedictfunction(ed,think) = OPB->function;
				}
				else
				{
					PRE_ERROR();
					prog->xstatement = st - cached_statements;
					prog->error_cmd("OP_STATE not supported by %s", prog->name);
				}
				DISPATCH_OPCODE();

#if !USE_COMPUTED_GOTOS
			default:
				PRE_ERROR();
				prog->error_cmd("Bad opcode %i in %s", st->op, prog->name);
				goto cleanup;
			}
#if PRVMSLOWINTERPRETER
			{
				if (prog->watch_global_type != ev_void)
				{
					prvm_eval_t *g = PRVM_GLOBALFIELDVALUE(prog->watch_global);
					prog->xstatement = st - cached_statements;
					PRVM_Watchpoint(prog, 0, "Global watchpoint hit", prog->watch_global_type, &prog->watch_global_value, g);
				}
				if (prog->watch_field_type != ev_void && prog->watch_edict < prog->max_edicts)
				{
					prvm_eval_t *g = PRVM_EDICTFIELDVALUE(prog->edicts + prog->watch_edict, prog->watch_field);
					prog->xstatement = st - cached_statements;
					PRVM_Watchpoint(prog, 0, "Entityfield watchpoint hit", prog->watch_field_type, &prog->watch_edictfield_value, g);
				}
			}
#endif
		}
#endif

#undef DISPATCH_OPCODE
#undef HANDLE_OPCODE
#undef USE_COMPUTED_GOTOS
#undef PRE_ERROR
#undef ADVANCE_PROFILE_BEFORE_JUMP

#undef OPB_WRITE
#ifdef PRVM_VIEWINTERPRETER
#undef OPA
#undef OPB
#define OPA ((prvm_eval_t *)&prog->globals.fp[st->operand[0]])
#define OPB ((prvm_eval_t *)&prog->globals.fp[st->operand[1]])
#endif

#undef VIEW_GLOBAL_WRITE
