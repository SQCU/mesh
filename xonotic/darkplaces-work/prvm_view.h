#ifndef PRVM_VIEW_H
#define PRVM_VIEW_H

#define PRVM_VIEW_PAGE 256
#define PRVM_VIEW_STATE_HEADER 15
#define PRVM_VIEW_RESPONSE_HEADER 12
#define PRVM_VIEW_STATE_WIDTH (PRVM_VIEW_STATE_HEADER + 6 * PRVM_VIEW_PAGE)
#define PRVM_VIEW_RESPONSE_WIDTH (PRVM_VIEW_RESPONSE_HEADER + 2 * PRVM_VIEW_PAGE)
#define PRVM_VIEW_READONLY (1u << 21)

typedef struct prvm_view_page_s
{
	int entity, offset;
	unsigned int generation, checked_sequence;
	int forcing;
	unsigned char present[PRVM_VIEW_PAGE], type[PRVM_VIEW_PAGE];
	float state[PRVM_VIEW_PAGE], residual[PRVM_VIEW_PAGE], velocity[PRVM_VIEW_PAGE];
	float read_time[PRVM_VIEW_PAGE];
	prvm_vec_t applied[PRVM_VIEW_PAGE];
	prvm_int_t integer_applied[PRVM_VIEW_PAGE], integer_state[PRVM_VIEW_PAGE];
	uint64_t applied_epoch[PRVM_VIEW_PAGE];
	struct prvm_view_page_s *next;
} prvm_view_page_t;

typedef struct prvm_view_s
{
	int owner, sequence, applied_sequence;
	unsigned int generation;
	unsigned char *global_types;
	unsigned char *global_readonly, *global_invariant, *field_readonly;
	prvm_vec_t *float_view;
	prvm_int_t *integer_view;
	uint64_t epoch;
	double source_time, duration, tau, decay, gain, applied_time, applied_source_time;
	unsigned long reads, reads_modified, calls;
	unsigned long faults;
	int fault_sequence;
	prvm_view_page_t ***records, *pages, **tail;
	struct prvm_view_s *next;
} prvm_view_t;

prvm_view_t *PRVM_ViewFor(struct prvm_prog_s *prog, int owner);
prvm_view_page_t *PRVM_ViewPage(struct prvm_prog_s *prog, prvm_view_t *view, int entity, int offset);
prvm_eval_t PRVM_ViewRead(struct prvm_prog_s *prog, int entity, int offset, int type);
prvm_view_t *PRVM_ViewBegin(struct prvm_prog_s *prog, int owner);
void PRVM_ViewEnd(struct prvm_prog_s *prog, prvm_view_t *previous);
void PRVM_ViewTime(prvm_view_t *view, double now);
int PRVM_ViewReadOnly(prvm_view_t *view, int entity, int offset);
void PRVM_ViewMask(prvm_view_t *view, prvm_view_page_t *page, int slot);
prvm_eval_t *PRVM_ViewGlobalRead(struct prvm_prog_s *prog, int offset, int type);
void VM_mesh_view_run(struct prvm_prog_s *prog);
void VM_mesh_view_set(struct prvm_prog_s *prog);
void VM_mesh_view_stat(struct prvm_prog_s *prog);
void VM_mesh_view_publish(struct prvm_prog_s *prog);
void VM_mesh_round_outcome(struct prvm_prog_s *prog);

#endif
