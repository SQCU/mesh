# rdma/mesh_compile.py at 1ac90a36, headers made unparseable by b25ec14b: lines 7611-7686 (_replays), 8182-8279
# (_bound), 8348-8512 (program), 8734-8764 (load) and 8767-8821 (_runner). Dead reference source: see README.md


# ExecuTorch, "Concepts", Memory planning; torch 2.14.0 torch/_inductor/cudagraph_trees.py; M. Lam, PLDI 1988, §2.3;
# metal_ir.h metal_ir_value_placed, metal_ir_step_placed, metal_ir_place; metal_recording.h state:buffer:status:,
# replay: and MetalReplayNative; design/compilation-composition-change-request.md, "Complete
# replacement workgroup" (6's stage E): each row's values bound to their placed copies (a plan value's copies as its
# reader's executions select them, _expanded; a realized value's own copies), its step word and transport view, its
# replay and compiled CPU entries the row (MetalCall `call` = j + 1) of the one native program, prepared with the
# plan's results (mesh-collective.h struct mesh_planned) and retained by it until retirement
_replays(programs, context, irs, planned, values, addresses, base, buffer, bound, owned, made, steps, every):
    lib, recorder, error = programs['lib'], programs['recorder'], C.create_string_buffer(4096)
    compiler = programs['compiler']
    queue, rows, selection = OBJECT(recorder, _SEL(b'queue')), [], {'every': every, 'rated': programs['rated']}

    def offsets(at):
        return (C.c_uint64 * len(at))(*[a - base for a in at])
    for entry in programs['rows']:
        if entry['builder'] is None:
            rows.append(None)
            continue
        k, prepared, p = entry['k'], entry['prepared'], entry['builder'].p
        for vi, owner in entry['builder'].planned.items():
            at, width = _expanded(selection, owner, k, addresses, base)
            compiler.metal_ir_value_placed(p, vi, buffer, at, width, 0, values[owner[0]][owner[1]].shares)
        for vi, index in entry['own'].items():
            value = values[k][index]
            compiler.metal_ir_value_placed(p, vi, buffer, offsets(addresses[k][index]), value.section.count, 0,
                                           value.shares)
        if k in steps:
            step = values[k][steps[k]]
            compiler.metal_ir_step_placed(p, buffer, offsets(addresses[k][steps[k]]), step.section.count, step.shares)
        STATE_BUFFER(recorder, _SEL(b'state:buffer:status:'), prepared, made.state, entry['status'])
        if compiler.metal_ir_place(p, error, len(error)):
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its placed accesses: %s'
                          % (entry['node'].name, k, error.value.decode()))
        replay = Replay()
        status = lib.MetalRecorderReplay(recorder, prepared, C.byref(replay), error, len(error))
        if status:
            raise OSError(status, 'call %s of compiled function %d: the replay: %s'
                          % (entry['node'].name, k, error.value.decode()))
        owned[id(replay)] = lambda replay=replay: lib.MetalReplayDestroy(C.byref(replay))
        rows.append(replay)
    calls = (Call * len(rows))(*[Call(C.pointer(r) if r is not None else None, row + 1, None, 0)
                                 for row, r in enumerate(rows)])
    # metal_recording.h MetalRecorderProgram
    arguments = (recorder, calls, len(calls), C.byref(planned))
    program = _owned(owned, lib.MetalProgramLoad(*arguments, programs['archive'], error, len(error))
                     if 'archive' in programs else lib.MetalRecorderProgram(*arguments, error, len(error)))
    if not program:
        raise OSError(errno.EINVAL, 'the program\'s calls: %s' % error.value.decode())
    # metal_recording.h MetalProgramHorizon
    horizon = lib.MetalProgramHorizon(program)
    period = _period({k: b[3] for k, b in enumerate(bound)}, horizon)
    # design/synchronization-signature-change-request.md items 1 and 2; metal_recording.h MetalProgramSignature
    if 'signature' in programs:
        signed = programs['signature']
    else:
        replayed = mesh_graph.Signature()
        status = lib.MetalProgramSignature(program, C.addressof(context), horizon + period, C.byref(replayed), error,
                                           len(error))
        if status:
            raise OSError(status, 'the program\'s signature: %s' % error.value.decode())
        try:
            signed = dict(mesh_graph.forms(replayed), record=mesh_graph.record(replayed))
        finally:
            mesh_graph.free(replayed)
    # tools/e2b_replay.m signature_write; bin/mesh-sync-monitor.py
    prefix = os.environ.get('MESH_SIGNATURE')
    if prefix:
        written = True
        for suffix, data in (('.record', signed['record']), ('.mtx', signed['matrix_market'].encode()),
                             ('.dot', signed['dot'].encode()), ('.txt', signed['space_time'].encode())):
            written = mesh_disk.write(prefix + suffix, data, 'a MESH_SIGNATURE artifact', 'w') and written
        if written:
            print('prepared signature %s digest %s' % (prefix, signed['digest']), flush=True)
    gpu = OBJECT(queue, _SEL(b'device'))
    return {'lib': lib, 'queue': queue, 'calls': calls, 'program': program, 'horizon': horizon,
            'ended': _owned(owned, OBJECT(gpu, _SEL(b'newSharedEvent'))), 'error': error, 'signature': signed}


_bound(context, state, retained, deadline):
    calls, every, given = retained['calls'], retained['every'], retained['given']
    nodes, page, irs, programs = retained['nodes'], retained['page'], retained['irs'], retained['programs']
    steps, arrays, actors, analysed = retained['steps'], retained['arrays'], retained['actors'], retained['analysed']
    functions = (mesh.Function * len(irs))(*[
        mesh.Function(C.cast(d, C.POINTER(mesh.Declared)), C.cast(w, C.POINTER(mesh.Weight)),
                      C.cast(v, C.POINTER(mesh.Value)), C.cast(e, C.POINTER(mesh.Dependence)), len(d), len(v),
                      len(e), 0, analysed.outstanding[k], 0,
                      *((C.cast(actors[k][0], C.POINTER(C.c_double)), C.cast(actors[k][1], C.POINTER(C.c_double)),
                         C.cast(actors[k][2], C.POINTER(mesh_graph.Dependence)), actors[k][3], actors[k][4])
                        if k in actors else (None, None, None, 0, 0)),
                      relation=actors[k][6].relation if k in actors else None)
        for k, (d, w, v, e) in enumerate(arrays)])
    # design/compilation-composition-change-request.md item 6, unit 6D: the plan's results for native
    # preparation (mesh-collective.h struct mesh_planned), each call's plan, lead, ordinal and every the
    # identities the specialization and the program's rows take, kept until the program is retired
    planned = mesh.Planned()
    arguments = (C.byref(context), _group(), _program(), functions, len(irs),
                 (mesh.U * len(calls))(*calls), len(calls), analysed.orders, analysed.causes,
                 analysed.norders, analysed.chains, deadline)
    if 'saved_plan' in retained:
        # mesh-collective.h mesh_plan_load: the saved plan's record, a span of the owned native archive, borrowed by
        # the loaded plan, which adopts one archive reference on success (metal_ir_archive_retain; released by
        # metal_ir_archive_free) and restores the functions' compiled parts and relation from it
        (record, length), archive, compiler = retained['saved_plan'], programs['archive'], programs['compiler']
        compiler.metal_ir_archive_retain(archive)
        status = mesh.LIB.mesh_plan_load(*arguments, record, length,
                                         RELEASE(C.cast(compiler.metal_ir_archive_free, C.c_void_p).value), archive,
                                         C.byref(planned))
        if status:
            compiler.metal_ir_archive_free(archive)
            mesh.check(status)
    else:
        mesh.check(mesh.LIB.mesh_plan(*arguments, C.byref(planned)))
    retained['functions'], retained['planned'] = functions, planned
    state['owned'][id(planned)] = lambda: mesh.LIB.mesh_planned_free(C.byref(planned))
    state.update(mapping=_Mapping(mesh.View()), devices={layout[2] for ir in irs for layout in ir.layouts})
    # JGroups ViewId.java
    views = [mesh.State.from_address(context.M + f.plan) for f in functions]
    if any((s.serial, s.sequencer, s.boot) != (nodes.serial, nodes.sequencer, nodes.boot) for s in views):
        raise OSError(errno.ESTALE, 'a plan\'s view is not the one its IR was built on: plan again')
    listed = [values for _, _, values, _ in arrays]
    # 6's stage D: the plan's transport words into each row's program, then its final emission
    made = _finalized(programs, context, irs, planned, listed, state['owned'])
    view, addresses = _mapped(context, functions, len(irs), listed, page)
    state['mapping'].view, hull = view, _hull(listed, addresses, page)
    buffer = _placed_buffer(hull)
    try:
        bound = [_bind(ir, values, at, buffer, hull.address, page, state['mapping'])
                 for ir, values, at in zip(irs, listed, addresses)]
        state['inputs'] = [next((b[2][i] for b in bound if i in b[2]), None) for i in range(len(given))]
        # 6's stage E: each row bound to its placed storage and replayed into the one native program, its
        # outputs specialized over the program's horizon
        state['replay'] = _replays(programs, context, irs, planned, listed, addresses, hull.address, buffer,
                                   bound, state['owned'], made, steps, every)
        state['horizon'] = state['replay']['horizon']
        if 'output_recipe' not in retained:
            retained['output_recipe'] = _specialize(programs, state['horizon'], bound, *retained['capture'],
                                                     retained['holders'], every)
        state['replay']['outputs'] = _outputs(retained['output_recipe'], bound)
        retained['bound'], retained['addresses'] = bound, addresses
    finally:
        if buffer:
            PUT(buffer, SELECTOR[b'release'])
    # torch 2.14.0 torch/export/exported_program.py, ExportedProgram; GSPMD section 4.5; Megatron-LM
    # core_r0.12.0 megatron/core/tensor_parallel/layers.py:156–165; design/problems-of-14b9a757-change-
    # request.md, "Sources for the second stops", 8, and "Sources for the third stops", 8;
    # design/torch-source-lowering-change-request.md item 1
    if 'initial' in retained:
        for k, v, copies in retained['initial']:
            value = listed[k][v]
            for at, data in zip(addresses[k][v], copies):
                if isinstance(at, list):
                    for page_index, address in enumerate(at):
                        piece = data[page_index * page:(page_index + 1) * page]
                        C.memmove(address, piece, len(piece))
                else:
                    C.memmove(at, data, value.section.bytes)
        return
    launched = False
    for ir, (_, _, _, copies) in zip(irs, bound):
        for v, data in ir.initial.items():
            for target in copies[v]:
                target.copy_(data)
                launched = launched or target.device.type == 'mps'
    if launched:
        torch.mps.synchronize()
    retained['initial'] = _initial(retained)
    programs['state_data'] = {}
    for row in programs['rows']:
        if row['builder'] is None:
            continue
        for v in range(row['builder'].source_values):
            info = ValueInfo()
            programs['compiler'].metal_ir_value_info(row['builder'].p, v, C.byref(info))
            if info.role == STATE and info.storage == v and (info.data or info.buffer):
                key, address, length = _constant_storage(info)
                programs['state_data'][key] = C.string_at(address, length)


# design/aot-compilation-change-request.md A1 to A3, A8 to A10 and appendix item 7;
# design/disjoint-pages-change-request.md C3.1, D.3.2 and D.3.3; mesh-collective.h mesh_plan, struct mesh_function and
# struct mesh_value; mesh-dataflow.h mesh_nodes_read, mesh_retire and struct mesh_ctx
# Two quantities, two owners: A10's outstanding bound, the width of each function's calls in the one traced block
# (_analyse); the executions in flight, item 2's Bff(e) that mesh_plan derives from each Function's t(v) (planned.horizon,
# the native program's lag through MetalProgramHorizon, sizing its copies and input waits); neither given
program(block, *example_inputs):
    tracer = Tracer()
    graph = tracer.trace(block)
    gm = torch.fx.GraphModule(tracer.root, graph)
    exporter = _Export(gm, FakeTensorMode())
    exporter.run(*[exporter.fake(x) for x in example_inputs])
    compiled, of, calls, sources = exporter.compiled, exporter.of, exporter.calls, exporter.sources
    # the block's returned leaves (_Export `marks` of its output's arguments): a call's result leaf the caller receives
    # at every node of the program, delivered where its function has no section (_deliver)
    produced = pytree.tree_leaves(map_arg(
        next(n for n in gm.graph.nodes if n.op == 'output').args, lambda a: exporter.marks.get(a)))
    returned = {(calls[leaf.call], leaf.leaf) for leaf in produced
                if isinstance(leaf, _Produced) and leaf.call is not None}
    # the traced block's calls of each function (`every`) and each call's ordinal among them (`place`): the static
    # call identities the analysis's bound, the capture's generations, step words and block-copy selection take
    # (mesh_plan receives `calls` and publishes the same, planned.call)
    first, grouped = mesh_graph.rows(len(compiled), calls)
    every, place = {f: first[f + 1] - first[f] for f in range(len(compiled))}, [0] * len(calls)
    for q, r in enumerate(grouped):
        place[r] = q - first[calls[r]]
    operands = [record.function.placement(record.gm, record.examples) for record in compiled]
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    # the block's reads, (consumer call j, its flat input leaf q, the producing leaf), one relation from _Export
    # `sources` (passed-through results resolved): block inputs' readers, deliveries and delivered operands take it
    reads = [(j, q, leaf) for j, read in enumerate(sources) for q, leaf in enumerate(read) if leaf is not None]
    readers = [[calls[j] for j, _, leaf in reads if leaf.call is None and leaf.leaf == i]
               for i in range(len(placeholders))]
    given = [(x.dtype, x.numel(), x.device.type, tuple(x.shape)) if isinstance(x, torch.Tensor) else None
             for x in example_inputs]
    constants = [None if isinstance(x, torch.Tensor) else x for x in example_inputs]
    context, state = _context(), {'stale': True, 'mapping': None, 'devices': (), 'executions': 0, 'horizon': 1,
                                  'inputs': [None] * len(given), 'timeout': None, 'replay': None, 'owned': {},
                                  'construction': None, 'constructed': {}}

    # design/aot-compilation-change-request.md A1 ("serialized objects we can resume without needing to repeatedly
    # re-compile anything"): the compilation's results at this node (_built), what an artifact retains, and what plan,
    # placement and binding derive from them each time (_bound)
    def _built(nodes, page):
        irs = [partition(record.gm, operand, nodes, page) for record, operand in zip(compiled, operands)]
        # design/compilation-composition-change-request.md items 1 and 6 (unit 6E); design/aot-compilation-change-
        # request.md A4: a function's results are whole at every node of its map (_partition), so a consumer at a node
        # outside its producer's map reads the producer's delivery there (_deliver). wanted[f, o]: the functions of the
        # calls reading function f's flat result leaf o (_Export `sources`), the same at every node
        wanted = {}
        for j, _, leaf in reads:
            if leaf.call is not None:
                wanted.setdefault((calls[leaf.call], leaf.leaf), set()).add(calls[j])
        _deliver(irs, wanted, returned, _program())
        for ir, record in zip(irs, compiled):
            lower(ir)
            _initial_values(ir, record.immutable - frozenset().union(*(c.written for c in compiled)))
            ir.weights = _weighed(ir, record.function)
            ir.delivered = {}
        # each consumer call's delivered operands here, before capture (the recorder's masked readers are emitted
        # there): ir.delivered[ordinal] = [(q, r, f, o, v, i, generation)], its flat input leaf q the leaf o of call r
        # (function f) landed in f's value v by f's transfer i (`irs[f].landed`), and the range's generation
        # (metal_recording.h struct MetalGeneration) {every, producer, ordinal} = {the consumer's calls of its
        # function a block, the producer's, call r's ordinal}: the consumer's execution t reads r's execution
        # (t div every) producer + ordinal (`every`, `place`)
        for j, q, leaf in reads:
            landed = irs[calls[leaf.call]].landed.get(leaf.leaf, (None, None)) if leaf.call is not None \
                else (None, None)
            irs[calls[j]].delivered.setdefault(place[j], [])
            if landed[1] is not None:
                irs[calls[j]].delivered[place[j]].append(
                    (q, leaf.call, calls[leaf.call], leaf.leaf, *landed,
                     (every[calls[j]], every[calls[leaf.call]], place[leaf.call])))
        holders = []
        for i, (x, functions) in enumerate(zip(given, readers)):
            k = next((k for k in functions if irs[k].device is not None), None)
            if x is not None and k is not None:
                irs[k].inputs[i] = (irs[k].value(*x[:3], 0, 0, 0, lag=1), x[3])
            holders.append(k if x is not None else None)
        # design/torch-capture-change-request.md item 9: "every node runs its captured program by replay, its
        # collectives included"; design/aot-compilation-change-request.md, appendix item 6
        # design/compilation-composition-change-request.md item 5
        # design/compilation-composition-change-request.md, "Complete replacement workgroup", 6's stages A to C:
        # each call's program declared and lowered, its compiled order's lifetimes the plan's operands
        programs = _programs(context, irs, compiled, gm, of, constants, holders, state['constructed'], calls)
        own = _lifetimes(programs, irs)
        # metal_ir.h metal_ir_step_placed; the later passes: each function's step word, its rows' plan execution
        # e = b every + ordinal, which the native program writes before row execution e; lag copies
        # (mesh-collective.c plan copies of a lagged value), horizon every of them, so each of the horizon block
        # executions in flight owns copy e mod copies; 8 bytes, e's 64 bits as the host's executions,
        # generations and count targets are
        steps = {}
        for k in dict.fromkeys(row['k'] for row in programs['rows'] if row['builder'] is not None
                                and row['step'] is not None
                                and programs['compiler'].metal_ir_step_state(row['builder'].p) == NONE):
            steps[k] = len(irs[k].values) + len(own[k])
            own[k].append(mesh.Value(section=mesh.Section(bytes=8), kind=mesh.EXECUTION, lag=1, run=1))
        # design/torch-capture-change-request.md item 2 and its 2026-11-09 appendix: each function's t(v) at this
        # node part by part (_actors), mesh_plan's Bff of their edges the horizon and host copies
        # (mesh-graph.h mesh_graph_actor_time ENODATA: a time over the measured operands alone, a lower bound the
        # plan keeps; the functions whose estimate is such a bound reported as `incomplete`, never a zero cost)
        actors = _actors(programs, irs, state['constructed'])
        state['incomplete'] = sorted(k for k, actor in actors.items() if not actor[5])
        analysed = _analyse(gm, of, compiled, sources, every)
        return {'nodes': nodes, 'page': page, 'irs': irs, 'holders': holders, 'programs': programs, 'steps': steps,
                'own': own, 'actors': actors, 'analysed': analysed, 'calls': calls, 'every': every,
                'given': given, 'constants': constants, 'capture': (gm, constants), 'returned': produced}

    # design/compilation-composition-change-request.md items 3, 6 and 7; metal-microbench metal_ir.h
    # metal_ir_specialize: a plan's own instance of the retained construction. mesh_plan writes its values'
    # sections and lags, the transport its ranges and releases, so each plan takes fresh arrays from the declared
    # inputs, copies of the declared marks, and each row's specialized program over the lowered one; `nodes` the
    # current view its binding is identified by
    def _planned(construction, nodes):
        programs, compiler, error = construction['programs'], construction['programs']['compiler'], \
            C.create_string_buffer(4096)
        rows = []
        for row in programs['rows']:
            planned, builder = dict(row), row['builder']
            if builder is not None:
                p = compiler.metal_ir_specialize(builder.p, error, len(error))
                if not p:
                    raise OSError(errno.EINVAL, 'call %s of compiled function %d: its specialization: %s'
                                  % (row['node'].name, row['k'], error.value.decode()))
                state['owned'][id(p)] = lambda p=p: compiler.metal_ir_program_free(p)
                planned['builder'] = types.SimpleNamespace(p=p, planned=builder.planned,
                                                           source_values=builder.source_values)
                planned['marks'] = {'received': [(value, f, v, Range.from_buffer_copy(bytes(declared)))
                                                 for value, f, v, declared in row['marks']['received']],
                                    'sent': [(value, Release.from_buffer_copy(bytes(release)))
                                             for value, release in row['marks']['sent']],
                                    'sending': row['marks']['sending']}
            rows.append(planned)
        arrays = [((mesh.Declared * len(ir.declared))(*ir.declared),
                   (mesh.Weight * len(ir.weights))(*ir.weights),
                   (mesh.Value * (len(ir.values) + len(mine)))(*ir.values, *mine),
                   (mesh.Dependence * len(ir.dependences))(*ir.dependences))
                  for ir, mine in zip(construction['irs'], construction['own'])]
        return dict(construction, nodes=nodes, programs=dict(programs, rows=rows), arrays=arrays)

    # the compilation's inputs: the outer closure's captured program and placement, this node's map (its count and
    # index), its page and the program's membership; a view's serial, sequencer and boot identify a plan only
    def make():
        deadline = time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS
        if state['mapping'] is not None or state['owned']:
            _retire(context, state)
        try:
            nodes, page = _nodes()
            key = (nodes.count, nodes.self, page, _program())
            if state['construction'] is None or state['construction']['key'] != key:
                _released(state['constructed'])
                state['construction'] = None
                try:
                    state['construction'] = dict(_built(nodes, page), key=key)
                except BaseException:
                    _released(state['constructed'])
                    raise
            state['built'] = _planned(state['construction'], nodes)
            _bound(context, state, state['built'], deadline)
        except BaseException:
            _retire(context, state)
            raise
        state.update(stale=False)

    return _runner(context, state, make)


load(path):
    with open(path, 'rb') as stream:
        data = stream.read()
    context, state = _context(), {'stale': True, 'mapping': None, 'devices': (), 'executions': 0, 'horizon': 1,
                                  'inputs': [], 'timeout': None, 'replay': None, 'owned': {}}

    def make():
        if state['mapping'] is not None or state['owned']:
            _retire(context, state)
        try:
            nodes = mesh.Nodes()
            mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
            programs = _restore_programs(data, state['owned'])
            saved = programs['saved']
            arrays = [(_declared_restore(d), _records(mesh.Weight, w), _records(mesh.Value, v),
                       _records(mesh.Dependence, e)) for d, w, v, e in saved['arrays']]
            outstanding, orders, causes, count, chains = saved['analysed']
            irs = [types.SimpleNamespace(layouts=layout, inputs=inputs, device=device, declared=d, values=v, homes={})
                   for (layout, inputs, device), (d, _, v, _) in zip(saved['irs'], arrays)]
            retained = {key: saved[key] for key in ('page', 'holders', 'steps', 'calls', 'every', 'given', 'constants',
                                                   'output_recipe', 'initial')}
            retained.update(nodes=nodes, irs=irs, arrays=arrays, actors={}, saved_plan=programs['saved_plan'],
                            analysed=Calls(outstanding, _records(mesh.U, orders), _records(mesh.U, causes), count,
                                           _records(mesh.U, chains)), programs=programs)
            state['built'] = retained
            _bound(context, state, retained, time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS)
            state.update(stale=False)
        except BaseException:
            _retire(context, state)
            raise
    return _runner(context, state, make)


_runner(context, state, make):
    def timed(work):
        if state['timeout'] is not None:
            raise OSError(errno.ETIMEDOUT, 'the program\'s run ended at its first ETIMEDOUT (R-7): %s'
                          % state['timeout'])
        try:
            return work()
        except OSError as failure:
            if failure.errno == errno.ETIMEDOUT:
                state['timeout'] = failure
            raise
    timed(make)

    def ended():
        return _ended(state['replay']) if state['replay'] is not None else 0

    def writable():
        if state['stale']:
            make()
        b = state['executions']
        _wait(lambda: ended() + state['horizon'] > b, ended, 'execution %d of the program: execution %d not ended, no '
              'end for MESH_PLATFORM_NS, so its inputs\' copy is not free (D.3.3)' % (b, b - state['horizon']))
        return tuple(None if placed is None else placed[1][b % len(placed[1])] for placed in state['inputs'])

    # design/compilation-composition-change-request.md item 5
    def execution():
        if state['stale']:
            make()
        b = state['executions']
        state['executions'] = b + 1
        try:
            return _replayed(context, state['replay'], b)
        except OSError as failure:
            if failure.errno in (errno.ESTALE, errno.ECANCELED):
                state['stale'] = True
            raise

    def inputs():
        return timed(writable)

    def run():
        return timed(execution)
    # MTL4CommandQueue.h signalEvent:value:
    run.inputs, run.ended = inputs, ended
    def save(path):
        if state['stale']:
            timed(make)
        return timed(lambda: _save_program(context, state, path))
    run.save = save
    run.synchronize = lambda: timed(lambda: _finish(state['replay']))
    run.signature = lambda: state['replay']['signature'] if state['replay'] is not None else None
    run.structure = lambda: _compared(state['built'], state['replay']['signature']) \
        if state['replay'] is not None and 'returned' in state['built'] else None
    weakref.finalize(run, _dropped, context, state)
    return run
