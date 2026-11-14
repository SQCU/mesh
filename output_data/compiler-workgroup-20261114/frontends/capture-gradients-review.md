# Original gradient targets and function-local training results

14 November 2026. Source/API observations, not a native training execution or performance result.

The patch changes the upstream `torch/export/exported_program.py` joint-decomposition OutputSpec producer from `target=source` to `target=spec.arg.name` for USER_INPUT. Its sibling parameter branch continues to use spec.target. Input placeholder renaming already uses the original input name; no downstream arg-number parsing is introduced.

Applied with patch --fuzz=0 --no-backup-if-mismatch to `/Users/mdot/.venv-mesh-uv`, Torch2.14.0 commit08187d9e0fba026dc8217405802ab5381dc88d90. Original fileSHA bb4908767a26f84fbdc0d84a3413d4818cf7f88e34ae450a29f537f8fa165fff; patched bf8418070373473faddea7a535962c2e9626a9493f98d67a0c7d1c17ed56d691. Exact source patch and application steps are compiler/patches/torch-export-user-input-gradients.patch and README.md.

Actual capture observation exits0/stderr0. Each known function returns its own TrainingResult(outputs,parameters,inputs) from its exported output kinds and target names. The outer program preserves its actual pytree. A(x).inputs['x'] is passed to B along with the original x; B's retained input references are exactly the first call's gradient output and the original input. All forward and gradient leaves belong to the same capture FakeTensorMode. Gradients are converted from the actual exported gradient tensor through one upstream FakeTensorConverter per call, preserving each call occurrence; no shape surrogate or second exported-body evaluator. Actual buffers and inputs remain unchanged by capture.

Durable raw evidence in the metal repository:
- output_data/compiler-workgroup-20261114/frontends/torch/observe/torch-gradient-patch.json
- output_data/compiler-workgroup-20261114/frontends/torch/observe/fake_export-grad-call.stdout
- output_data/compiler-workgroup-20261114/frontends/torch/observe/fake_export-grad-call.stderr
- SAME974 session42.jsonl (model claude-opus-5-5,9740398d-f513-4c78-8578-f80b79fa2225).

Current source AST checks pass export/driver/ir/runtime. Intentional malformed original paths remain mesh_compile.py7628, lower_coreml.py321, examples/tp.py121. Native joint-call training, aliases and mutation execution are not established by these observations. Mutation local-write source handoff and observed Work/share consumption remain separate active corrections.
