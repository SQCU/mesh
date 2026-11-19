"""Committed provenance and deployment failure semantics; no cluster access."""
import ast
import contextlib
import importlib.util
import io
import json
import os
import plistlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RDMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RDMA))
import mesh_hourly
import mesh_install
import mesh_nodes
import mesh_source


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True).stdout


def repository(path):
    path.mkdir()
    git(path, 'init', '-q')
    git(path, 'config', 'user.email', 'fixture@example.invalid')
    git(path, 'config', 'user.name', 'Provenance fixture')
    (path / 'code.py').write_text('COMMITTED = 1\n')
    git(path, 'add', 'code.py')
    git(path, 'commit', '-qm', 'committed fixture')
    return path


class SourceTests(unittest.TestCase):
    def test_snapshot_ignores_dirty_staged_and_untracked_files_and_pins_old_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = repository(root / 'repo')
            sha = mesh_source.pin(repo)
            (repo / 'code.py').write_text('STAGED = 2\n')
            git(repo, 'add', 'code.py')
            (repo / 'code.py').write_text('UNCOMMITTED = 3\n')
            (repo / 'injected.py').write_text('raise RuntimeError("must never ship")\n')
            before = git(repo, 'status', '--porcelain=v1'), (repo / '.git/index').read_bytes()
            snapshot = root / 'snapshot'
            mesh_source.export(repo, sha, snapshot)
            self.assertEqual((snapshot / 'code.py').read_text(), 'COMMITTED = 1\n')
            self.assertFalse((snapshot / 'injected.py').exists())
            self.assertEqual(mesh_source.verify(snapshot)['commit'], sha)
            self.assertEqual(before, (git(repo, 'status', '--porcelain=v1'), (repo / '.git/index').read_bytes()))
            git(repo, 'commit', '-qm', 'move HEAD after pin')
            self.assertNotEqual(mesh_source.pin(repo), sha)
            mesh_source.export(repo, sha, root / 'later')
            self.assertEqual((root / 'later/code.py').read_text(), 'COMMITTED = 1\n')

    def test_committed_external_symlinks_remain_opaque_in_export_and_upload(self):
        import hashlib
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = repository(root / 'repo')
            external = root / 'outside'
            external.mkdir()
            secret = b'external target contents must never enter the archive'
            (external / 'uncommitted.txt').write_bytes(secret)
            links = {'active': str(external), 'missing': str(root / 'missing-target'),
                     'relative': '../outside'}
            for name, target in links.items():
                (repo / name).symlink_to(target)
            git(repo, 'add', *links)
            git(repo, 'commit', '-qm', 'opaque committed links')
            snapshot = root / 'snapshot'
            document = mesh_source.export(repo, mesh_source.pin(repo), snapshot)
            for name, target in links.items():
                self.assertTrue((snapshot / name).is_symlink())
                self.assertEqual(os.readlink(snapshot / name), target)
                self.assertEqual(document['files'][name]['sha256'], hashlib.sha256(os.fsencode(target)).hexdigest())
            payload = mesh_hourly.archive_snapshot(snapshot)
            self.assertNotIn(secret, payload)
            with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
                self.assertEqual(set(tar.getnames()), {'code.py', *links, mesh_source.MANIFEST})
                for name, target in links.items():
                    item = tar.getmember(name)
                    self.assertTrue(item.issym())
                    self.assertEqual(item.linkname, target)
                    self.assertEqual(item.size, 0)
            self.assertEqual((external / 'uncommitted.txt').read_bytes(), secret)

    def test_archive_cannot_write_through_a_symlink_parent_in_either_entry_order(self):
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / 'outside'
            outside.mkdir()
            for link_first in (True, False):
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w') as tar:
                    link = tarfile.TarInfo('redirect')
                    link.type, link.linkname = tarfile.SYMTYPE, str(outside)
                    child = tarfile.TarInfo('redirect/injected.py')
                    child.size = 4
                    for member in (link, child) if link_first else (child, link):
                        tar.addfile(member, io.BytesIO(b'evil') if member.isfile() else None)
                with patch('mesh_source.git', return_value=stream.getvalue()):
                    with self.assertRaisesRegex(ValueError, 'symlink parent'):
                        mesh_source.export(root, 'fixture', root / str(link_first))
                self.assertFalse((outside / 'injected.py').exists())

    def test_duplicate_leaf_cannot_overwrite_an_external_symlink_target(self):
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / 'outside.py'
            outside.write_bytes(b'original')
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as tar:
                link = tarfile.TarInfo('same')
                link.type, link.linkname = tarfile.SYMTYPE, str(outside)
                tar.addfile(link)
                duplicate = tarfile.TarInfo('same')
                duplicate.size = 4
                tar.addfile(duplicate, io.BytesIO(b'evil'))
            with patch('mesh_source.git', return_value=stream.getvalue()):
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    mesh_source.export(root, 'fixture', root / 'snapshot')
            self.assertEqual(outside.read_bytes(), b'original')

    def test_upload_rejects_directory_replaced_with_external_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = repository(root / 'repo')
            (repo / 'package').mkdir()
            (repo / 'package/code.py').write_text('committed package\n')
            git(repo, 'add', '.')
            git(repo, 'commit', '-qm', 'package fixture')
            snapshot = root / 'snapshot'
            mesh_source.export(repo, mesh_source.pin(repo), snapshot)
            external = root / 'external'
            (snapshot / 'package').rename(external)
            (snapshot / 'package').symlink_to(external)
            with self.assertRaisesRegex(ValueError, 'symlink parent'):
                mesh_hourly.archive_snapshot(snapshot)

    def test_changed_snapshot_is_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = repository(root / 'repo')
            mesh_source.export(repo, mesh_source.pin(repo), root / 'source')
            (root / 'source/code.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'committed source differs'):
                mesh_hourly.archive_snapshot(root / 'source')

    def test_missing_committed_helper_records_failure_without_git_mutations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repos = [repository(root / name) for name in ('mesh', 'bench')]
            before = [(mesh_source.pin(repo), git(repo, 'status', '--porcelain'), (repo / '.git/index').read_bytes()) for repo in repos]
            executed = []
            real = subprocess.run

            def traced(argv, **kwargs):
                executed.append([str(arg) for arg in argv])
                return real(argv, **kwargs)

            with patch('subprocess.run', side_effect=traced), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mesh_hourly.launcher(*repos, root / 'out'), 1)
            record = json.loads((root / 'out/log.jsonl').read_text())
            self.assertIn('lacks shared hourly prerequisites', record['errors'][0]['error'])
            self.assertEqual(record['commits'], dict(zip(('mesh', 'metal-microbench'), (row[0] for row in before))))
            for repo, state in zip(repos, before):
                self.assertEqual(state, (mesh_source.pin(repo), git(repo, 'status', '--porcelain'), (repo / '.git/index').read_bytes()))
            self.assertTrue(executed)
            self.assertTrue(all(cmd[0] == 'git' and cmd[3] in ('rev-parse', 'archive', 'ls-tree') for cmd in executed))

    def test_success_dispatch_uses_committed_driver_and_imports_even_when_worktree_is_poisoned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh, bench = repository(root / 'mesh'), repository(root / 'bench')
            for helper in mesh_hourly.REQUIRED:
                path = mesh / helper
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# committed prerequisite fixture\n')
            committed_driver = (
                "import json, pathlib, sys\n"
                "from committed_helper import VALUE\n"
                "request = pathlib.Path(sys.argv[2])\n"
                "record = json.loads(request.read_text())\n"
                "record.update(status='completed', executed_driver=VALUE, imported_from=__file__)\n"
                "(request.parent/'result.json').write_text(json.dumps(record))\n")
            (mesh / 'rdma/mesh_hourly.py').write_text(committed_driver)
            (mesh / 'rdma/committed_helper.py').write_text("VALUE = 'committed helper'\n")
            configuration = bench / 'tools/mesh/hourly/workloads.json'
            configuration.parent.mkdir(parents=True)
            configuration.write_text('{"workloads": []}\n')
            for repo in (mesh, bench):
                git(repo, 'add', '.')
                git(repo, 'commit', '-qm', 'shared committed driver and config')
            expected = {'mesh': mesh_source.pin(mesh), 'metal-microbench': mesh_source.pin(bench)}
            (mesh / 'rdma/mesh_hourly.py').write_text("raise RuntimeError('dirty driver executed')\n")
            (mesh / 'rdma/committed_helper.py').write_text("raise RuntimeError('dirty helper imported')\n")
            configuration.write_text('not valid JSON; never a committed operand')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mesh_hourly.launcher(mesh, bench, root / 'out'), 0)
            record = json.loads((root / 'out/log.jsonl').read_text())
            self.assertEqual(record['commits'], expected)
            self.assertEqual(record['executed_driver'], 'committed helper')
            self.assertTrue(Path(record['imported_from']).resolve().is_relative_to((root / 'out').resolve()))
            self.assertEqual(record['driver_exit'], 0)

    def test_archive_excludes_build_leftovers(self):
        with tempfile.TemporaryDirectory() as temp:
            import tarfile
            root = Path(temp)
            repo = repository(root / 'repo')
            mesh_source.export(repo, mesh_source.pin(repo), root / 'source')
            (root / 'source/untracked_helper.py').write_text('bad')
            with tarfile.open(fileobj=io.BytesIO(mesh_hourly.archive_snapshot(root / 'source'))) as tar:
                self.assertEqual(set(tar.getnames()), {'code.py', mesh_source.MANIFEST})


class RunnerTests(unittest.TestCase):
    def test_remote_helpers_are_module_imports_not_stdin_source(self):
        node = mesh_nodes.Node('1', 'fixture.invalid')
        node.source = '.cache/snapshot/mesh'
        with patch.object(node, 'run', return_value=(0, '')) as run:
            node.py('installed', 'abc')
        args, kwargs = run.call_args
        self.assertIn('-m mesh_nodes node installed abc', args[0])
        self.assertIn('.cache/snapshot/mesh/rdma', args[0])
        self.assertNotIn('input', kwargs)
        self.assertNotIn('python - ', args[0])

    def test_install_failure_does_not_verify_or_enter_measurement(self):
        node = mesh_nodes.Node('7', 'fixture.invalid')
        with tempfile.TemporaryDirectory() as temp:
            with patch('mesh_hourly.stage', return_value={'commit': 'abc'}), patch.object(node, 'py', side_effect=[(0, ''), (7, '')]) as run:
                result = mesh_hourly.prepare_node(node, {'mesh': Path(temp)}, {'run': 'fixture', 'commits': {'mesh': 'abc'}}, Path(temp), 0)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['install_exit'], 7)
        self.assertEqual([call.args[0] for call in run.call_args_list], ['resets', 'install'])

    def test_shared_lease_rejects_competitor_and_cleans_its_own_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lease'
            with mesh_nodes.exclusive_lock('competitor', path):
                with self.assertRaisesRegex(RuntimeError, 'competitor'):
                    with mesh_nodes.exclusive_lock('hourly', path):
                        self.fail('contended lease entered')
                self.assertEqual((path / 'owner').read_text(), 'competitor\n')
            self.assertFalse(path.exists())

    def test_occupied_lease_is_retained_as_failure_without_node_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = root / 'request.json'
            request.write_text(json.dumps({'errors': [], 'status': 'pending'}))
            with patch.object(mesh_nodes, 'LOCK', root / 'lock'), mesh_nodes.exclusive_lock('competitor', root / 'lock'), patch('mesh_hourly._execute_locked') as run:
                self.assertEqual(mesh_hourly.execute(request), 1)
                run.assert_not_called()
            self.assertEqual(json.loads((root / 'result.json').read_text())['errors'][0]['stage'], 'cluster-lease')


class InstallTests(unittest.TestCase):
    def environment(self, root):
        source = root / 'source'
        (source / 'rdma').mkdir(parents=True)
        document = root / 'Library/Application Support/io.mesh.bridge/document'
        document.parent.mkdir(parents=True)
        document.write_text('/fixture/config.json\n')
        plist = root / 'Library/LaunchAgents/io.mesh.bridge.plist'
        plist.parent.mkdir(parents=True)
        plist.write_bytes(plistlib.dumps({'Label': 'io.mesh.bridge', 'ProgramArguments': ['/disk/mesh-heartbeat', '/disk/mesh-flow', '/config.plist']}))
        return source, plist

    def test_disabled_operator_state_does_not_build_or_signal(self):
        with patch('mesh_install.disabled', return_value=True), patch('mesh_install.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'operator-disabled'):
                mesh_install._install_locked(Path('/fixture'), 'abc', True)
        run.assert_not_called()

    def test_live_client_is_not_killed_or_released(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.environment(root)
            commands = []

            def command(*args, **kwargs):
                commands.append(args)
                return subprocess.CompletedProcess(args, 0, json.dumps({'present': True, 'clients': [42]}), '')

            with patch('mesh_install.Path.home', return_value=root), patch('mesh_install.disabled', return_value=False), patch('mesh_install.active_program', return_value='/old/mesh-heartbeat'), patch('mesh_install.loaded_program_arguments', return_value=['/old/mesh-heartbeat', '/old/mesh-flow', '/config.plist']), patch('mesh_install.run', side_effect=command):
                with self.assertRaisesRegex(RuntimeError, 'live client holders'):
                    mesh_install._install_locked(source, 'abc', True)
            self.assertFalse(any(args[0] == 'launchctl' for args in commands))

    def test_transition_releases_before_unload_and_verifies_new_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.environment(root)
            commands, states = [], iter([{'present': True, 'clients': []}, {'present': False, 'clients': []}])

            def command(*args, **kwargs):
                commands.append(args)
                text = json.dumps(next(states)) if str(args[0]).endswith('mesh-install-state') else '{"up":true}'
                return subprocess.CompletedProcess(args, 0, text, '')

            with patch('mesh_install.Path.home', return_value=root), patch('mesh_install.disabled', return_value=False), patch('mesh_install.active_program', return_value='/old/mesh-heartbeat'), patch('mesh_install.loaded_program_arguments', return_value=['/old/mesh-heartbeat', '/old/mesh-flow', '/config.plist']), patch('mesh_install.run', side_effect=command), patch('mesh_install.check_package'), patch('mesh_install.verify_active', return_value={'commit': 'abc'}) as checked, contextlib.redirect_stdout(io.StringIO()):
                mesh_install._install_locked(source, 'abc', True)
            events = [('launchctl', args[1], args[2]) if args[0] == 'launchctl' else str(args[0]) for args in commands]
            release = next(i for i, x in enumerate(events) if isinstance(x, tuple) and x[1] == 'kill')
            unload = next(i for i, x in enumerate(events) if isinstance(x, tuple) and x[1] == 'bootout')
            build = next(i for i, x in enumerate(events) if isinstance(x, str) and x.endswith('mesh-user.sh'))
            self.assertLess(release, unload)
            self.assertLess(unload, build)
            checked.assert_called_once_with(source, 'abc')

    def test_failed_transition_restores_previous_job_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, plist = self.environment(root)
            commands, states = [], iter([{'present': True, 'clients': []}, {'present': False, 'clients': []}])
            old_definition = plist.read_bytes()
            bootstrapped = []

            def command(*args, **kwargs):
                commands.append(args)
                if args[:2] == ('launchctl', 'bootstrap'):
                    bootstrapped.append(plistlib.loads(plist.read_bytes())['ProgramArguments'])
                if str(args[0]).endswith('mesh-user.sh'):
                    plist.write_bytes(b'new definition')
                    raise RuntimeError('build failed')
                text = json.dumps(next(states)) if str(args[0]).endswith('mesh-install-state') else ''
                return subprocess.CompletedProcess(args, 0, text, '')

            with patch('mesh_install.Path.home', return_value=root), patch('mesh_install.disabled', return_value=False), patch('mesh_install.active_program', side_effect=['/old/mesh-heartbeat', None]), patch('mesh_install.loaded_program_arguments', return_value=['/loaded/mesh-heartbeat', '/loaded/mesh-flow', '/loaded/config.plist']), patch('mesh_install.run', side_effect=command):
                with self.assertRaisesRegex(RuntimeError, 'build failed'):
                    mesh_install._install_locked(source, 'abc', True)
            self.assertEqual(plist.read_bytes(), old_definition)
            self.assertEqual(bootstrapped, [['/loaded/mesh-heartbeat', '/loaded/mesh-flow', '/loaded/config.plist']])
            self.assertTrue(any(args[:2] == ('launchctl', 'bootstrap') for args in commands))

    def test_loaded_arguments_preserve_actual_config_path_with_spaces(self):
        text = ("program = /loaded/rdma/mesh-heartbeat\narguments = {\n"
                "  /loaded/rdma/mesh-heartbeat\n  /loaded/rdma/mesh-flow\n"
                "  /Users/fixture/Library/Application Support/io.mesh.bridge/configuration.plist\n}\n")
        with patch('mesh_install.run', return_value=subprocess.CompletedProcess([], 0, text, '')):
            argv = mesh_install.loaded_program_arguments('fixture')
        self.assertEqual(argv, ['/loaded/rdma/mesh-heartbeat', '/loaded/rdma/mesh-flow',
                               '/Users/fixture/Library/Application Support/io.mesh.bridge/configuration.plist'])

    def test_active_path_mismatch_cannot_claim_installed(self):
        with patch('mesh_install.active_program', return_value='/old/mesh-heartbeat'):
            with self.assertRaisesRegex(RuntimeError, 'not the requested committed snapshot'):
                mesh_install.verify_active(Path('/new/source'), 'abc')


class ByteCounterTests(unittest.TestCase):
    def test_physical_payload_includes_relays_and_exact_tails(self):
        spec = importlib.util.spec_from_file_location('shared_benchmarks', RDMA / 'torch-mesh/torch_mesh/benchmarks.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = [dict(cable=0, peer=1, wire=0, sent=100, landed=50), dict(cable=1, peer=2, wire=0, sent=7, landed=2)]
        after = [dict(cable=0, peer=1, wire=0, sent=103, landed=53), dict(cable=1, peer=2, wire=0, sent=10, landed=5)]
        result = module.traffic_delta(before, after)
        self.assertEqual(sum(row['sent'] for row in result), 6)  # one 3-byte value, two physical hops
        with self.assertRaisesRegex(ValueError, 'backwards'):
            module.traffic_delta(after, before)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            module.traffic_delta(before, after[:1])

class ObserverAbiTests(unittest.TestCase):
    def test_actual_observer_accepts_only_proven_153_154_holder_layouts(self):
        from test_native_lifetimes import function
        source = (RDMA / 'mesh-install-state.c').read_text()
        body = r'''
#include "mesh.h"
#include <assert.h>
#include <string.h>
static int mesh_holds(pid_t pid,uint64_t start){return pid==42 && start==55;}
'''
        body += function(source, 'install_state_valid') + '\n' + function(source, 'install_state_clients')
        body += r'''
int main(void){
  _Alignas(64) unsigned char memory[512]={0};
  struct hdr *h=(struct hdr *)memory;
  h->magic=MESH_MAGIC;h->clients=2;h->client_off=256;
  struct mesh_client *slots=(struct mesh_client *)(memory+256);
  atomic_store(&slots[0].owner,42);slots[0].start=55;
  atomic_store(&slots[1].owner,42);slots[1].start=56; /* reused pid is not a holder */
  pid_t found[MESH_CLIENTS];
  for(uint32_t version=153;version<=154;version++){
    h->version=version;assert(install_state_valid(h,sizeof memory));
    assert(install_state_clients(h,found)==1 && found[0]==42);
  }
  h->version=0;assert(install_state_valid(h,sizeof memory));
  h->version=152;assert(!install_state_valid(h,sizeof memory));
  h->version=155;assert(!install_state_valid(h,sizeof memory));
  h->version=154;h->clients=MESH_CLIENTS+1;assert(!install_state_valid(h,sizeof memory));
  h->clients=2;h->client_off=UINT64_MAX;assert(!install_state_valid(h,sizeof memory));
  h->client_off=448;assert(!install_state_valid(h,sizeof memory));
  h->client_off=257;assert(!install_state_valid(h,sizeof memory));
  h->client_off=64;assert(!install_state_valid(h,sizeof memory));
  h->client_off=256;assert(!install_state_valid(h,sizeof(struct hdr)-1));
  h->magic=0;assert(!install_state_valid(h,sizeof memory));
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'observer.c').write_text(body)
            subprocess.run(['clang', '-std=c11', '-g', '-O1', '-fsanitize=address,undefined',
                            '-I', str(RDMA), str(root / 'observer.c'), '-o', str(root / 'observer')], check=True, capture_output=True)
            subprocess.run([str(root / 'observer')], check=True, capture_output=True)

    def test_hourly_installed_check_uses_imported_snapshot_not_old_active_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            active = root / '.cache/mesh/active-source'
            active.parent.mkdir(parents=True)
            active.write_text('/old/snapshot\n')
            with patch.object(mesh_nodes, 'HOME', root), patch('mesh_source.verify', return_value={'commit': 'abc'}), patch('mesh_install.check_package'), patch('mesh_install.verify_active', return_value={}) as check, contextlib.redirect_stdout(io.StringIO()):
                mesh_nodes.node_installed('abc', '1')
            check.assert_called_once_with(RDMA.parent.resolve(), 'abc')


if __name__ == '__main__':
    unittest.main()
