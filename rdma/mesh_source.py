"""Read-only committed-source export and verification shared by installers and runners."""
import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

MANIFEST = '.mesh-source.json'


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout


def pin(root):
    return git(root, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()


def snapshot_path(root, name):
    """A member's lexical path; parents must never redirect into a symlink target."""
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'unsupported archive member {name}')
    parent = Path(root)
    for part in path.parts[:-1]:
        parent /= part
        if parent.is_symlink():
            raise ValueError(f'archive member has symlink parent: {name}')
    return Path(root) / path


def export(root, commit, destination):
    """Export exactly one resolved commit; no index, worktree, refs or remote changes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    archive = git(root, 'archive', '--format=tar', commit)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = tar.getmembers()
        links = {PurePosixPath(member.name) for member in members if member.issym()}
        names = [PurePosixPath(member.name) for member in members]
        if len(set(names)) != len(names):
            raise ValueError('archive contains duplicate member names')
        # Validate the entire namespace before extraction, regardless of entry order.
        for member in members:
            path = PurePosixPath(member.name)
            if any(parent in links for parent in path.parents):
                raise ValueError(f'archive member has symlink parent: {member.name}')
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.islnk():
                raise ValueError(f'unsupported archive member {member.name}')
            if member.isdir():
                continue
            if not (member.isfile() or member.issym()):
                raise ValueError(f'unsupported archive type {member.name}')
            target = snapshot_path(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                # A committed link is opaque source data, even when its target is
                # external or absent. Hash/archive its text and never open its target.
                target.symlink_to(member.linkname)
            else:
                target.write_bytes(tar.extractfile(member).read())
                target.chmod(member.mode)
            files[member.name] = {'mode': member.mode,
                                  'kind': 'symlink' if member.issym() else 'file'}
    tracked = {row.split(b'\t', 1)[1].decode() for row in git(root, 'ls-tree', '-rz', commit).split(b'\0') if row}
    if tracked != files.keys():
        raise ValueError(f'archive did not export every committed file: {sorted(tracked - files.keys())}')
    manifest = {'schema': 1, 'commit': commit, 'files': files}
    (destination / MANIFEST).write_text(json.dumps(manifest, sort_keys=True) + '\n')
    return manifest


def manifest(root):
    return json.loads((Path(root) / MANIFEST).read_text())


def verify(root):
    root = Path(root)
    document = manifest(root)
    for name, record in document['files'].items():
        path = snapshot_path(root, name)
        if not (path.is_symlink() if record['kind'] == 'symlink' else path.is_file()):
            raise ValueError(f'committed source member missing or wrong kind: {name}')
        # Git versions executable status, not permissions modified by a node's umask.
        if not path.is_symlink() and bool(path.stat().st_mode & 0o111) != bool(record['mode'] & 0o111):
            raise ValueError(f'committed source mode differs: {name}')
    return document


def files(root, relative):
    root, relative = Path(root), Path(relative)
    if (root / MANIFEST).exists():
        names = manifest(root)['files']
        return sorted(str(Path(name).relative_to(relative)) for name in names
                      if Path(name).is_relative_to(relative))
    return git(root / relative, 'ls-files', '-z').decode().rstrip('\0').split('\0')


def revision(root):
    return manifest(root)['commit'] if (Path(root) / MANIFEST).exists() else pin(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('files', 'revision', 'verify'))
    parser.add_argument('root', type=Path)
    parser.add_argument('relative', nargs='?', default='.')
    args = parser.parse_args()
    if args.action == 'files':
        import sys
        sys.stdout.buffer.write(b''.join(os.fsencode(p) + b'\0' for p in files(args.root, args.relative)))
    elif args.action == 'revision':
        print(revision(args.root))
    else:
        print(json.dumps({'verified_commit': verify(args.root)['commit']}))


if __name__ == '__main__':
    main()
