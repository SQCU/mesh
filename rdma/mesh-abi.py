import json
from pathlib import Path
import re
import subprocess
import sys


def generate(header, output):
    tree = json.loads(subprocess.check_output(['clang', '-Xclang', '-ast-dump=json',
        '-fsyntax-only', '-x', 'c', str(header)], text=True))
    names = {'hdr': 'Region', 'mesh_ctx': 'Context', 'mesh_row_range': 'RowRange', 'mesh_row_map': 'RowMap',
        'mesh_row_function': 'RowFunction', 'mesh_row_binding': 'RowBinding',
        'mesh_row_metadata': 'Metadata', 'mesh_memory_span': 'MemorySpan'}
    records = {node['name']: node for node in tree['inner']
        if node['kind'] == 'RecordDecl' and node.get('completeDefinition') and node.get('name') in names}
    primitive = {'void': 'None', 'char': 'c.c_char', 'unsigned char': 'c.c_ubyte',
        'int': 'c.c_int', 'unsigned int': 'c.c_uint', 'size_t': 'c.c_size_t',
        'uint16_t': 'c.c_uint16', 'uint32_t': 'c.c_uint32', 'uint64_t': 'c.c_uint64',
        'int64_t': 'c.c_int64'}

        def kind(value):
        value = re.sub(r'\b(const|restrict|volatile)\b', '', value).strip()
        array = re.fullmatch(r'(.+)\[(\d+)\]', value)
        if array: return f'({kind(array[1])} * {array[2]})'
        if value.endswith('*'):
            base = value[:-1].strip()
            if base == 'void': return 'c.c_void_p'
            if base == 'char': return 'c.c_char_p'
            return f'c.POINTER({kind(base)})'
        if value.startswith('struct '): return names[value[7:]]
        return primitive[value]

    lines = ['import ctypes as c', 'from pathlib import Path',
        '_lib = c.CDLL(str(Path(__file__).resolve().parents[1] / "libmesh.dylib"), use_errno=True)',
        'ABSENT = (1 << 32) - 1', 'WRITING = 1 << 63']
    for name in records: lines.append(f'class {names[name]}(c.Structure): pass')
    for name, node in records.items():
        fields = []
        for field in node['inner']:
            if field['kind'] != 'FieldDecl': continue
            fields.append((field['name'], kind(field['type']['qualType'])))
            if name == 'hdr' and field['name'] == 'data_off': break
        lines.append(f'{names[name]}._fields_ = [' + ', '.join(f'({key!r}, {value})' for key, value in fields) + ']')
    for node in tree['inner']:
        if node['kind'] != 'FunctionDecl' or node.get('storageClass') == 'static' or not node.get('name', '').startswith('mesh_'): continue
        result = node['type']['qualType'].split('(')[0].strip()
        arguments = [kind(arg['type']['qualType']) for arg in node.get('inner', []) if arg['kind'] == 'ParmVarDecl']
        lines.extend([f"_lib.{node['name']}.restype = {kind(result)}",
            f"_lib.{node['name']}.argtypes = [{', '.join(arguments)}]"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
