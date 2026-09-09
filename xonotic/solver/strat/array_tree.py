import json
import os
import zipfile
from dataclasses import fields, is_dataclass
from enum import Enum

import numpy as np


def write_payload(target, payload):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, value in payload.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)

def atomic_save(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".new", "wb") as target:
        write_payload(target, payload)
        target.flush()
        os.fsync(target.fileno())
    os.replace(path + ".new", path)

def pack_state(state, prefix="__runtime__"):
    payload, objects, references = {}, [], {}
    strings, string_ids = [], {}
    def string_id(value):
        if value not in string_ids:
            string_ids[value] = len(strings)
            strings.append(value)
        return string_ids[value]
    def encode(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str):
            return {'string': string_id(value)}
        if value is None or isinstance(value, (int, float, bool)):
            return {"value": value}
        identity = id(value)
        if identity in references:
            return {"ref": references[identity]}
        index = references[identity] = len(objects)
        objects.append(None)
        if type(value) in (list, tuple) and all(isinstance(item, str) for item in value):
            key = prefix + str(index)
            payload[key] = np.fromiter((string_id(item) for item in value), dtype=np.uint32, count=len(value))
            record = {'string_sequence': type(value).__name__, 'array': key}
        elif (isinstance(value, np.ndarray) or hasattr(value, "__array__")):
            key = prefix + str(index)
            payload[key] = np.asarray(value)
            record = {"array": key}
        elif isinstance(value, Enum):
            record = {"enum": type(value).__name__, "value": value.value}
        elif is_dataclass(value):
            record = {"type": type(value).__name__, "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
        elif isinstance(value, dict):
            record = {"dict": [[encode(key), encode(item)] for key, item in value.items()]}
        else:
            record = {"sequence": type(value).__name__, "items": [encode(item) for item in value]}
        objects[index] = record
        return {"ref": index}
    root = encode(state)
    metadata = json.dumps({'root': root, 'objects': objects, 'strings': strings}, separators=(',', ':')).encode()
    payload[prefix + 'meta'] = np.frombuffer(metadata, dtype=np.uint8)
    return payload

def unpack_state(payload, prefix="__runtime__", types=None):
    types = {value.__name__: value for value in (tuple, list, set)} | dict(types or {})
    encoded = np.asarray(payload[prefix + 'meta'])
    metadata = json.loads(encoded.tobytes().decode() if encoded.dtype == np.uint8 else str(encoded))
    strings = metadata.get('strings', ())
    restored = {}
    def decode(reference):
        if "value" in reference:
            return reference["value"]
        if 'string' in reference:
            return strings[reference['string']]
        index = reference["ref"]
        if index not in restored:
            record = metadata["objects"][index]
            if 'string_sequence' in record:
                value = types[record['string_sequence']](strings[int(index)] for index in payload[record['array']])
            elif "array" in record:
                value = np.asarray(payload[record["array"]])
            elif "enum" in record:
                value = types[record["enum"]](record["value"])
            elif "type" in record:
                value = types[record["type"]](**{key: decode(item) for key, item in record["fields"].items()})
            elif "dict" in record:
                value = {decode(key): decode(item) for key, item in record["dict"]}
            else:
                items = [decode(item) for item in record["items"]]
                kind = types[record["sequence"]]
                value = kind(*items) if hasattr(kind, "_fields") else kind(items)
            restored[index] = value
        return restored[index]
    return decode(metadata["root"])

