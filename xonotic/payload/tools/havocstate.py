import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import struct


TYPES = {0: "void", 1: "string", 2: "float", 3: "vector", 4: "entity", 5: "field", 6: "function", 7: "pointer"}


def clean(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                  lambda match: "\n" * match[0].count("\n"), text, flags=re.S)


def load(path):
    data = path.read_bytes()
    header = struct.unpack_from("<15i", data)
    string_offset = header[10]

    def string(offset):
        begin = string_offset + offset
        return data[begin:data.index(b"\0", begin)].decode("utf8", "replace")

    def definitions(offset, count):
        rows = []
        for index in range(count):
            kind, position, name = struct.unpack_from("<HHi", data, offset + index * 8)
            rows.append({"name": string(name), "type": TYPES[kind & 32767], "offset": position})
        return rows

    functions = []
    for index in range(header[9]):
        record = struct.unpack_from("<7i8B", data, header[8] + index * 36)
        functions.append({"name": string(record[4]), "statement": record[0],
                          "local_base": record[1], "local_words": record[2],
                          "parameter_words": sum(record[7:7 + record[6]])})
    fields = definitions(header[6], header[7])
    aliases = {f"{row['name']}_{axis}" for row in fields if row["type"] == "vector" for axis in "xyz"}
    fields = [row for row in fields if row["type"] != "void" and row["name"] not in aliases]
    return {"file": str(path), "sha256": hashlib.sha256(data).hexdigest(), "version": header[0],
            "entity_words": header[14], "global_words": header[13], "string_bytes": header[11],
            "fields": fields, "globals": definitions(header[4], header[5]), "functions": functions}


def source_inventory(root):
    declarations = defaultdict(list)
    references = defaultdict(set)
    functions = {}
    globals_ = defaultdict(list)
    for path in sorted((root / "server/bot").rglob("*")):
        if path.suffix not in (".qc", ".qh"):
            continue
        source = str(path.relative_to(root))
        text = clean(path.read_text())
        for match in re.finditer(r"(?m)^\.\w+(?:\([^;]*?\))?\s+([^;]+);", text):
            for item in match[1].split(","):
                name = re.match(r"\s*(\w+)", item)
                if name:
                    declarations[name[1]].append({"file": source, "line": text[:match.start()].count("\n") + 1})
        for name in re.findall(r"\.\s*([A-Za-z_]\w*)", text):
            references[name].add(source)
        for match in re.finditer(r"(?m)^(?:float|int|bool|vector|entity|string|void)\s+(\w+)\s*\([^;{}]*\)\s*\{", text):
            functions[match[1]] = source
        for match in re.finditer(r"(?m)^(?:float|int|bool|vector|entity|string)\s+([^;{}()]+);", text):
            for item in match[1].split(","):
                name = re.match(r"\s*(\w+)", item)
                if name:
                    globals_[name[1]].append(source)
    return declarations, references, functions, globals_


def coordinates(fields):
    result = []
    for field in fields:
        for component in range(3 if field["type"] == "vector" else 1):
            result.append({"offset": field["offset"] + component,
                           "coordinate": field["name"] + ("." + "xyz"[component] if field["type"] == "vector" else ""),
                           "numeric_type": "float" if field["type"] in ("float", "vector") else "integer",
                           "qc_type": field["type"], "field": field["name"]})
    return sorted(result, key=lambda row: row["offset"])


def write_csv(path, rows, columns):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def engine_definitions(path, category, offset):
    result = []
    for kind, name in re.findall(r"PRVM_DECLARE_server" + category + r"(\w+)\((\w+)\)", path.read_text()):
        result.append({"name": name, "type": "entity" if kind == "edict" else kind, "offset": offset})
        offset += 3 if kind == "vector" else 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("progs", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    program = load(args.progs)
    declared, referenced, functions, global_names = source_inventory(args.source)
    all_coordinates = coordinates(program["fields"])
    assert [row["offset"] for row in all_coordinates] == list(range(program["entity_words"]))
    selected = []
    for field in program["fields"]:
        name = field["name"].split("[")[0]
        if name in declared:
            selected.append({**field, "declarations": declared[name]})
    direct = [field for field in program["fields"] if field["name"] in referenced]
    bot_functions = [{**function, "source": functions[function["name"]]}
                     for function in program["functions"] if function["name"] in functions]
    intervals = {offset for function in bot_functions if function["statement"] > 0
                 for offset in range(function["local_base"], function["local_base"] + function["local_words"])}
    bot_globals = [row for row in program["globals"] if row["name"].split("[")[0] in global_names]
    engine_fields = engine_definitions(args.engine / "prvm_offsets.h", "field", program["entity_words"])
    engine_globals = engine_definitions(args.engine / "prvm_offsets.h", "global", program["global_words"])
    groups = defaultdict(list)
    for field in selected:
        groups[field["declarations"][-1]["file"]].append(field)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    columns = ["offset", "coordinate", "numeric_type", "qc_type", "field"]
    write_csv(output / "entity-coordinates.csv", all_coordinates, columns)
    write_csv(output / "bot-declared-coordinates.csv", coordinates(selected), columns)
    write_csv(output / "bot-direct-field-references.csv", coordinates(direct), columns)
    write_csv(output / "bot-functions.csv", bot_functions, ["name", "source", "parameter_words", "local_words", "local_base", "statement"])
    write_csv(output / "program-globals.csv", program["globals"], ["offset", "name", "type"])
    write_csv(output / "bot-declared-globals.csv", bot_globals, ["offset", "name", "type"])
    write_csv(output / "engine-added-entity-coordinates.csv", coordinates(engine_fields), columns)
    write_csv(output / "runtime-entity-coordinates.csv", all_coordinates + coordinates(engine_fields), columns)
    write_csv(output / "engine-added-global-coordinates.csv", coordinates(engine_globals), columns)
    global_slots = defaultdict(list)
    for row in program["globals"]:
        if row["type"] != "void":
            for coord in coordinates([row]):
                global_slots[coord["offset"]].append(coord)
    global_coordinates = [{"offset": offset,
                           "names": " | ".join(sorted({row['coordinate'] for row in global_slots[offset]})),
                           "declared_types": " | ".join(sorted({row['qc_type'] for row in global_slots[offset]})),
                           "storage": "VM word; unnamed/reused words retain integer bit representation"}
                          for offset in range(program["global_words"])]
    write_csv(output / "program-global-coordinates.csv", global_coordinates, ["offset", "names", "declared_types", "storage"])
    summary = {key: program[key] for key in ("file", "sha256", "version", "entity_words", "global_words", "string_bytes")}
    summary.update(entity_numeric_types=dict(Counter(row["numeric_type"] for row in all_coordinates)),
                   entity_reference_types=dict(Counter(row["qc_type"] for row in all_coordinates if row["numeric_type"] == "integer")),
                   bot_declared_fields=len(selected), bot_declared_coordinates=len(coordinates(selected)),
                   bot_declared_numeric_types=dict(Counter(row["numeric_type"] for row in coordinates(selected))),
                   bot_direct_field_coordinates=len(coordinates(direct)),
                   bot_function_count=len(bot_functions), bot_function_local_storage_union=len(intervals),
                   bot_function_max_local_words=max(function["local_words"] for function in bot_functions),
                   bot_declared_global_coordinates=len(coordinates(bot_globals)),
                   bot_declared_global_numeric_types=dict(Counter(row["numeric_type"] for row in coordinates(bot_globals))),
                   engine_added_entity_coordinates=len(coordinates(engine_fields)),
                   engine_added_entity_numeric_types=dict(Counter(row["numeric_type"] for row in coordinates(engine_fields))),
                   runtime_entity_words=program["entity_words"] + len(coordinates(engine_fields)),
                   runtime_entity_numeric_types=dict(Counter(row["numeric_type"] for row in all_coordinates + coordinates(engine_fields))),
                   engine_added_global_coordinates=len(coordinates(engine_globals)),
                   runtime_global_words=program["global_words"] + len(coordinates(engine_globals)),
                   groups={name: {"fields": len(fields), "coordinates": len(coordinates(fields)),
                                  "numeric_types": dict(Counter(row["numeric_type"] for row in coordinates(fields)))}
                           for name, fields in groups.items()},
                   limitations=["Full entity layout includes fields belonging to other entity classes and game modes.",
                                "Source-declared bot fields also include waypoint, weapon and scripting entities.",
                                "Source lexical references are a lower bound: macros, dynamic field selectors and external callees add accesses.",
                                "Function locals overlap in this optimized program and must not be summed.",
                                "Whole-program globals include constants, field/function addresses and reusable temporaries.",
                                "String/reference payloads, external entity populations, native engine state and call stacks are separate storage."])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Havocbot declared entity state", "", "Generated from compiled field offsets and declarations under `server/bot`. Vectors expand to three floats; references are integer handles. These declarations span bot, weapon, waypoint and scripting entities.", ""]
    for name, fields in groups.items():
        lines.extend(["## " + name, "", "| Offset | Field | Scalar coordinates | Numeric type |", "| --- | --- | --- | --- |"])
        for field in fields:
            lines.append(f"| {field['offset']} | `{field['name']}` | {3 if field['type'] == 'vector' else 1} | {'float' if field['type'] in ('float', 'vector') else 'integer (' + field['type'] + ')'} |")
        lines.append("")
    (output / "bot-declared-state.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
