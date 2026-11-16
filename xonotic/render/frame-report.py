import argparse
import json
from pathlib import Path
import statistics


def distribution(values):
    ordered = sorted(values)
    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lo = int(position)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)
    return {"mean": statistics.fmean(ordered), "p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99), "max": ordered[-1]}


def summarize(rows):
    return {"frames": len(rows), "interval_ms": distribution([r["interval_ms"] for r in rows]),
            "screen_ms": distribution([r["screen_ms"] for r in rows]), "swap_ms": distribution([r["swap_ms"] for r in rows]),
            "outside_screen_ms": distribution([r["interval_ms"] - r["screen_ms"] - r["swap_ms"] for r in rows]),
            "particles": distribution([r["particles"] for r in rows]) if rows[0]["counter_schema"] else None,
            "draws": distribution([r["draws"] for r in rows]) if rows[0]["counter_schema"] else None,
            "quality": distribution([r["quality"] for r in rows]),
            "over_60hz_budget_fraction": sum(r["interval_ms"] > 1000 / 60 for r in rows) / len(rows),
            "over_30hz_budget_fraction": sum(r["interval_ms"] > 1000 / 30 for r in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    rows, sources = [], []
    for path in args.logs:
        count = invalid = 0
        with path.open(errors="replace") as handle:
            for line in handle:
                position = line.find('{"kind":"client_frames"')
                if position >= 0:
                    try:
                        batch = json.loads(line[position:])
                        rows.extend({**dict(zip(batch["fields"], row)), "counter_schema": batch.get("schema", 0), "vsync": batch["vsync"], "width": batch["width"], "height": batch["height"]} for row in batch["rows"])
                        count += len(batch["rows"])
                    except json.JSONDecodeError:
                        invalid += 1
        sources.append({"path": str(path), "frames": count, "invalid_batches": invalid})
    groups = {}
    for row in rows:
        key = f"{'active' if row['active'] else 'background'};{row['width']}x{row['height']};vsync={row['vsync']};particles="
        key += "unavailable" if not row["counter_schema"] else "0" if row["particles"] == 0 else "1-63" if row["particles"] < 64 else "64+"
        groups.setdefault(key, []).append(row)
    print(json.dumps({"sources": sources, "scope": "regular visible renders; CPU/API timing, not GPU completion or physical display presentation; particle groups are association, not causation",
                      "groups": {name: summarize(group) for name, group in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
