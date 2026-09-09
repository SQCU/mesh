import hashlib
import json
import numpy as np


def compact_j_report(report):
    lens, oracle = report.get("j_lens", {}), report.get("j_oracle", {})
    strata = []
    for row in lens.get("coordinate_strata", []):
        covariance = row.get("cross_covariance")
        covariance_norms = covariance['row_norm'] if isinstance(covariance, dict) else np.linalg.norm(covariance, axis=-1) if covariance is not None else ()
        residual = row.get("j_to_source_feature_affine_projection", {}).get("residual_mean_square")
        variance = row.get("input_variance")
        projection = row.get("j_to_source_feature_affine_projection", {})
        schema = json.dumps([row.get("input_labels"), row.get("j_labels")], separators=(",", ":"))
        strata.append({"key": hashlib.sha256(schema.encode()).hexdigest()[:16], "j_labels": row.get("j_labels", []), "mass": row["mass"], "finite_mass": row["finite_atom_mass"],
                       "j_width": row["j_coordinates"], "feature_width": row["input_coordinates"], "j_variance": row.get("j_variance"),
                       "singular_values": None if projection.get("domain_singular_values") is None else np.asarray(projection["domain_singular_values"]).tolist(), "rank": projection.get("domain_numerical_rank"),
                       "coordinates": {"names": row.get("input_labels", []), "variance": variance,
                                       "covariance_norm": covariance_norms, "residual": residual}})
    outcomes = []
    for arm, group in oracle.get("policy_arm_measures", {}).items():
        for name, measure in group.get("outcome_measures", {}).items():
            outcomes.append({"arm": arm, "name": name, **{key: measure.get(key) for key in ("mass", "integral", "mean", "variance")},
                             "covariance_norm": float(np.linalg.norm(measure["j_covariance"])) if measure.get("j_covariance") is not None else None})
    return {"strata": strata, "outcomes": outcomes, "window": report.get("observation_window", {}),
            "joins": {key: oracle.get(key) for key in ("state_application_mass", "state_application_joined_mass", "delivery_mass", "delivery_joined_mass", "applied_mass", "applied_joined_mass", "event_mass", "event_joined_mass")}}


def page_j_report(report, key='', query='', offset=0, width=200):
    offset, width = max(int(offset), 0), max(int(width), 1)
    strata = report.get('strata', [])
    selected = next((row for row in strata if row['key'] == key), strata[0] if strata else None)
    rows = []
    for row in strata:
        result = {name: value for name, value in row.items() if name not in ('coordinates', 'j_labels', 'j_variance')}
        result['j_labels'] = row.get('j_labels', [])[:1]
        if row is selected:
            result['j_variance'] = None if row.get('j_variance') is None else np.asarray(row['j_variance'])[offset:offset + width].tolist()
            result['j_offset'] = offset
            coordinates = row.get('coordinates', {})
            if isinstance(coordinates, list):
                matches = [value for value in coordinates if query.lower() in value['name'].lower()]
                result['coordinates'] = matches[offset:offset + width]
                result['coordinate_matches'] = len(matches)
            else:
                names = coordinates.get('names', [])
                indices = [i for i, name in enumerate(names) if query.lower() in name.lower()] if query else range(len(names))
                result['coordinate_matches'] = len(indices)
                result['coordinates'] = [{'name': names[i], **{name: None if coordinates.get(name) is None or i >= len(coordinates[name]) else float(coordinates[name][i])
                    for name in ('variance', 'covariance_norm', 'residual')}} for i in indices[offset:offset + width]]
            result['coordinate_offset'] = offset
        rows.append(result)
    return {**{name: value for name, value in report.items() if name not in ('strata', 'model')},
            'strata': rows, 'selected_key': None if selected is None else selected['key'],
            'coordinate_window': {'offset': offset, 'width': width, 'operation': 'exact_slice'}}


def page_model(model, offset=0, width=200):
    rows = [{**row, 'j_width': len(row.get('j', [])), 'j_offset': offset,
             'j': np.asarray(row.get('j', []))[offset:offset + width].tolist()}
            for row in model.get('row_outputs', [])]
    coupling = model.get('coupling')
    factor = np.asarray(coupling) if coupling is not None else np.empty((0, 0))
    focus = factor.reshape(len(factor), -1)[:, offset:offset + width].tolist() if len(factor) else None
    return {**{name: value for name, value in model.items() if name not in ('row_outputs', 'j', 'j_labels', 'coupling')},
            'row_outputs': rows, 'coupling': focus,
            'coordinate_window': {'offset': offset, 'width': width, 'operation': 'exact_slice'}}
