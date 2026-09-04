import json, math, statistics, time, urllib.request

def _median(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None

def _sum(values):
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) if finite else None

def _producer(sample, environment):
    rows = (sample.get("workload") or {}).get("producers") or []
    return [row for row in rows if (row.get("labels") or {}).get("environment") == environment]

def node_measurement(sample, environment):
    producers = _producer(sample, environment)
    capacity = ((sample.get("capacity") or {}).get("metrics") or {})
    machine = ((sample.get("machine") or {}).get("metrics") or {})
    lower_flops = _sum((row.get("fp32") or {}).get("lower_flops") for row in producers)
    upper_flops = _sum((row.get("fp32") or {}).get("upper_flops") for row in producers)
    lower_gflops = _sum((row.get("fp32") or {}).get("lower_gflops_s") for row in producers)
    upper_gflops = _sum((row.get("fp32") or {}).get("upper_gflops_s") for row in producers)
    lower_bytes = _sum((row.get("memory") or {}).get("lower_bytes") for row in producers)
    upper_bytes = _sum((row.get("memory") or {}).get("upper_bytes") for row in producers)
    lower_gbs = _sum((row.get("memory") or {}).get("lower_gbs") for row in producers)
    upper_gbs = _sum((row.get("memory") or {}).get("upper_gbs") for row in producers)
    peak = capacity.get("gpu_fp32_gflops")
    memory_peak = capacity.get("memory_gbs")
    intensity = lower_flops / lower_bytes if lower_flops is not None and lower_bytes else None
    roofline = min(float(peak), float(memory_peak) * intensity) if peak and memory_peak and intensity else None
    active = float(machine["gpu_active_pct"]) / 100 if machine.get("gpu_active_pct") is not None else None
    activity_upper = float(peak) * active if peak is not None and active is not None and producers else None
    upper_candidates = [value for value in (upper_gflops, activity_upper) if value is not None]
    combined_upper = (
        max(lower_gflops, min(upper_candidates)) if lower_gflops is not None
        else min(upper_candidates)
    ) if upper_candidates else None
    coordinates = {
        name: [
            (row.get("operations") or {}).get(name)
            for row in producers
            if (row.get("operations") or {}).get(name) is not None
        ]
        for name in ("players", "teams", "carts")
    }
    deadline_load = [
        float(row["elapsed_s"]) / float(row["deadline_s"])
        for row in producers
        if row.get("deadline_s") is not None and float(row["deadline_s"]) > 0
    ]
    local_only_load = [
        float((row.get("operations") or {})["local_only_plan_elapsed_s"]) / float(row["deadline_s"])
        for row in producers
        if (row.get("operations") or {}).get("local_only_plan_elapsed_s") is not None
        and row.get("deadline_s") is not None and float(row["deadline_s"]) > 0
    ]
    local_only_lower_load = [
        float((row.get("operations") or {})["local_only_plan_lower_s"]) / float(row["deadline_s"])
        for row in producers
        if (row.get("operations") or {}).get("local_only_plan_lower_s") is not None
        and row.get("deadline_s") is not None and float(row["deadline_s"]) > 0
    ]
    local_only_upper_load = [
        float((row.get("operations") or {})["local_only_plan_upper_s"]) / float(row["deadline_s"])
        for row in producers
        if (row.get("operations") or {}).get("local_only_plan_upper_s") is not None
        and row.get("deadline_s") is not None and float(row["deadline_s"]) > 0
    ]
    remote_request_rows = _sum(
        (row.get("operations") or {}).get("remote_request_row_mass")
        for row in producers
    )
    remote_output_rows = _sum(
        (row.get("operations") or {}).get("remote_output_row_mass")
        for row in producers
    )
    memory_lower = machine.get("memory_total_lower_gbs")
    memory_upper = machine.get("memory_total_upper_gbs")
    roles = sorted({role for row in producers if (role := (row.get("labels") or {}).get("host_role"))})
    producer_hosts = sorted({host for row in producers if (host := (row.get("labels") or {}).get("host"))})
    return {
        "name": sample.get("name"),
        "producers": len(producers),
        **{
            name: max((int(value) for value in values), default=None)
            for name, values in coordinates.items()
        },
        "fp32_lower_gflops_s": lower_gflops,
        "fp32_upper_gflops_s": combined_upper,
        "analytic_fp32_upper_gflops_s": upper_gflops,
        "activity_fp32_upper_gflops_s": activity_upper,
        "roofline_gflops_s": roofline,
        "roofline_fraction": lower_gflops / roofline if roofline and lower_gflops is not None else None,
        "roofline_lower_fraction": lower_gflops / roofline if roofline and lower_gflops is not None else None,
        "roofline_upper_fraction": min(1.0, combined_upper / roofline) if roofline and combined_upper is not None else None,
        "arithmetic_intensity_flops_byte": intensity,
        "workload_memory_lower_gbs": lower_gbs,
        "workload_memory_upper_gbs": upper_gbs,
        "machine_memory_lower_gbs": memory_lower,
        "machine_memory_upper_gbs": memory_upper,
        "machine_memory_lower_fraction": float(memory_lower) / float(memory_peak) if memory_lower is not None and memory_peak else None,
        "machine_memory_upper_fraction": float(memory_upper) / float(memory_peak) if memory_upper is not None and memory_peak else None,
        "gpu_active_fraction": active,
        "maximum_deadline_load": max(deadline_load, default=None),
        "mean_deadline_load": _median(deadline_load),
        "local_only_deadline_load": max(local_only_load, default=None),
        "local_only_deadline_lower_load": max(local_only_lower_load, default=None),
        "local_only_deadline_upper_load": max(local_only_upper_load, default=None),
        "remote_request_row_mass": remote_request_rows,
        "remote_output_row_mass": remote_output_rows,
        "remote_output_row_fraction": remote_output_rows / remote_request_rows if remote_output_rows is not None and remote_request_rows else None,
        "capacity": capacity,
        "roles": roles,
        "producer_hosts": producer_hosts,
        "role_hosts": {
            role: sorted({
                (row.get("labels") or {}).get("host")
                for row in producers
                if (row.get("labels") or {}).get("host_role") == role
                and (row.get("labels") or {}).get("host")
            })
            for role in roles
        },
        "lower_flops": lower_flops,
        "upper_flops": upper_flops,
        "lower_bytes": lower_bytes,
        "upper_bytes": upper_bytes,
    }

def compact_observation(payload, environment):
    latest = payload.get("latest")
    if latest is None:
        latest = {name: rows[-1] for name, rows in (payload.get("nodes") or {}).items() if rows}
    fabric = payload.get("fabric")
    live = None if not isinstance(fabric, dict) or "reachable" not in fabric else set(fabric.get("reachable") or ())
    latest = {
        name: sample for name, sample in latest.items()
        if sample.get("reachable") and (live is None or name in live)
    }
    return {
        "observed_at": (payload.get("fabric") or {}).get("observed_at", time.time()),
        "fabric": payload.get("fabric") or {},
        "nodes": {
            name: node_measurement(sample, environment)
            for name, sample in latest.items()
        },
    }

def producer_measures(payload, environment):
    latest = payload.get("latest")
    if latest is None:
        latest = {name: rows[-1] for name, rows in (payload.get("nodes") or {}).items() if rows}
    fabric = payload.get("fabric")
    live = None if not isinstance(fabric, dict) or "reachable" not in fabric else set(fabric.get("reachable") or ())
    latest = {
        name: sample for name, sample in latest.items()
        if sample.get("reachable") and (live is None or name in live)
    }
    return {
        name: [
            {
                key: row.get(key)
                for key in (
                    "name", "pid", "started_at", "sampled_at", "sequence",
                    "labels", "operations", "measures",
                )
            }
            for row in _producer(sample, environment)
            if row.get("measures")
        ]
        for name, sample in latest.items()
        if any(row.get("measures") for row in _producer(sample, environment))
    }

def summarize_points(samples, target_memory_fraction, bandwidth_node=None,
                     bandwidth_role=None, required_roles=None,
                     minimum_producer_nodes=1):
    required_roles = set(required_roles or ())
    grouped = {}
    unkeyed_sample_mass = 0
    for sample in samples:
        configured = sample.get("configuration") or {}
        coordinates = []
        for name in ("players", "teams", "carts"):
            values = [row.get(name) for row in sample["nodes"].values() if row.get(name) is not None]
            value = max((int(item) for item in values), default=configured.get(name))
            coordinates.append(None if value is None else int(value))
        if any(value is None for value in coordinates):
            unkeyed_sample_mass += 1
            continue
        grouped.setdefault(tuple(coordinates), []).append(sample)
    points = []
    for (players, teams, carts), rows in sorted(grouped.items()):
        names = sorted({name for row in rows for name in row["nodes"]})
        nodes = {}
        for name in names:
            values = [row["nodes"][name] for row in rows if name in row["nodes"]]
            fields = set().union(*(value.keys() for value in values))
            nodes[name] = {}
            for field in fields:
                if field in ("name", "capacity", "roles", "producer_hosts", "role_hosts") or not all(
                    value.get(field) is None or isinstance(value.get(field), (int, float))
                    for value in values
                ):
                    continue
                observations = [
                    float(value[field]) for value in values
                    if value.get(field) is not None and math.isfinite(float(value[field]))
                ]
                nodes[name][field] = max(observations) if field in (
                    "maximum_deadline_load", "local_only_deadline_load",
                    "local_only_deadline_lower_load", "local_only_deadline_upper_load",
                ) and observations else _median(observations)
                nodes[name][field + "_sample_mass"] = len(observations)
                nodes[name][field + "_variance"] = statistics.pvariance(observations) if observations else None
            nodes[name]["capacity"] = values[-1].get("capacity") or {}
            nodes[name]["roles"] = sorted({role for value in values for role in value.get("roles") or [] if role})
            nodes[name]["producer_hosts"] = sorted({
                host for value in values for host in value.get("producer_hosts") or [] if host
            })
            nodes[name]["role_hosts"] = {
                role: sorted({
                    host for value in values
                    for host in (value.get("role_hosts") or {}).get(role, ()) if host
                })
                for role in nodes[name]["roles"]
            }
        producer_names = [name for name, value in nodes.items() if int(value.get("producers") or 0) > 0]
        role_names = [
            name for name in producer_names
            if bandwidth_role in (nodes[name].get("roles") or [])
        ] if bandwidth_role else []
        memory_name = bandwidth_node if bandwidth_node in nodes else max(
            role_names or producer_names or nodes, key=lambda name: float((nodes[name].get("capacity") or {}).get("memory_gbs") or 0),
            default=None,
        )
        memory = nodes.get(memory_name, {})
        lo = memory.get("machine_memory_lower_fraction")
        hi = memory.get("machine_memory_upper_fraction")
        if lo is None or hi is None:
            memory_distance = math.inf
        else:
            memory_distance = max(float(lo) - target_memory_fraction, target_memory_fraction - float(hi), 0)
        producers = [nodes[name] for name in producer_names]
        roofline_lower = [value.get("roofline_lower_fraction") for value in producers if value.get("roofline_lower_fraction") is not None]
        roofline_upper = [value.get("roofline_upper_fraction") for value in producers if value.get("roofline_upper_fraction") is not None]
        roofline_intervals = [
            (value.get("roofline_lower_fraction"), value.get("roofline_upper_fraction"))
            for value in producers
            if value.get("roofline_lower_fraction") is not None
            and value.get("roofline_upper_fraction") is not None
        ]
        deadline_load = [value.get("maximum_deadline_load") for value in producers if value.get("maximum_deadline_load") is not None]
        local_only_load = [value.get("local_only_deadline_load") for value in producers if value.get("local_only_deadline_load") is not None]
        local_only_lower_load = [value.get("local_only_deadline_lower_load") for value in producers if value.get("local_only_deadline_lower_load") is not None]
        local_only_upper_load = [value.get("local_only_deadline_upper_load") for value in producers if value.get("local_only_deadline_upper_load") is not None]
        roles = {role for value in producers for role in value.get("roles") or []}
        role_nodes = {
            role: sorted(
                name for name in producer_names
                if role in (nodes[name].get("roles") or [])
            )
            for role in sorted(roles | required_roles)
        }
        producer_hosts = sorted({
            host for name in producer_names for host in nodes[name].get("producer_hosts") or ()
        })
        role_hosts = {
            role: sorted({
                host for name in producer_names
                for host in (nodes[name].get("role_hosts") or {}).get(role, ())
            })
            for role in sorted(roles | required_roles)
        }
        required_role_distinct_node_pair_mass = sum(
            left_node != right_node
            for left_index, left in enumerate(sorted(required_roles))
            for right in sorted(required_roles)[left_index + 1:]
            for left_node in role_nodes.get(left, ())
            for right_node in role_nodes.get(right, ())
        )
        required_role_distinct_host_pair_mass = sum(
            left_host != right_host
            for left_index, left in enumerate(sorted(required_roles))
            for right in sorted(required_roles)[left_index + 1:]
            for left_host in role_hosts.get(left, ())
            for right_host in role_hosts.get(right, ())
        )
        minimum_roofline = min(roofline_lower) if roofline_lower else None
        maximum_deadline_load = max(deadline_load) if deadline_load else None
        maximum_local_only_load = max(local_only_load) if local_only_load else None
        maximum_local_only_lower_load = max(local_only_lower_load) if local_only_lower_load else None
        maximum_local_only_upper_load = max(local_only_upper_load) if local_only_upper_load else None
        roofline_distance = max(
            (max(float(lower) - 1.0, 1.0 - float(upper), 0.0)
             for lower, upper in roofline_intervals),
            default=None,
        )
        role_distance = len(required_roles - roles)
        host_distance = max(
            max(0, int(minimum_producer_nodes) - len(producer_names)),
            max(0, int(minimum_producer_nodes) - len(producer_hosts)),
        )
        role_host_separation_distance = int(
            len(required_roles) > 1
            and required_role_distinct_node_pair_mass == 0
            and required_role_distinct_host_pair_mass == 0
        )
        components = {
            "memory": memory_distance if math.isfinite(memory_distance) else None,
            "roofline": roofline_distance,
            "deadline": None if maximum_deadline_load is None else max(0.0, maximum_deadline_load - 1.0),
            "local_only": None if maximum_local_only_lower_load is None else max(0.0, 1.0 - maximum_local_only_lower_load),
            "roles": float(role_distance),
            "hosts": float(host_distance),
            "role_host_separation": float(role_host_separation_distance),
        }
        performance_coordinates = [
            value for name, value in components.items()
            if name in ("memory", "roofline", "deadline", "local_only") and value is not None
        ]
        missing_performance_coordinates = [
            name for name in ("memory", "roofline", "deadline", "local_only")
            if components[name] is None
        ]
        target_squared_distance = (
            sum(value * value for value in components.values() if value is not None)
            / sum(value is not None for value in components.values())
            if performance_coordinates else None
        )
        points.append({
            "players": players,
            "teams": teams,
            "carts": carts,
            "nodes": nodes,
            "bandwidth_node": memory_name,
            "bandwidth_role": bandwidth_role,
            "memory_target_fraction": target_memory_fraction,
            "memory_target_distance": memory_distance if math.isfinite(memory_distance) else None,
            "minimum_roofline_fraction": min(roofline_lower) if roofline_lower else None,
            "minimum_roofline_lower_fraction": minimum_roofline,
            "minimum_roofline_upper_fraction": min(roofline_upper) if roofline_upper else None,
            "maximum_roofline_target_distance": roofline_distance,
            "maximum_deadline_load": maximum_deadline_load,
            "maximum_local_only_deadline_load": maximum_local_only_load,
            "maximum_local_only_deadline_lower_load": maximum_local_only_lower_load,
            "maximum_local_only_deadline_upper_load": maximum_local_only_upper_load,
            "required_roles": sorted(required_roles),
            "observed_roles": sorted(roles),
            "producer_nodes": producer_names,
            "producer_node_mass": len(producer_names),
            "producer_hosts": producer_hosts,
            "producer_host_mass": len(producer_hosts),
            "minimum_producer_nodes": int(minimum_producer_nodes),
            "role_nodes": role_nodes,
            "role_hosts": role_hosts,
            "required_role_distinct_node_pair_mass": required_role_distinct_node_pair_mass,
            "required_role_distinct_host_pair_mass": required_role_distinct_host_pair_mass,
            "role_distance": role_distance,
            "host_distance": host_distance,
            "role_host_separation_distance": role_host_separation_distance,
            "target_distance_components": components,
            "target_coordinate_mass": sum(value is not None for value in components.values()),
            "performance_coordinate_mass": len(performance_coordinates),
            "missing_performance_coordinates": missing_performance_coordinates,
            "missing_target_coordinates": [name for name, value in components.items() if value is None],
            "target_squared_distance": target_squared_distance,
            "samples": len(rows),
        })
    measured_points = [
        point for point in points if point["target_squared_distance"] is not None
    ]
    center = min(
        measured_points,
        key=lambda point: float(point["target_squared_distance"]),
        default=None,
    )
    return {"sample_mass": len(samples), "keyed_sample_mass": len(samples) - unkeyed_sample_mass,
            "unkeyed_sample_mass": unkeyed_sample_mass,
            "points": points, "target_center_observation": center,
            "objective": "squared_distance_to_memory_roofline_distributed_deadline_local_only_deadline_role_and_host_targets"}

class LiveOperatingProfile:
    def __init__(self, specification, teams, carts, ceiling, initial, environment, output):
        self.specification = dict(specification or {})
        self.environment = str(environment)
        self.output = output
        self.teams = max(1, int(teams))
        self.carts = max(0, int(carts))
        self.step = self.teams
        self.ceiling = max(self.step, int(ceiling))
        self.ceiling -= self.ceiling % self.step
        self.initial = min(self.ceiling, max(self.step, int(initial)))
        self.target = float(self.specification.get("memory_target_fraction", 0.5))
        self.samples_per_level = max(1, int(self.specification.get("samples_per_level", 3)))
        self.period = max(0.1, float(self.specification.get("sample_period", 2.0)))
        self.settle = max(0.0, float(self.specification.get("settle_seconds", self.period)))
        self.url = str(self.specification.get("observer", "http://127.0.0.1:8787/latest.json"))
        self.bandwidth_node = self.specification.get("bandwidth_node")
        self.bandwidth_role = self.specification.get("bandwidth_role")
        self.required_roles = tuple(self.specification.get("required_roles") or ())
        self.minimum_producer_nodes = int(self.specification.get("minimum_producer_nodes", 1))
        self.current = None
        self.lower = None
        self.upper = None
        self.search_exhausted_mass = 0
        self.search_boundary_measures = {
            "repeated_target_mass": 0,
            "engine_capacity_mass": 0,
            "operating_interval_mass": 0,
        }
        self.targets = []
        self.next_sample = 0.0
        self.level_started = 0.0
        self.level_sample_start = 0
        self.level_samples = 0
        self.samples = []
        self.events = []
        self.producer_measure_records = {}

    def _command(self, server, value):
        proc = server.get("process")
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(f"sv_cmd setbots {int(value)}\n")
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def _snap(self, value):
        value = min(self.ceiling, max(self.step, int(value)))
        return max(self.step, value - value % self.step)

    def _advance(self, server, now, level):
        level = self._snap(level)
        if level in self.targets:
            self._observe_search_boundary(server, "repeated_target_mass")
            return False
        self.current = level
        self.targets.append(level)
        sent = self._command(server, level)
        self.level_started = now
        self.level_sample_start = len(self.samples)
        self.level_samples = 0
        self.next_sample = now + self.settle
        self.events.append({"at": time.time(), "target_bots": level, "command_sent": sent})
        return True

    def _observe_search_boundary(self, server, coordinate):
        self.search_exhausted_mass = 1
        self.search_boundary_measures[coordinate] += 1
        center = summarize_points(
            self.samples, self.target, self.bandwidth_node,
            self.bandwidth_role, self.required_roles, self.minimum_producer_nodes,
        ).get("target_center_observation")
        if center is not None:
            self.current = self._snap(center["players"])
            sent = self._command(server, self.current)
            self.events.append({"at": time.time(), "operating_bots": self.current, "command_sent": sent})

    def _current_point(self):
        rows = self.samples[self.level_sample_start:]
        points = summarize_points(
            rows, self.target, self.bandwidth_node,
            self.bandwidth_role, self.required_roles, self.minimum_producer_nodes,
        )["points"]
        return min(points, key=lambda point: abs(point["players"] - self.current)) if points else None

    def _choose_next(self, server, now):
        point = self._current_point()
        if point is None:
            self.level_sample_start = len(self.samples)
            self.level_samples = 0
            self.next_sample = now + self.period
            self.events.append({"at": time.time(), "target_bots": self.current, "measurement_mass": 0})
            return
        distances = {
            value["players"]: (
                float(value["target_squared_distance"])
                if value["target_squared_distance"] is not None else math.inf
            )
            for value in summarize_points(
                self.samples, self.target, self.bandwidth_node,
                self.bandwidth_role, self.required_roles, self.minimum_producer_nodes,
            )["points"]
        }
        measured_distance = point.get("target_squared_distance")
        distance = float(measured_distance) if measured_distance is not None else math.inf
        self.events.append({
            "at": time.time(), "target_bots": self.current,
            "observed_players": point.get("players"),
            "target_squared_distance": measured_distance,
            "target_distance_components": point.get("target_distance_components"),
            "target_coordinate_mass": point.get("target_coordinate_mass"),
            "performance_coordinate_mass": point.get("performance_coordinate_mass"),
            "missing_target_coordinates": point.get("missing_target_coordinates"),
        })
        if point.get("missing_performance_coordinates"):
            self.level_sample_start = len(self.samples)
            self.level_samples = 0
            self.next_sample = now + self.period
            self.events.append({
                "at": time.time(), "target_bots": self.current,
                "measurement_acquisition": point.get("missing_performance_coordinates"),
            })
            return
        if self.upper is None:
            prior = {level: value for level, value in distances.items() if level < self.current}
            if prior and distance > min(prior.values()):
                self.upper = self.current
                self.lower = min(prior, key=prior.get)
            if self.current >= self.ceiling:
                self._observe_search_boundary(server, "engine_capacity_mass")
                return
            if self.upper is None:
                candidate = min(self.ceiling, max(self.current + self.step, self.current * 2))
                self._advance(server, now, candidate)
                return
        if self.current not in (self.lower, self.upper):
            if distance <= distances.get(self.lower, math.inf):
                self.lower = self.current
            else:
                self.upper = self.current
        if self.upper - self.lower <= self.step:
            self._observe_search_boundary(server, "operating_interval_mass")
            return
        candidate = self.lower + (self.upper - self.lower) // 2
        self._advance(server, now, candidate)

    def poll(self, server, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.current is None:
            self._advance(server, now, self.initial)
            return
        if self.search_exhausted_mass or now < self.next_sample:
            return
        try:
            with urllib.request.urlopen(self.url, timeout=self.period) as response:
                payload = json.load(response)
            measured = producer_measures(payload, self.environment)
            self.producer_measure_records.update(measured)
            sample = compact_observation(payload, self.environment)
            sample["target_bots"] = self.current
            sample["configuration"] = {
                "players": self.current,
                "teams": self.teams,
                "carts": self.carts,
            }
            sample["level_elapsed_s"] = now - self.level_started
            self.samples.append(sample)
            self.level_samples += 1
        except Exception as exc:
            self.events.append({"at": time.time(), "target_bots": self.current, "sample_error": f"{type(exc).__name__}: {exc}"})
        self.next_sample = now + self.period
        if self.level_samples >= self.samples_per_level:
            self._choose_next(server, now)

    def finish(self):
        summary = summarize_points(
            self.samples, self.target, self.bandwidth_node,
            self.bandwidth_role, self.required_roles, self.minimum_producer_nodes,
        )
        result = {
            "schema": 2,
            "environment": self.environment,
            "targets": self.targets,
            "teams": self.teams,
            "carts": self.carts,
            "team_quantum": self.step,
            "engine_capacity": self.ceiling,
            "initial_bots": self.initial,
            "search_lower": self.lower,
            "search_upper": self.upper,
            "search_exhausted_mass": self.search_exhausted_mass,
            "search_boundary_measures": self.search_boundary_measures,
            "target_memory_fraction": self.target,
            "bandwidth_node": self.bandwidth_node,
            "bandwidth_role": self.bandwidth_role,
            "required_roles": list(self.required_roles),
            "minimum_producer_nodes": self.minimum_producer_nodes,
            "samples_per_level": self.samples_per_level,
            "samples": self.samples,
            "events": self.events,
            "producer_measure_records": self.producer_measure_records,
            **summary,
        }
        with open(self.output, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return result

__all__ = [
    "LiveOperatingProfile", "compact_observation",
    "node_measurement", "producer_measures", "summarize_points",
]
