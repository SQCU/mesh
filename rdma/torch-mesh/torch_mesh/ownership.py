"""Frontend sharding declarations normalized to preparation-time page owners.

The frontend supplies types and placements; the retained numerical IR supplies
access relations. Neither record is an execution object.
"""
import itertools

from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.utils import _pytree as pytree


def from_spec(spec):
    """Keep a declared global type and every rank's ordered page selection."""
    from .partition import _rank_selection
    mesh = spec.mesh
    coordinates = itertools.product(*(range(mesh.size(i)) for i in range(mesh.ndim)))
    owners = {rank: dict(zip(('selection', 'shape'), _rank_selection(spec, rank)))
              for rank in coordinates}
    return dict(type=spec.tensor_meta, owners=owners,
                dimensions=tuple(i for i, placement in enumerate(spec.placements) if placement.is_shard()),
                reductions=tuple((i, placement) for i, placement in enumerate(spec.placements)
                                 if placement.is_partial()),
                mesh=mesh, spec=spec)


class _DeclaredTypes(ShardingPropagator):
    """Use the captured result types with the existing sharding registries."""
    def __init__(self, source, result_types):
        self.source, self.result_types = source, result_types

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _propagate_tensor_meta_non_cached(self, schema):
        return self.result_types


def propagate(operation, arguments, keywords, mesh, *, result_types):
    """Consume existing operator schemas without invoking numerical dispatch.

    Tensor leaves are DTensorSpecs, including explicit replicated input specs.
    Result types come from the captured graph; propagation does not infer them
    again by running the operation on another set of fake inputs.
    OutputSharding retains compound results and the required input schema. Its
    input redistributions remain declarations for the shared access compiler.
    Missing operator declarations raise in the upstream propagator; this adapter
    never substitutes replication. Ragged placements retain their configured
    shares when upstream rules change which tensor axis a mesh axis partitions.
    """
    from .partition import _keys, _plain, _retag, _schema
    specs = [value for value in pytree.tree_leaves((arguments, keywords)) if isinstance(value, DTensorSpec)]
    if any(spec.mesh != mesh for spec in specs):
        raise ValueError('operator input specs must name their enclosing ownership mesh')
    keys = _keys(specs)
    propagator = _DeclaredTypes(DTensor._op_dispatcher.sharding_propagator, result_types)
    schema_info = propagator.op_to_schema_info.get(operation)
    if schema_info is None:
        schema_info = propagator.op_to_schema_info_for_single_dim_strategy.get(operation)
    schema = OpSchema(operation, arguments, keywords, schema_info=schema_info)
    result = propagator.propagate_op_sharding_non_cached(_schema(schema, _plain))
    output = pytree.tree_map_only(DTensorSpec, lambda spec: _retag(spec, keys), result.output_spec)
    required = (_schema(result.redistribute_schema, lambda spec: _retag(spec, keys))
                if result.redistribute_schema is not None else None)
    return OutputSharding(output, required, needs_redistribute=result.needs_redistribute,
                          use_val_from_redistribute_schema=result.use_val_from_redistribute_schema)
