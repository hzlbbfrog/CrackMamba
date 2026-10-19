"""Lower selective scan into standard ONNX Loop operators without custom plugins."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _lower_one(node, index):
    """Build the grouped real-valued Mamba recurrence with a carried float32 state."""
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    batch, channels, length, groups, states = (
        int(attributes[key]) for key in ("batch", "channels", "length", "groups", "states")
    )
    if channels % groups:
        raise ValueError("Scan channels must be divisible by scan groups")
    width = channels // groups
    prefix = f"SelectiveScan_{index}"
    u, delta, decay, source, readout, skip, bias = node.input
    nodes = []

    def name(suffix):
        return f"{prefix}_{suffix}"

    def constant(suffix, value, dtype=np.int64):
        output = name(suffix)
        tensor = numpy_helper.from_array(np.asarray(value, dtype=dtype), name=output)
        nodes.append(helper.make_node("Constant", [], [output], value=tensor, name=output))
        return output

    def operation(kind, inputs, suffix, **kwargs):
        output = name(suffix)
        nodes.append(helper.make_node(kind, inputs, [output], name=output, **kwargs))
        return output

    group_shape = constant("group_shape", [batch, groups, width, length])
    state_shape = [batch, groups, width, states]
    values = operation("Reshape", [u, group_shape], "values")
    bias_shape = constant("bias_shape", [1, channels, 1])
    bias3 = operation("Reshape", [bias, bias_shape], "bias")
    delta_bias = operation("Add", [delta, bias3], "delta_bias")
    positive_steps = operation("Softplus", [delta_bias], "softplus")
    steps = operation("Reshape", [positive_steps, group_shape], "steps")
    decay_shape = constant("decay_shape", [1, groups, width, states])
    decay4 = operation("Reshape", [decay, decay_shape], "decay")
    trip_count = constant("trip_count", length)
    condition = constant("condition", True, np.bool_)
    initial_state = constant("initial_state", np.zeros(state_shape, dtype=np.float32), np.float32)

    iteration, keep_going, state = name("iteration"), name("keep_going"), name("state")
    body_nodes = []

    def body_op(kind, inputs, suffix, **kwargs):
        output = name(suffix)
        body_nodes.append(helper.make_node(kind, inputs, [output], name=output, **kwargs))
        return output

    axis3, axis2 = name("axis3"), name("axis2")
    for axis_name, axis in ((axis3, 3), (axis2, 2)):
        body_nodes.append(helper.make_node("Constant", [], [axis_name], name=axis_name,
            value=numpy_helper.from_array(np.asarray([axis], dtype=np.int64))))
    value_i = body_op("Gather", [values, iteration], "value_i", axis=3)
    step_i = body_op("Gather", [steps, iteration], "step_i", axis=3)
    source_i = body_op("Gather", [source, iteration], "source_i", axis=3)
    readout_i = body_op("Gather", [readout, iteration], "readout_i", axis=3)
    value_i4 = body_op("Unsqueeze", [value_i, axis3], "value_i4")
    step_i4 = body_op("Unsqueeze", [step_i, axis3], "step_i4")
    source_i4 = body_op("Unsqueeze", [source_i, axis2], "source_i4")
    readout_i4 = body_op("Unsqueeze", [readout_i, axis2], "readout_i4")
    exponent = body_op("Mul", [step_i4, decay4], "exponent")
    transition = body_op("Exp", [exponent], "transition")
    carried = body_op("Mul", [transition, state], "carried")
    injection = body_op("Mul", [step_i4, source_i4], "injection")
    driven = body_op("Mul", [injection, value_i4], "driven")
    next_state = body_op("Add", [carried, driven], "next_state")
    weighted = body_op("Mul", [next_state, readout_i4], "weighted")
    output_i = body_op("ReduceSum", [weighted, axis3], "output_i", keepdims=0)
    next_condition = body_op("Identity", [keep_going], "next_condition")
    body = helper.make_graph(body_nodes, name("body"), [
        helper.make_tensor_value_info(iteration, TensorProto.INT64, []),
        helper.make_tensor_value_info(keep_going, TensorProto.BOOL, []),
        helper.make_tensor_value_info(state, TensorProto.FLOAT, state_shape),
    ], [
        helper.make_tensor_value_info(next_condition, TensorProto.BOOL, []),
        helper.make_tensor_value_info(next_state, TensorProto.FLOAT, state_shape),
        helper.make_tensor_value_info(output_i, TensorProto.FLOAT, [batch, groups, width]),
    ])
    final_state, time_major = name("final_state"), name("time_major")
    nodes.append(helper.make_node("Loop", [trip_count, condition, initial_state],
        [final_state, time_major], body=body, name=prefix))
    grouped_output = operation("Transpose", [time_major], "grouped_output", perm=[1, 2, 3, 0])
    flat_shape = constant("flat_shape", [batch, channels, length])
    scan_output = operation("Reshape", [grouped_output, flat_shape], "scan_output")
    skip_shape = constant("skip_shape", [1, channels, 1])
    skip3 = operation("Reshape", [skip, skip_shape], "skip")
    bypass = operation("Mul", [u, skip3], "bypass")
    nodes.append(helper.make_node("Add", [scan_output, bypass], list(node.output), name=name("output")))
    return nodes


def lower_selective_scans(model):
    """Replace all explicit bridge nodes and reject missing or unexpected scans."""
    result = []
    count = 0
    for node in model.graph.node:
        if node.domain == "crackmamba" and node.op_type == "SelectiveScan":
            result.extend(_lower_one(node, count))
            count += 1
        else:
            result.append(node)
    if count != 8:
        raise ValueError(f"Expected eight global selective scans, exported {count}")
    del model.graph.node[:]
    model.graph.node.extend(result)
    kept_imports = [item for item in model.opset_import if item.domain != "crackmamba"]
    del model.opset_import[:]
    model.opset_import.extend(kept_imports)
    onnx.checker.check_model(model, full_check=True)
    return model
