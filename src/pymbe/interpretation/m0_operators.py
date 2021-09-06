import operator as op

from ..interpretation.interpretation import LiveExpressionNode, ValueHolder
from ..model import Instance, InstanceDictType

# TODO: go through KerML spec for the remaining operators
OPERATORS = {
    "+": op.add,
    "-": op.sub,
    "*": op.mul,
    "/": op.truediv,
    "**": op.pow,
    "%": op.mod,
}


def sequence_dot_operator(left_item, right_side_seqs):
    if not (left_item and right_side_seqs):
        return []
    left_len = len(left_item)
    right_len = len(right_side_seqs[0])

    matched_items = []
    for right_item in right_side_seqs:
        if left_len != right_len:
            if str(left_item) == str(right_item[0 : (right_len - 1)]):
                matched_items.append(right_item)
        else:
            if str(left_item[1:None]) == str(right_item[0 : (right_len - 1)]):
                matched_items.append(right_item)

    return matched_items


def evaluate_and_apply_collect(
    base_scope: Instance,
    m0_expr: LiveExpressionNode,
    instance_dict: InstanceDictType,
    m0_collection_input: ValueHolder,
    m0_collection_path: ValueHolder,
    result_holder: ValueHolder,
) -> None:
    # print("Applying collect to " + str(m0_collection_input))
    # apply the dot operator
    path_result = []
    first_step = sequence_dot_operator([base_scope], m0_collection_input.value)
    for collect_seq in first_step:
        collect_match = sequence_dot_operator(collect_seq, m0_collection_path.value)[0]
        path_result.append(collect_match)
    final_answer = [coll[-1] for coll in path_result]
    result_holder.value = final_answer


def evaluate_and_apply_dot(
    base_scope: Instance,
    m0_expr: LiveExpressionNode,
    instance_dict: InstanceDictType,
    m0_collection_input: ValueHolder,
    m0_collection_path: ValueHolder,
    result_holder: ValueHolder,
) -> None:
    # print("Applying collect to " + str(m0_collection_input))
    # apply the dot operator
    path_result = []
    first_step = sequence_dot_operator([base_scope], m0_collection_input.value)
    for collect_seq in first_step:
        collect_match = sequence_dot_operator(collect_seq, m0_collection_path.value)[0]
        path_result.append(collect_match)
    final_answer = [coll[-1] for coll in path_result]
    result_holder.value = final_answer


def evaluate_and_apply_fre(
    m0_expr: LiveExpressionNode,
    instance_dict: InstanceDictType,
) -> list:
    """
    Evaluate a feature reference expression at m0, e.g., return the list of sequences
    :param m0_expr:
    :param instance_dict:
    :return:
    """
    referent_id = m0_expr.base_att["referent"]["@id"]
    if referent_id in instance_dict:
        fre_result = m0_expr.base_att["result"]["@id"]
        target_list = instance_dict[fre_result]
        for target in target_list:
            target[-1].value = instance_dict[referent_id]
        return instance_dict[referent_id]
    return []


def evaluate_and_apply_literal(m0_expr: LiveExpressionNode, m0_result: ValueHolder):
    """
    Evaluate a literal expression at m0, pushing the value to all instances of a
    viable result feature

    :param m0_expr:
    :param m0_result:
    """
    literal_value = m0_expr.base_att["value"]
    m0_result.value = literal_value


def evaluate_and_apply_sum(m0_expr: ValueHolder, m0_result: ValueHolder):
    total = 0
    if m0_expr.value is None:
        return
    for item in m0_expr.value:
        total += item.value if item.value is not None else 0
    m0_result.value = total


def evaluate_and_apply_atomic_binary(
    x_expr: ValueHolder, y_expr: ValueHolder, m0_result: ValueHolder, operator: str
):
    numbers = [0.0, 0.0]

    for index, expr in enumerate([x_expr, y_expr]):
        if isinstance(expr.value, (float, int)):
            numbers[index] = expr.value
        elif isinstance(expr.value, (list, tuple)):
            if isinstance(expr.value[0], (list, tuple)):
                numbers[index] = expr.value[0][-1].value
            elif isinstance(expr.value[0].value, (float, int)):
                numbers[index] = expr.value[0].value
        assert isinstance(
            numbers[index], (int, float)
        ), f"Should be a number, not {numbers[index]}"

    func = OPERATORS.get(operator)
    if not func:
        # Start by looking at existing operators at:
        # https://docs.python.org/3/library/operator.html#mapping-operators-to-functions
        raise NotImplementedError(f"Operator ({operator}) needs to be implemented")

    m0_result.value = func(*numbers)
