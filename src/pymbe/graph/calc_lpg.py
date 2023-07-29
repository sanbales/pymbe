from typing import List, Tuple

import networkx as nx

from ..interpretation.m0_operators import (
    OPERATORS,
    evaluate_and_apply_atomic_binary,
    evaluate_and_apply_collect,
    evaluate_and_apply_dot,
    evaluate_and_apply_fre,
    evaluate_and_apply_literal,
    evaluate_and_apply_sum,
)
from ..model import InstanceDictType
from .lpg import SysML2LabeledPropertyGraph

COLLECTABLE_EXPRESSIONS = ("Expression", "FeatureReferenceExpression")


class CalculationGroup:
    """A graph to represent the active expression tree in a model."""

    def __init__(
        self,
        eig: nx.MultiDiGraph,
        instance_dict: InstanceDictType,
        calculation_list: List[Tuple[str, str, str]],
    ):
        self.graph = eig
        self.instance_dict = instance_dict

        self.solved_nodes = []
        self.unsolved_nodes = list(eig.nodes)

        self.calculation_list = calculation_list

        self.calculation_log = []

    def solve_graph(self, lpg: SysML2LabeledPropertyGraph):  # pylint: disable=too-many-statements
        # evaluating the expression tree is a reverse-order breadth-first search
        # (cover all children of a given node and then move up to that node)
        elements = lpg.model.elements
        for step in self.calculation_list:
            src, tgt, type_ = step
            src_data = lpg.model.elements[src]
            src_metatype = src_data.get("@type")
            source_instances = self.instance_dict.get(src)
            target_instances = self.instance_dict.get(tgt)
            if type_ in ("Assignment", "ValueBinding"):
                if not source_instances:
                    continue
                if target_instances:
                    for index, source in enumerate(source_instances):
                        target_instances[index][-1].value = source[-1].value
                else:
                    print(source_instances)
                    print(f"{elements[tgt].label}, id={tgt} has no elements")
            elif type_ == "Redefinition":
                if not source_instances:
                    continue
                for source_instance in source_instances:
                    for target_instance in target_instances:
                        if source_instance[:-1] == target_instance[:-1]:
                            target_instance[-1].value = source_instance[-1].value
            elif type_ == "Output":
                if "Literal" in src_metatype:
                    for index, seq in enumerate(source_instances):
                        evaluate_and_apply_literal(seq[-1], target_instances[index][-1])

                        self.calculation_log.append(
                            f"[Literal] {seq} is being assigned to {target_instances[index]}"
                        )

                elif src_metatype == "FeatureReferenceExpression":
                    for m0_obj in source_instances:
                        self.calculation_log.append(f"[FRE] {m0_obj} is expanding FRE...")

                        evaluate_and_apply_fre(m0_obj[-1], self.instance_dict)

                        for target_inst in target_instances:
                            self.calculation_log.append(f"[FRE]... result includes {target_inst}")

                elif src_metatype == "OperatorExpression":
                    if src_data.operator == "collect":
                        collect_sub_expressions = []
                        collect_sub_expression_results = []
                        collect_sub_inputs = []
                        for member in src_data["member"]:
                            if lpg.nodes[member["@id"]]["@type"] in COLLECTABLE_EXPRESSIONS:
                                collect_sub_expressions.append(lpg.nodes[member["@id"]])
                                collect_sub_expression_results.append(
                                    lpg.nodes[lpg.nodes[member["@id"]]["result"]["@id"]]
                                )

                        for member in src_data["input"]:
                            collect_sub_inputs.append(lpg.nodes[member["@id"]])

                        for index, m0_operator_seq in enumerate(source_instances):
                            input_point = None
                            input_instances = self.instance_dict[collect_sub_inputs[0]["@id"]]
                            for input_inst in input_instances:
                                if input_inst[0] == m0_operator_seq[0]:
                                    input_point = input_inst[-1]
                            path_point = None
                            input_instances = self.instance_dict[
                                collect_sub_expression_results[1]["@id"]
                            ]
                            for input_inst in input_instances:
                                if input_inst[0] == m0_operator_seq[0]:
                                    path_point = input_inst[-1]

                            print(
                                f"Calling collect with base = {m0_operator_seq[0]}, collection "
                                f"input {input_point}, and path input {path_point}"
                            )

                            if path_point is None or path_point.value is None:
                                print("Path point value is empty! {path_point}")
                            else:
                                evaluate_and_apply_collect(
                                    m0_operator_seq[0],
                                    m0_operator_seq[-1],
                                    self.instance_dict,
                                    input_point,
                                    path_point,
                                    target_instances[index][-1],
                                )

                    elif src_data["operator"] in OPERATORS:
                        for member in src_data["input"]:
                            collect_sub_inputs.append(lpg.nodes[member["@id"]])

                        plus_inputs = []

                        for member in src_data["input"]:
                            plus_inputs.append(lpg.nodes[member["@id"]])

                        for index, m0_operator_seq in enumerate(source_instances):
                            x_point = None
                            y_point = None
                            for input_index, input_ in enumerate(plus_inputs):
                                input_instances = self.instance_dict[input_["@id"]]
                                for input_inst in input_instances:
                                    if input_inst[0] == m0_operator_seq[0]:
                                        if input_index == 0:
                                            x_point = input_inst[-1]
                                        elif input_index == 1:
                                            y_point = input_inst[-1]

                            evaluate_and_apply_atomic_binary(
                                x_point, y_point, target_instances[index][-1], src_data["operator"]
                            )

                elif src_metatype == "InvocationExpression":
                    invoke_type = src_data.type[0]
                    if invoke_type.name == "sum":
                        sum_inputs = []

                        for member in src_data.input:
                            sum_inputs.append(member)

                        for index, m0_operator_seq in enumerate(source_instances):
                            input_point = None
                            input_instances = self.instance_dict[sum_inputs[0].get("@id")]
                            for input_inst in input_instances:
                                if input_inst[0] == m0_operator_seq[0]:
                                    input_point = input_inst[-1]

                            evaluate_and_apply_sum(
                                input_point,
                                target_instances[index][-1],
                            )

                elif src_metatype == "PathStepExpression":
                    collect_sub_expressions = []
                    collect_sub_expression_results = []
                    collect_sub_inputs = []
                    for member in src_data.member:
                        if member.get("@type") in COLLECTABLE_EXPRESSIONS:
                            collect_sub_expressions.append(member)
                            collect_sub_expression_results.append(member.result)

                    for input_ in src_data.input:
                        collect_sub_inputs.append(input_)

                    # Base sequence is there to filter as appropriate to the expression scope

                    # note that the PathStepExpression has two arguments in order, which are the
                    # steps in the path expressed as FeatureReferenceExpressions

                    # the input parameter is now expected to deliver instances of the start of the
                    # path in the PathStepExpression

                    for index, m0_operator_seq in enumerate(source_instances):
                        input_point = None
                        input_instances = self.instance_dict[collect_sub_inputs[0]._id]
                        for input_inst in input_instances:
                            if input_inst[0] == m0_operator_seq[0]:
                                input_point = input_inst[-1]
                        path_point = None
                        input_instances = self.instance_dict[collect_sub_expression_results[1]._id]
                        for input_inst in input_instances:
                            if input_inst[0] == m0_operator_seq[0]:
                                path_point = input_inst[-1]

                        self.calculation_log.append(
                            f"[PSE] Calling collect with base = {m0_operator_seq[0]}\n"
                            + f", collection input {input_point}\n, and path input {path_point}"
                        )

                        if path_point is None or path_point.value is None:
                            print("Path point value is empty! " + str(path_point))
                        else:
                            evaluate_and_apply_dot(
                                m0_operator_seq[0],
                                m0_operator_seq[-1],
                                self.instance_dict,
                                input_point,
                                path_point,
                                target_instances[index][-1],
                            )

    def solve_graph_with_openmdao(self, lpg):
        """
        Similar to solve graph, but use OpenMDAO to fill in values rather than literal expressions
        :param lpg:
        :return:
        """

        # Look through the calculation list and the relevant sequences, use this to build needed
        # alternative OpenMDAO problems - turn over passing of variables to OpenMDAO - ignore the
        # Assignment / Value Binding steps

        # Once problems are built, execute and bring results back to input / result parameter
        # sequences as needed
