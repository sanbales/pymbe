# A set of tools to make interpretations easier to read
from typing import List

from ..label import get_label, get_label_for_id
from ..model import Model


def pprint_interpretation(interpretation: dict, model: Model, show_empty: bool = True) -> list:
    print_lines = []
    for key, val in interpretation.items():
        if show_empty or len(val) > 0:
            print_lines.append(
                get_label(model.elements[key]) + ", id = " + key + ", size = " + str(len(val))
            )
            short_list = []
            for indx, ind_val in enumerate(val):
                if indx < 5:
                    short_list.append(ind_val)
            if len(val) > 4:
                short_list.append([".."])

            print_lines.extend(short_list)
    return print_lines


def pprint_single_id_list(element_ids: List[str], model: Model) -> list:
    return [get_label_for_id(element_id, model) for element_id in element_ids]


def pprint_double_id_list(list_to_print: list, model: Model) -> list:
    print_lines = []
    for seq in list_to_print:
        seq_line = []
        for item in seq:
            seq_line.append(get_label_for_id(item, model))
        print_lines.append(seq_line)

    return print_lines


def pprint_edges(list_to_print: list, model: Model) -> list:
    print_lines = []
    for seq in list_to_print:
        seq_line = [get_label_for_id(seq[0], model), get_label_for_id(seq[1], model), seq[2]]
        print_lines.append(seq_line)

    return print_lines


def pprint_calc_steps(list_to_print: list, model: Model, signature_map: dict) -> list:
    print_lines = []
    for seq in list_to_print:
        left_sig = f"{get_label_for_id(seq[0], model)} <<{model.elements[seq[0]]._metatype}>>"
        right_sig = f"{get_label_for_id(seq[1], model)} <<{model.elements[seq[1]]._metatype}>>"
        if seq[0] in signature_map:
            left_sig = f"{signature_map[seq[0]]} <<{model.elements[seq[0]]._metatype}>>"
        if seq[1] in signature_map:
            right_sig = f"{signature_map[seq[1]]} <<{model.elements[seq[1]]._metatype}>>"
        left_label = f"{left_sig}"
        right_label = f"{right_sig}"
        seq_line = [left_label, right_label, seq[2]]
        print_lines.append(seq_line)

    return print_lines


def pprint_dict_keys(dict_to_print: dict, model: Model) -> dict:
    print_dict = {}
    for key, item in list(dict_to_print.items()):
        print_dict.update({get_label_for_id(key, model): item})

    return print_dict
