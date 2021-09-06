import itertools
import random
from typing import Dict, List

from ..model import Element
from .interpretation import LiveExpressionNode, ValueHolder

# Adaptations to simplify dictionary production

# Random production of dictionaries through a couple of techniques:

# 1. Generate instances up to a specific quantity (a la classifier dictionaries)

# 2. Sub-select from other sets (a la feature dictionaries)

# In both cases, use a reference sequence to find the minimal length intepretations
# Both classifiers and features can be made this way, just difference of lengthh

VALUE_HOLDER_TYPES = ("AttributeDefinition", "AttributeUsage", "DataType")


def create_set_with_new_instances(
    sequence_template: List[Element],
    quantities: List[int],
    name_hints: Dict[str, str] = None,
) -> list:
    """
    Generate a list of lists with pre-set quantities and templates based on M1 model Types.

    New Instances will be constructed at each position in the sequence.
    :param sequence_template: Sequence of Types (full data) to use as data sources for
    new Instances
    :param quantities: Number of Instances to create for each sequence
    :param name_hints: Pre-made short names to support instance naming
    :return: A set of instances built into a Cartesian product based on a type sequence
    """
    individual_lists = []
    name_hints = name_hints or {}

    for index, m1_type in enumerate(sequence_template):
        new_list = []
        m1_metatype = m1_type._metatype
        m1_typename = m1_type.name
        for value_index in range(quantities[index]):
            if m1_metatype in VALUE_HOLDER_TYPES:
                new_list.append(
                    ValueHolder(
                        [],
                        m1_typename,
                        None,
                        m1_type,
                        value_index,
                    )
                )
            else:
                new_list.append(m1_type())
        individual_lists.append(new_list)

    # walk the sequence and generate an appropriately named instance
    if len(sequence_template) > 1:
        cartesian_of_lists = [
            list(cartesian) for cartesian in itertools.product(*individual_lists)
        ]
    else:
        cartesian_of_lists = [[individual] for individual in individual_lists[0]]
    return cartesian_of_lists


def extend_sequences_by_sampling(
    previous_sequences: list,
    lower_mult: int,
    upper_mult: int,
    sample_set: list,
    fallback_to_generate: bool,
    fallback_type: dict,
) -> list:
    """
    Extends a set of sequences by random numbers of instances drawn from a sample set

    :param previous_sequences: The set of sequences of a length n - 1 of desired
    :param lower_mult: lower bound on how many to draw from the sample set
    :param upper_mult: upper bound on how many to draw from the sample set
    :param sample_set: the set of Instances from which to draw to extend input sequences
    :param fallback_to_generate: Whether or not to make new instances if a sample set is
    not present
    :param fallback_type: The type to use as the fallback
    :return:
    """

    # FIXME: Need to be sure that each instance is drawn once even when multiple features
    #        have the same type!

    total_draw, draws_per = 0, []
    for _ in range(0, len(previous_sequences)):
        draw = random.randint(lower_mult, upper_mult)
        total_draw = total_draw + draw
        draws_per.append(draw)

    set_extended = []
    if not sample_set and fallback_to_generate:
        new_list = [fallback_type() for _ in range(total_draw)]

        last_draw = 0
        for index, seq in enumerate(previous_sequences):
            for pull in new_list[last_draw : last_draw + draws_per[index]]:
                new_seq = []
                new_seq = new_seq + seq
                new_seq.append(pull)

                # TODO: Look at making a generator instead

                set_extended.append(new_seq)

            last_draw = last_draw + draws_per[index]
    else:
        try:
            pulled_instances = random.sample(sample_set, total_draw)

            last_draw = 0
            for index, seq in enumerate(previous_sequences):
                for pull in pulled_instances[last_draw : last_draw + draws_per[index]]:
                    new_seq = []
                    new_seq = new_seq + seq
                    new_seq.append(pull)

                    # TODO: Look at making a generator instead

                    set_extended.append(new_seq)

                last_draw = last_draw + draws_per[index]
        except ValueError as exc:
            print(f"Sample Set is:\n\t{sample_set}")
            print(f"Previous sequences include:\n\t{previous_sequences}\n\t{draws_per}")
            raise ValueError(
                f"Tried to pull {total_draw} instances from a length of {len(sample_set)}"
            ) from exc

    return set_extended


def extend_sequences_with_new_expr(
    previous_sequences: List[Element],
    expr_string: str,
    expr: dict,
) -> list:

    new_sequences = []
    for seq in previous_sequences:
        new_holder = LiveExpressionNode(
            seq,
            expr_string,
            expr,
        )
        new_sequences.append(seq + [new_holder])

    return new_sequences


def extend_sequences_with_new_value_holder(
    previous_sequences: list,
    base_name: str,
    base_ele: dict,
) -> list:
    new_sequences = []

    for indx, seq in enumerate(previous_sequences):
        new_holder = ValueHolder(seq, base_name, None, base_ele, indx)
        new_sequences.append(seq + [new_holder])

    return new_sequences
