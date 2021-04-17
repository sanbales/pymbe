from ..api import Instance
import itertools
import random

# Adaptations to simplify dictionary production

# Random production of dictionaries through a couple of techniques:

# 1. Generate instances up to a specific quantity (a la classifier dictionaries)

# 2. Sub-select from other sets (a la feature dictionaries)

# In both cases, use a reference sequence to find the minimal length intepretations
#### Both classifiers and features can be made this way, just difference of lengths

def create_set_with_new_instances(sequence_template: list, quantities: list, name_hints: list) -> list:
    """
    Generate a tuple of tuples with pre-set quantities and templates based on M1 model Types.

    New Instances will be constructed at each position in the sequence.
    :param sequence_template: Sequence of Types (full data) to use as data sources for new Instances
    :param quantities: Number of Instances to create for each sequence
    :param name_hints: Pre-made short names to support instance naming
    :return: A set of instances built into a Cartesian product based on a type sequence
    """

    individual_lists = []

    # walk the sequence and generate an appropriately named instance

    for indx, m1_type in enumerate(sequence_template):

        new_list = []

        for m in range(0, quantities[indx]):

            new_instance = Instance(
                m1_type['name'],
                m,
                name_hints
            )

            new_list.append(new_instance)

        individual_lists.append(new_list)

    if len(sequence_template) > 1:
        cartesian_of_lists = []
        cartesian = itertools.product(*individual_lists)
        # TODO: There *must* be an easier way to turn the tuple from itertools into a list

        for cart in cartesian:
            line = []
            for item in cart:
                line.append(item)

            cartesian_of_lists.append(line)
    else:
        cartesian_of_lists = []
        for ind in individual_lists[0]:
            cartesian_of_lists.append([ind])

    return cartesian_of_lists

def extend_sequences_by_sampling(previous_sequences: list, lower_mult: int, upper_mult: int, sample_set: list) -> list:
    """
    Extends a set of sequences by random numbers of instances drawn from a sample set

    :param previous_sequences: The set of sequences of a length n - 1 of desired
    :param lower_mult: lower bound on how many to draw from the sample set
    :param upper_mult: upper bound on how many to draw from the sample set
    :param sample_set: the set of Instances from which to draw to extend input sequences
    :return:
    """

    set_extended = []

    total_draw = 0
    draws_per = []

    for n in range(0, len(previous_sequences)):
        draw = random.randint(lower_mult, upper_mult)
        total_draw = total_draw + draw
        draws_per.append(draw)

    pulled_instances = random.sample(sample_set, total_draw)

    last_draw = 0

    for indx, seq in enumerate(previous_sequences):

        for pull in pulled_instances[last_draw:last_draw+draws_per[indx]]:
            new_seq = []
            for step in seq:
                new_seq.append(step)
            for item in pull:
                new_seq.append(item)

            set_extended.append(new_seq)

        last_draw = last_draw + draws_per[indx]

    return set_extended