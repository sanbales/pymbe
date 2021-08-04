from dataclasses import dataclass, field
from typing import Any

from ..label import get_label
from ..model import Element, Model

# What visual representation to use for instances based on their M1 Metatype
REPRESENTATION_BY_METATYPE = dict(
    ConnectionUsage="line",
    # PartDefinition="rectangle",
    PartUsage="rounded-rectangle",
    # PortDefinition="rectangle",
    PortUsage="port",
)


@dataclass
class InterpretationSequence:
    """
    A class to represent a single sequence within a model interpretation.

    Objects of this class should support the drawing of interpretation
    diagrams and be the eventual target of validity checkers.
    """

    instances: tuple
    owning_entry: "InterpretationDictionaryEntry" = None

    def get_line_ends(self):
        # placeholder for using M1 reference to figure out what the right ends for the
        # connector line are
        if not self.owning_entry or self.owning_entry.draw_kind != "Line":
            return []

        line_ends = self.owning_entry.base.connectorEnd
        line_source, line_target = None, None
        for index, line_end in enumerate(line_ends):
            for entry in self.owning_entry.master_list:
                if entry.key == line_end._id:
                    end_interpretation = entry.value
                    for seq in end_interpretation:
                        for instance in seq.instances:
                            if instance == self.instances[-1]:
                                if index == 0:
                                    line_source = seq
                                if index == 1:
                                    line_target = seq

        return [line_source, line_target]

    def get_nesting_list(self):
        # placeholder for using M1 reference to figure out what the path of parent shapes are
        pass

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return f"({', '.join(map(str, self.instances))})"


class InterpretationSet(set):
    """
    A class to represent the set of sequences of an interpretation of a single M1 element
    """

    MAX_REPR_LENGTH = 10

    def __repr__(self):
        max_len = self.MAX_REPR_LENGTH
        if len(self) > max_len:
            return str(f"{list(self)[:max_len]}...")
        return super().__repr__()


@dataclass
class InterpretationDictionaryEntry:
    """
    A class to represent a key value pair for a master interpretation dictionary, which points
    from M1 user model elements to a set of sequences of atoms that are the interpretation
    """

    base: Element
    interprets: set
    master_list: dict = field(default_factory=dict)
    value: InterpretationSet = field(default_factory=InterpretationSet)

    @property
    def key(self):
        return self.base._id

    def __post_init__(self):
        for item in self.interprets:
            self.value.add(item)
            # link the sequence owning entry back here to leave a breadcrumb for
            # plotting, checking, etc.
            item.owning_entry = self
        # build hinting for diagram
        self.draw_kind = REPRESENTATION_BY_METATYPE.get(self.base._metatype)

    def __repr__(self):
        return f"Entry: <{get_label(self.base)}, {self.value}>"


@dataclass
class ValueHolder:
    """
    A class to represent instances of the attribution of real things in the M0 universe
    interpreted from the model. Sequences of instances and value holders are intended to
    follow the mathematical base semantics of SysML v2.

    Additionally, the value holders are meant to be variables in numerical analyses.
    """

    path: list
    name: str
    value: Any
    base_att: Element
    index: int

    holder_string: str = ""

    def __post_init__(self):
        if not self.holder_string:
            self.holder_string = ".".join(map(str, self.path + [f"{self.name}#{self.index}"]))

    def __repr__(self):
        value = self.value if self.value is not None else "unset"
        return f"{self.holder_string} ({value})"


class LiveExpressionNode:  # pylint: disable=too-few-public-methods
    """
    A class to be instances of expressions that can support computations in the M0 space.
    Also, this is where implementation code for a specific expression type can be placed
    are segments of a tree can be subsumed by a specific procedure.
    """

    def __init__(self, path, expr, base_att):
        # path is list of instance references
        self.holder_string = expr
        self.path = path
        self.base_att = base_att

    def __repr__(self):
        return self.holder_string


def shorten_name(name: str, shorten_pre_bake: dict = None) -> str:
    """
    Helper to get short names for reporting many instances

    :param name: Existing name to shorten
    :param shorten_pre_bake: A pre-computed dictionary mapping long names to custom short names
    :return: a shortened version of the input name if the input is longer than five characters
    """
    short_name = ""
    shorten_pre_bake = shorten_pre_bake or {}
    if not name:
        return "Unnamed"
    if len(name) > 5:
        if name in shorten_pre_bake:
            return shorten_pre_bake[name]
        space_place = name.find(" ")
        if space_place > -1:
            short_name = short_name + name[0]
            short_name = short_name + name[space_place + 1]
            next_space = name.find(" ", space_place + 1)
            while next_space > -1:
                short_name = short_name + name[next_space + 1]
                next_space = name.find(" ", next_space + 1)
            return short_name
    return name


def repack_instance_dictionaries(instance_dict: dict, mdl: Model):
    """
    Temporary method to repack the instance dictionaries into objects to be sure
    this is how we want things to work

    :param instance_dict: Completed instance dictionary
    :return: An object version of the instance dictionary
    """

    instance_list = []

    for key, sequence_set in instance_dict.items():
        new_set = {InterpretationSequence(seq) for seq in sequence_set}
        instance_list.append(
            InterpretationDictionaryEntry(mdl.elements[key], new_set, instance_list)
        )

    return instance_list
