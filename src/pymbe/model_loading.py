from collections import defaultdict
import networkx as NX
import math

class ModelingSession:
    """
    A class to contain a live, in-memory version of the model and support rapid execution of commands
    """

    def __init__(self, ele_list=None):
        self.lookup = ModelLookup()
        self.ele_list = ele_list
        self.graph_manager = GraphManager(session_handle=self)

    def thaw_json_data(self, jsons):
        self.lookup.memoize(jsons)
        self.graph_manager.build_graphs_from_data(jsons)

    def get_data_by_id(self, ele_id=''):
        trial = self.lookup.get_element_by_id(ele_id)
        if trial is None:
            for ele in self.ele_list:
                if ele['@id'] == ele_id:
                    self.lookup.memoize([ele])
                    return ele
        else:
            return trial

        raise ValueError("No element found with ID = " + str(ele_id))

    def get_name_by_id(self, ele_id=''):
        trial = self.get_data_by_id(ele_id=ele_id)
        if trial is not None and 'name' in trial:
            return trial['name']
        else:
            return None

    def get_signature_by_id(self, ele_id=''):
        return self.get_element_signature(self.get_data_by_id(ele_id=ele_id))

    def get_metaclass_by_id(self, ele_id=''):
        trial = self.get_data_by_id(ele_id=ele_id)
        if trial is not None and '@type' in trial:
            return trial['@type']
        else:
            return None

    def get_all_of_metaclass(self, metaclass_name=''):
        if metaclass_name in self.lookup.metaclass_lookup:
            return [self.get_data_by_id(this_id) for this_id in self.lookup.metaclass_lookup[metaclass_name]]
        else:
            return []

    def feature_lower_multiplicity(self, feature_id=''):
        feature = self.get_data_by_id(feature_id)
        if feature['multiplicity'] is not None:
            if '@id' in feature['multiplicity']:
                mult = self.get_data_by_id(feature['multiplicity']['@id'])
                if '@id' in mult['lowerBound']:
                    return self.get_data_by_id(mult['lowerBound']['@id'])['value']

        return 1

    def feature_upper_multiplicity(self, feature_id=''):
        feature = self.get_data_by_id(feature_id)
        if feature['multiplicity'] is not None:
            if '@id' in feature['multiplicity']:
                mult = self.get_data_by_id(feature['multiplicity']['@id'])
                if '@id' in mult['upperBound']:
                    return self.get_data_by_id(mult['upperBound']['@id'])['value']

        return 1

    def get_element_signature(self, element):
        if not 'relatedElement' in element:
            return element['name'] + ', id ' + element['@id'] + '[' + element['@type'] + ']'
        elif element['@type'] == 'FeatureTyping':
            typed_feature_name = self.get_name_by_id(element['typedFeature']['@id'])
            if typed_feature_name is None:
                typed_feature_name = ''
            type_name = self.get_name_by_id(element['type']['@id'])
            if type_name is None:
                type_name = ''
            return typed_feature_name + ', id ' + element['typedFeature']['@id'] + \
                ' - > ' + type_name + ', id ' + element['type']['@id'] + \
                ' <' + element['@type'] + '>'
        elif element['@type'] == 'FeatureMembership':
            typed_feature_name = self.get_name_by_id(element['memberFeature']['@id'])
            if typed_feature_name is None:
                typed_feature_name = ''
            owning_type_name = self.get_name_by_id(element['owningType']['@id'])
            if owning_type_name is None:
                owning_type_name = ''
            return typed_feature_name + ', id ' + element['memberFeature']['@id'] + \
                ' - > ' + owning_type_name + ', id ' + element['owningType']['@id'] + \
                ' <' + element['@type'] + '>'
        else:
            return 'Uncaptured relationship with metatype ' + element['@type']


class ModelLookup:
    """
    Support rapid return of commonly queried attributes such as name, id, and metatype
    """

    def __init__(self):
        self.id_memo_dict = {}
        self.metaclass_lookup = defaultdict(list)

    def memoize(self, elements: list):
        self.id_memo_dict.update({
            element['@id']: element
            for element in elements
        })
        types_mapping = defaultdict(list)
        _ = {
            types_mapping[element["@type"]].append(element["@id"])
            for element in elements
        }
        self.metaclass_lookup.update(types_mapping)

    def get_element_by_id(self, element_id: str = ""):
        return self.id_memo_dict.get(element_id, None)


class KernelReference:
    """
    Utility class to hold key information about the SysML v2 language
    """


class GraphManager:
    """
    Class to handle multiple syntactic graphs to support rapid analysis of the current model
    and also semantic interpretations
    """

    _TYPE_MAPPINGS = dict(
        Superclassing=dict(source="general", target="specific"),
        FeatureTyping=dict(source="type", target="typedFeature"),
        FeatureMembership=dict(source="owningType", target="memberFeature"),
        Redefinition=dict(source="redefiningFeature", target="redefinedFeature")
    )

    def __init__(self, session_handle=None):
        self.graph = NX.MultiDiGraph()
        self.banded_featuring_graph = NX.DiGraph()
        self.superclassing_graph = NX.DiGraph()
        self.feature_typing_graph = NX.DiGraph()
        self.part_featuring_graph = NX.DiGraph()
        self.attribute_featuring_graph = NX.DiGraph()
        self.redefinition_graph = NX.DiGraph()
        self.session = session_handle

    def build_graphs_from_data(self, elements: list):
        """
        Packs elements into utility syntax graphs for later processing
        :param ele_list: list of element JSONs to pack
        :return: nothing
        """

        for element in elements:
            element_type = element["@type"]
            mapping = self._TYPE_MAPPINGS.get(element_type, None)
            if mapping is None:
                continue

            source = element[mapping["source"]]["@id"]
            target = element[mapping["target"]]["@id"]


            if element['@type'] == 'Superclassing':
                general = element['general']['@id']
                specific = element['specific']['@id']
                self.superclassing_graph.add_node(general, name=self.session.get_name_by_id(ele_id=general))
                self.superclassing_graph.add_node(specific, name=self.session.get_name_by_id(ele_id=specific))
                self.superclassing_graph.add_edge(specific, general)

                self.banded_featuring_graph.add_node(general, name=self.session.get_name_by_id(ele_id=general))
                self.banded_featuring_graph.add_node(specific, name=self.session.get_name_by_id(ele_id=specific))
                self.banded_featuring_graph.add_edge(specific, general, kind='Superclassing')

            elif element['@type'] == 'FeatureTyping':
                typ = element['type']['@id']
                feature = element['typedFeature']['@id']

                # limit to part usages for now

                if self.session.get_metaclass_by_id(feature) == 'PartUsage':

                    self.feature_typing_graph.add_node(feature, name=self.session.get_name_by_id(ele_id=feature))
                    self.feature_typing_graph.add_node(typ, name=self.session.get_name_by_id(ele_id=typ))
                    self.feature_typing_graph.add_edge(feature, typ)

                    self.banded_featuring_graph.add_node(feature, name=self.session.get_name_by_id(ele_id=feature))
                    self.banded_featuring_graph.add_node(typ, name=self.session.get_name_by_id(ele_id=typ))
                    self.banded_featuring_graph.add_edge(typ, feature, kind='FeatureTyping^-1')

            elif element['@type'] == 'FeatureMembership':
                owner = element['owningType']['@id']
                feature = element['memberFeature']['@id']

                # limit to part usages for now

                if self.session.get_metaclass_by_id(feature) == 'PartUsage':
                    self.part_featuring_graph.add_node(owner, name=self.session.get_name_by_id(ele_id=owner))
                    self.part_featuring_graph.add_node(feature, name=self.session.get_name_by_id(ele_id=feature))
                    self.part_featuring_graph.add_edge(owner, feature)

                    self.banded_featuring_graph.add_node(feature, name=self.session.get_name_by_id(ele_id=feature))
                    self.banded_featuring_graph.add_node(owner, name=self.session.get_name_by_id(ele_id=owner))
                    self.banded_featuring_graph.add_edge(feature, owner, kind='FeatureMembership^-1')

                elif self.session.get_data_by_id(feature)['@type'] == 'AttributeUsage':
                    self.attribute_featuring_graph.add_node(feature, name=self.session.get_name_by_id(ele_id=feature))
                    self.attribute_featuring_graph.add_node(owner, name=self.session.get_name_by_id(ele_id=owner))
                    self.attribute_featuring_graph.add_edge(feature, owner, kind='FeatureMembership^-1')

            elif element['@type'] == 'Redefinition':
                redefined = element['redefinedFeature']['@id']
                redefining = element['redefiningFeature']['@id']

                if self.session.get_data_by_id(redefined)['@type'] == 'AttributeUsage':

                    self.redefinition_graph.add_node(redefined, name=self.session.get_name_by_id(ele_id=redefined))
                    self.redefinition_graph.add_node(redefining, name=self.session.get_name_by_id(ele_id=redefining))
                    self.redefinition_graph.add_edge(redefining, redefined, kind='Redefinition^-1')


    def get_feature_type_name(self, feature_id=''):
        types = list(self.feature_typing_graph.successors(feature_id))
        if len(types) > 1:
            return 'Multiple Names'
        elif len(types) == 1:
            return self.session.get_name_by_id(types[0])
        elif len(types) == 0:
            return 'Part'

    def get_feature_type(self, feature_id=''):
        types = list(self.feature_typing_graph.successors(feature_id))
        if len(types) > 1:
            return types
        elif len(types) == 1:
            return types
        elif len(types) == 0:
            return []

    def roll_up_lower_multiplicities(self):
        banded_roots = [self.session.get_data_by_id(node)
                        for node in self.banded_featuring_graph.nodes
                        if self.banded_featuring_graph.out_degree(node) == 0]

        part_multiplicity = {}

        for part_use in self.session.get_all_of_metaclass(metaclass_name='PartUsage'):
            corrected_mult = 0
            for part_tree_root in banded_roots:
                try:
                    part_path = NX.shortest_path(
                        self.banded_featuring_graph,
                        part_use['@id'],
                        part_tree_root['@id'])
                    # TODO: check that the path actually exists
                    corrected_mult = math.prod(
                        [self.session.feature_lower_multiplicity(feature_id=node)
                         for node in part_path])
                except NX.NetworkXNoPath:
                    pass
            part_multiplicity.update({part_use['@id']: corrected_mult})

        return part_multiplicity

    def roll_up_upper_multiplicities(self):
        banded_roots = [self.session.get_data_by_id(node)
                        for node in self.banded_featuring_graph.nodes
                        if self.banded_featuring_graph.out_degree(node) == 0]

        part_multiplicity = {}

        for part_use in self.session.get_all_of_metaclass(metaclass_name='PartUsage'):
            corrected_mult = 0
            for part_tree_root in banded_roots:
                try:
                    part_path = NX.shortest_path(
                        self.banded_featuring_graph,
                        part_use['@id'],
                        part_tree_root['@id'])
                    # TODO: check that the path actually exists
                    corrected_mult = math.prod(
                        [self.session.feature_upper_multiplicity(feature_id=node)
                         for node in part_path])
                except NX.NetworkXNoPath:
                    pass
            part_multiplicity.update({part_use['@id']: corrected_mult})

        return part_multiplicity

    def map_attributes_to_types(self):

        # Examine the superclassing and feature typing parts of the graph to see which types the attribute should
        # expect to be a member of
        pass

    def partition_abstract_type(self, abstract_type_id=''):
        specifics = list(self.banded_featuring_graph.predecessors(abstract_type_id))
        return specifics

    def get_att_literal_values(self, att_use=None):
        literal_values = []
        for att_member in att_use['ownedMember']:
            if att_member['@id'] in self.session.lookup.id_memo_dict:
                if self.session.lookup.id_memo_dict[att_member['@id']]['@type'] == 'LiteralReal':
                    literal_values.append(self.session.lookup.id_memo_dict[att_member['@id']])

        return literal_values