
def banded_graph_filter():
    """
    Generate library settings for a "banded graph" of types, generalization, and feature membership to capture
    both SysML v1 and v2 styled composition structures
    :return:
    """

    filter_dict = {
        'nodes': None,
        'node_types': None,
        'edges': None,
        'edge_types': ("Superclassing", "FeatureMembership", "FeatureTyping"),
        'reverse_edge_types': ("FeatureMembership", "FeatureTyping")
    }

    return filter_dict