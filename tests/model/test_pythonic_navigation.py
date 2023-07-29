def test_pythonic_attributes(basic_load_files):
    """
    Test that primary attributes are loaded into elements such that they can be referenced as normal Python prpperties
    """

    level3 = basic_load_files["Level3"]

    literals = basic_load_files["Literals"]

    key_invar = [
        invariant
        for invariant in list(level3.elements.values())
        if invariant._metatype == "Invariant"
    ][0]

    assert hasattr(key_invar.ownedRelationship[0].ownedRelatedElement[0], "operator")
    assert key_invar.ownedRelationship[0].ownedRelatedElement[0].operator == "=="

    literal_test_literals = [
        literal
        for literal in list(literals.elements.values())
        if literal._metatype == "LiteralRational"
    ]

    assert hasattr(literal_test_literals[0], "value")
    assert literal_test_literals[0].value == 3.0

    assert True


def test_pythonic_references(basic_load_files):
    """
    Test that primary references are loaded into elements such that they can be referenced as normal Python prpperties
    """

    assert True


def test_pythonic_through_reverse(basic_load_files):
    """
    Test that 'through' and 'reverse' Pythonic properties work correctly
    """

    level1 = basic_load_files["Level1"]

    leve1_features = [
        feature for feature in list(level1.elements.values()) if feature._metatype == "Feature"
    ]
    leve1_feature_typings = [
        ft for ft in list(level1.elements.values()) if ft._metatype == "FeatureTyping"
    ]

    level3 = basic_load_files["Level3"]

    key_invar = [
        invariant
        for invariant in list(level3.elements.values())
        if invariant._metatype == "Invariant"
    ][0]

    assert (
        len(key_invar.ownedRelationship[0].ownedRelatedElement[0].throughParameterMembership) == 2
    )

    # assert same navigation across "through" rel as from relationship ends

    for ft in leve1_feature_typings:
        assert ft.source[0].throughFeatureTyping[0] == ft.target[0]

    test_ft_1 = [tf for tf in leve1_features if tf.declaredName == "Test Feature 1"]

    assert len(test_ft_1[0].reverseFeatureMembership) == 1
