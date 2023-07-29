def test_derive_type_attribute(basic_load_files):
    level1 = basic_load_files["Level1"]

    level1_features = [
        feature for feature in list(level1.elements.values()) if feature._metatype == "Feature"
    ]

    bcf = [
        feature
        for feature in list(level1.elements.values())
        if feature._metatype == "Feature" and feature.declaredName == "Bare Classed Feature"
    ]

    assert bcf[0].type[0].declaredName == "My Bare Class"


def test_derive_ownedmember(basic_load_files):
    level1 = basic_load_files["Level1"]

    ns = [oe for oe in level1.ownedElement if oe._metatype == "Namespace"]

    assert len(ns[0].ownedMember) == 2
