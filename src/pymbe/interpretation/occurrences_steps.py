def is_feature_involving_self(feat):

    """
    Look to see if the feature is one of the special performance / occurrence
    features that includes at least the featuring type itself. Examples include:

    self
    timeEnclosedOccurrences
    timeCoincidentOccurrences
    spaceEnclosedOccurrences
    spaceTimeEnclosedOccurrences
    spaceTimeEnclosedPoints
    spaceTimeCoincidentOccurrences
    matingOccurrences - this one seems like an error
    portions
    portionOf
    timeSlices
    timeSliceOf
    snapshots
    startShot
    endShot
    spaceSlices
    spaceSliceOf
    dispatchScope
    thisPerformance
    """

    self_referencers = [
        "self",
        "sameLifeOccurrences",
        "timeEnclosedOccurrences",
        "timeCoincidentOccurrences",
        "spaceEnclosedOccurrences",
        "spaceTimeEnclosedOccurrences",
        "spaceTimeEnclosedPoints",
        "spaceTimeCoincidentOccurrences",
        "matingOccurrences",
        "portions",
        "portionOf",
        "timeSlices",
        "timeSliceOf",
        "snapshots",
        "startShot",
        "endShot",
        "spaceSlices",
        "spaceSliceOf",
        "spaceShots",
        "dispatchScope",
        "thisPerformance",
    ]

    return feat.basic_name in self_referencers


def respect_check_step_subperformance_spec_constraint():
    """
    KerML spec 8.4.4.7.2 - Steps
    Further, the checkStepEnclosedPerformanceSpecialization
    and checkStepSubperformanceSpecialization constraints require that a Step whose owningType is a
    Behavior or another Step specialize Performances::Performance::enclosedPerformance or, if it is
    composite, Performances::Performance::subperformance (see 9.2.6.2.13). Finally, the
    checkStepOwnedPerformanceSpecialization constraint requires that a composite Step whose
    owningType is a Structure or a Feature typed by a Structure specialize
    Objects::Object::ownedPerformance (see 9.2.5.2.7).
    """

    return None
