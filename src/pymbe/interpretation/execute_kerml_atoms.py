import copy
import json
from importlib import resources as lib_resources
from pathlib import Path
from typing import Any, Collection, Dict, List, Tuple, Union
from uuid import uuid4

import pymbe.api as pm
from pymbe.interpretation.indeterminate_boundaries import build_indefinite_boundaries
from pymbe.interpretation.occurrences_steps import is_feature_involving_self
from pymbe.interpretation.working_maps import FeatureTypeWorkingMap
from pymbe.metamodel import (
    assoc_metas,
    classifier_metas,
    connector_metas,
    datatype_metas,
    derive_inherited_featurememberships,
    feature_metas,
)
from pymbe.model import Element, Model
from pymbe.model_modification import *
from pymbe.query.metamodel_navigator import (
    does_behavior_have_write_features,
    get_effective_basic_name,
    get_effective_lower_multiplicity,
    get_feature_bound_values,
    get_finite_multiplicity_types,
    get_lower_multiplicity,
    get_more_general_types,
    get_most_specific_feature_type,
    get_upper_multiplicity,
    has_type_named,
    identify_connectors_one_side,
    is_multiplicity_one,
    is_multiplicity_specific_finite,
    is_type_undefined_mult,
)


class KermlForwardExecutor:
    """
    An execution class that applies the methods from the draft KerML Annex A for execution. This class has a goal of examining a 
    model and assigning values to the Features that are encountered as it traversals the model.
    """

    # The map that will record feature value assignments as the executor progresses.
    _working_map = None
    # The package where new elements will be created
    _working_package = None

    # some placeholder items for lives that are indeterminant in extent
    _indefinite_time_type = None
    _indefinite_space_type = None
    _indefinite_life_type = None

    # a log that is indexed by instance that walks through the executor steps taken
    _builder_log = {}

    # a log that walks thorugh the steps of traversal between elements
    _traversal_log : List[str]

    # a log that records the state of the value assignments at particular points
    _value_map_log = []

    # a tracker to index the logs
    _current_loc = ""

    def __init__(self, model, target_package):
        self._working_map = FeatureTypeWorkingMap(model=model)
        self._working_package = target_package
        (
            self._indefinite_time_type,
            self._indefinite_space_type,
            self._indefinite_life_type,
        ) = build_indefinite_boundaries(
            library_model=self._working_map._model._referenced_models[0],
            target_package=self._working_package,
            active_model=self._working_map._model,
        )

        self._traversal_log = []
        self._builder_log = {}

    def execute_from_classifier(self, classifier: Element):
        """
        Run the executor on the classifier provided as an argument

        :param classifier: The classifier to act as the top-level context of the
            execution.
        """

        self._generate_values_for_features_in_type(
            input_model=self._working_map._model,
            package_to_populate=self._working_package,
            type_to_value=classifier,
            atom_index=0,
            passed_featuring_type=None,
            passed_feature_path=[],
            passed_this=None,
            path_from_this=[],
        )

        self._working_map.cover_features_in_new_atoms(self._working_package)

    def _generate_values_for_features_in_type(
        self,
        input_model: Model,
        package_to_populate: Element,
        type_to_value: Element,
        atom_index: int,
        passed_featuring_type: Element,
        passed_feature_path: List[Element],
        passed_this: Element,
        path_from_this: List[Element],
    ):

        """
        Inspect the type and determine what values should be applied to its Features. This is a procedure
        that will be executed in nested Features, so information about the traversal path to reach the
        current Type under consideration is needed.

        :param input_model: The model from which to fetch additional information as needed during execution
        :param package_to_populate: The Package in which to add any generated new model elements
        :param type_to_value: The Type that will have the algorithm applied to it
        :param atom_index: Which of the parallel elements in a Type will be
            generated (e.g., wheel #1 or #2 for a bike's two wheels)
        :param passed_featuring_type: The type under which this solution will be considered
        :param passed_feature_path: How far into the nested path the current Type is
        :param passed_this: element to become the scope for this (passing down for suboccurrences)
        :param path_from_this: The full path of traversal from this to the current type_to_value
        """

        self._current_loc = str(type_to_value) + "." + str(passed_feature_path)
        self._builder_log.update({self._current_loc: []})
        #print(f"Applying Annex A atom algorithm to {str(type_to_value)}")
        self._builder_log[self._current_loc].append(f"Applying Annex A atom algorithm to {str(type_to_value)}")
        self._traversal_log.append(f"Stepping in on {str(type_to_value)}")

        new_classifier_instance = None

        top_portion = passed_this

        featuring_type = None

        working_path = []

        if type_to_value._metatype in classifier_metas() and not has_type_named(
            type_to_value, "FeatureWritePerformance"
        ):

            # Annex A Step 1 - create new atom from the classifier
            base_name = type_to_value.basic_name
            new_classifier_instance = build_from_classifier_pattern(
                owner=package_to_populate,
                name=f"My{base_name}",
                model=input_model,
                specific_fields={},
                metatype=type_to_value._metatype,
                superclasses=[type_to_value],
            )

            self._working_map._add_type_instance_to_map(new_classifier_instance)

            #print(
            #    f"Executing step 1. Working from {base_name} to create {new_classifier_instance}"
            #)

            self._builder_log[self._current_loc].append(
                f"**KERML ANNEX A Step 1** Working from {base_name} to create {new_classifier_instance}"
            )

            featuring_type = new_classifier_instance

            if top_portion is None:
                top_portion = new_classifier_instance

        # Use derived property 'feature' to get all features including the inherited ones
        candidate_features = type_to_value.feature

        # If the type to value is a Feature, need to get a Classifier to be its Featuring Type by convention

        if new_classifier_instance is None:
            featuring_type = passed_featuring_type
            working_path = passed_feature_path

        try:
            
            featured_meta = type_to_value.throughFeatureTyping[0]._metatype

            self._builder_log[self._current_loc].append(
                f"Type {type_to_value.basic_name} is has a type of metatype {featured_meta}."
            )

            #print(f"Type {type_to_value.basic_name} is has a type of metatype {featured_meta}.")
            if featured_meta in datatype_metas():
                #print(
                #    f"Type {type_to_value.basic_name} is typed by a datatype. Bypassing further elaboration."
                #)
                self._builder_log[self._current_loc].append(
                    f"Type {type_to_value.basic_name} is typed by a datatype. Bypassing further elaboration."
                )
                return None
        except:
            pass

        #print(
        #    f"...Found features {candidate_features} under the type to value {str(type_to_value)}."
        #)

        self._builder_log[self._current_loc].append(
            f"...Found features {candidate_features} under the type to value {str(type_to_value)}."
        )

        # Assure that there are specific evaluation passes in a given order on the list of Features underneath
        # the type to value

        passes = ["Non-connector Features", "FeatureWritePerformances", "Connector Features"]

        for pass_number, pass_kind in enumerate(passes):

            #print(f"Currently on pass {pass_kind} under {type_to_value}.")
            #print(f"Current values dict at this step is")
            #print(f"{self._working_map}")

            self._builder_log[self._current_loc].append(
                f"=={pass_kind}=="
            )
            self._value_map_log.append(f"Values dict log at start of pass {pass_kind} under {type_to_value}:")
            self._value_map_log.append(f"{self._working_map}")

            for cf in candidate_features:

                eval_feature, lower_mult, values_set_in_model = self._common_features_preprocess(
                    pass_kind=pass_kind,
                    type_instance=featuring_type,
                    candidate_feature=working_path + [cf],
                    top_portion=top_portion,
                )

                if not eval_feature:
                    continue

                # Pass-specific core steps

                if has_type_named(cf, "FeatureWritePerformance"):
                    self._process_feature_write_features(
                        type_instance=featuring_type,
                        candidate_feature=working_path + [cf],
                        passed_this=passed_this,
                        path_to_this=path_from_this + [cf],
                    )

                elif (
                    cf._metatype not in connector_metas() and pass_kind == "Non-connector Features"
                ):

                    self._process_nonconnector_features(
                        type_instance=featuring_type,
                        candidate_feature=working_path + [cf],
                        feature_multiplicity=lower_mult,
                        values_set_in_model=values_set_in_model,
                        top_portion=top_portion,
                        path_from_this=path_from_this + [cf],
                    )

                if cf._metatype in connector_metas() and pass_kind == "Connector Features":

                    self._process_connector_features(
                        type_instance=featuring_type,
                        candidate_feature=working_path + [cf],
                        feature_multiplicity=lower_mult,
                        values_set_in_model=values_set_in_model,
                        top_portion=top_portion,
                        passed_this=passed_this,
                        path_from_this=path_from_this + [cf],
                    )

        if type_to_value._metatype in datatype_metas():
            #print(
            #    f"Classifier {type_to_value.basic_name} is a datatype. Bypassing further elaboration."
            #)
            self._builder_log[self._current_loc].append(
                f"Classifier {type_to_value.basic_name} is a datatype. Bypassing further elaboration."
            )
            return None

        elif get_most_specific_feature_type(type_to_value) is not None:

            try:
                if get_most_specific_feature_type(type_to_value)._metatype in datatype_metas():
                    #print(
                    #    f"Classifier {type_to_value.basic_name} is a datatype or typed by one. Bypassing further elaboration."
                    #)
                    self._builder_log[self._current_loc].append(
                        f"Classifier {type_to_value.basic_name} is a datatype or typed by one. Bypassing further elaboration."
                    )
                    return None
            except:
                pass

        self._common_postprocess(
            featuring_type=featuring_type, candidate_features=candidate_features
        )

        #print(f"Current values dict is")
        #print(str(self._working_map))

        return None

    def _common_features_preprocess(
        self,
        pass_kind: str,
        type_instance: Element,
        candidate_feature: List[Element],
        top_portion: Element,
    ):

        """
        The common preprocess step focuses on common steps such as checking whether or not a given Feature has been
        declared to have a finite multiplicity, is redefined, or has values bound to it already.

        :param pass_kind: What the current evaluation pass is looking for (e.g., non-connectors or connectors)
        :param type_instance: the current type instance under which to place feature values
        :param candidate_feature: the feature being evaluated
        :param top_portion: the instance current considered for the "this" feature in the nested traversal
        return eval_feature, lower_mult, values_set_in_model
        """
        if (
            has_type_named(candidate_feature[-1], "FeatureWritePerformance")
            and not pass_kind == "FeatureWritePerformances"
        ):
            return (False, None, None)
        if (
            candidate_feature[-1]._metatype not in connector_metas()
            and not pass_kind == "Non-connector Features"
        ):
            if not has_type_named(candidate_feature[-1], "FeatureWritePerformance"):
                return (False, None, None)
        if (
            candidate_feature[-1]._metatype in connector_metas()
            and not pass_kind == "Connector Features"
        ):
            return (False, None, None)
        if get_effective_basic_name(candidate_feature[-1]) in (
            "thisPerformance",
            "dispatchScope",
            "this",
            "runToCompletionScope",
        ):

            #print(
            #    f"Adding value for top portioned feature {candidate_feature[-1]} to featuring type {type_instance}"
            #)
            self._builder_log[self._current_loc].append(
                f"Adding value for top portioned feature {candidate_feature[-1]} to featuring type {type_instance}"
            )
            self._working_map._add_atom_value_to_feature(
                type_instance, candidate_feature, top_portion
            )
            return (False, None, None)
        if get_effective_basic_name(candidate_feature[-1]) in ("self", "myself"):
            #print(
            #    f"Adding value for self-affiliated feature {candidate_feature[-1]} with effective name  "
            #    + f"{get_effective_basic_name(candidate_feature[-1])} to featuring type {type_instance}"
            #)
            self._builder_log[self._current_loc].append(
                f"Adding value for self-affiliated feature {candidate_feature[-1]} with effective name  "
                + f"{get_effective_basic_name(candidate_feature[-1])} to featuring type {type_instance}"
            )
            self._working_map._add_atom_value_to_feature(
                type_instance, candidate_feature, type_instance
            )
            return (False, None, None)

        # common pre-processing steps

        cf_name = get_effective_basic_name(candidate_feature[-1])

        lower_mult = get_effective_lower_multiplicity(candidate_feature[-1])

        if lower_mult > -1:
            # need to test multiplicity
            #print(
            #    f"...Found effective lower multiplicity of {cf_name} ({candidate_feature[-1]._id}) as {lower_mult}."
            #)
            self._builder_log[self._current_loc].append(
                f"...Found effective lower multiplicity of {cf_name} ({candidate_feature[-1]._id}) as {lower_mult}."
            )
        elif candidate_feature[-1]._metatype not in connector_metas():
            #print(
            #    f"...{cf_name} ({candidate_feature[-1]._id}) has unbounded multiplicity. Skipping."
            #)
            self._builder_log[self._current_loc].append(
                f"...{cf_name} ({candidate_feature[-1]._id}) has unbounded multiplicity. Skipping."
            )
            return (False, None, None)

        redefining_features = set(candidate_feature[-1].reverseRedefinition)
        cf_redefined = False

        for rf in redefining_features:
            #print(
            #    f"...Discovered that {candidate_feature[-1]} ({candidate_feature[-1]._id}) is redefined by {rf} ({rf._id})! Skipping."
            #)
            self._builder_log[self._current_loc].append(
                f"...Discovered that {candidate_feature[-1]} ({candidate_feature[-1]._id}) is redefined by {rf} ({rf._id})! Skipping."
            )
            cf_redefined = True

        if cf_redefined:
            return (False, None, None)

        # Step 2 - find Features with lower multiplicity > 0 that are not connectors

        # Look for existing values for the feature or previous atom assignment

        values_set_in_model = self._find_model_existing_values_for_feature(
            type_instance, candidate_feature, top_portion
        )

        # return True for having a Feature to further interpret
        return (True, lower_mult, values_set_in_model)

    def _process_feature_write_performances():

        pass

    def _process_nonconnector_features(
        self,
        type_instance: Element,
        candidate_feature: List[Element],
        feature_multiplicity: int,
        values_set_in_model: List[Element],
        top_portion: Element,
        path_from_this: List[Element],
    ):
        """
        Runs the execution rules specific to Features that are non-connectors. This will check to see if there are
        already values created for the feature. If not, it will call for n, where n = lower multiplicity of the Feature,
        values to be created and initiate depth-first search on further nested Features for execution.

        """

        handled_as_self_reference = is_feature_involving_self(candidate_feature[-1])

        if handled_as_self_reference:
            #print(
            #    f"{candidate_feature[-1]} is a self-involving reference. Will skip until handling rules are built."
            #)
            self._builder_log[self._current_loc].append(
                f"{candidate_feature[-1]} is a self-involving reference. Will skip until handling rules are built."
            )

        if get_effective_basic_name(candidate_feature[-1]) == "portionOfLife":
            self._process_portion_of_life_features(
                type_instance=type_instance,
                candidate_feature=candidate_feature,
                passed_this=top_portion,
                passed_path_to_this=path_from_this,
            )

            return None

        cf_name = get_effective_basic_name(candidate_feature[-1])

        if len(values_set_in_model) == 0 and handled_as_self_reference == False:
            # Step 3 - create new Atoms to go with the Features

            used_typ = get_most_specific_feature_type(candidate_feature[-1])

            if used_typ._metatype in datatype_metas():
                #print(
                #    f"**KERML ANNEX A Step 2** Identified {cf_name} ({candidate_feature[-1]._id}) "
                #    + f"as a non-connector Feature. No existing values found. "
                #    + f"This is a datatype. Skipping generation of values."
                #)
                self._builder_log[self._current_loc].append(
                    f"**KERML ANNEX A Step 2** Identified {cf_name} ({candidate_feature[-1]._id}) "
                    + f"as a non-connector Feature. No existing values found. "
                    + f"This is a datatype. Skipping generation of values."
                )
                return None
            else:
                #print(
                #    f"**KERML ANNEX A Step 2** Identified {cf_name} ({candidate_feature[-1]._id}) as a" +
                #    f"non-connector Feature. No existing values found. " +
                #    + f"Generating {feature_multiplicity} new values specializing type {used_typ}."
                #)
                self._builder_log[self._current_loc].append(
                    f"**KERML ANNEX A Step 2** Identified {cf_name} ({candidate_feature[-1]._id}) as a " +
                    f"non-connector Feature. No existing values found. "
                    + f"Generating {feature_multiplicity} new values specializing type {used_typ}."
                )

            for i in range(0, feature_multiplicity):
                self._inspect_feature_type_for_values(
                    input_model=self._working_map._model,
                    package_to_populate=self._working_package,
                    featuring_type=type_instance,
                    atom_index=i,
                    cf=candidate_feature,
                    top_portion=top_portion,
                    path_from_this=path_from_this,
                )

        return None

    def _process_connector_features(
        self,
        type_instance: Element,
        candidate_feature: List[Element],
        feature_multiplicity: int,
        values_set_in_model: List[Element],
        top_portion: Element,
        passed_this: Element,
        path_from_this: List[Element],
    ):
        
        """
        Execute on connector Features. This requires the inspection of ends to see if there
        are bound Features with values already. The multiplicity of the ends of the connector will also
        determine how to generate values for the connector.

        1-to-1 connectors with one finite end and one infinite end will have the infinite end multiplicity
        constrained by that of the finite end. Also, even if both ends are infinite but a bound Feature has
        values, this will determine the number of values of the other end.
        """

        #print(
        #    f"Executing step 4. Identified {get_effective_basic_name(candidate_feature[-1])} as a Connector"
        #)

        self._builder_log[self._current_loc].append(
            f"**KERML ANNEX A Step 4** Identified {candidate_feature[-1]} as a Connector"
        )

        # check the feature ends of the connector

        used_typ = get_most_specific_feature_type(candidate_feature[-1])

        connector_atoms = []

        connector_ends = candidate_feature[-1].throughEndFeatureMembership

        lm_end1 = get_effective_lower_multiplicity(connector_ends[0])
        lm_end2 = get_effective_lower_multiplicity(connector_ends[1])

        if lm_end1 == 1 and lm_end2 == 1:
            #print(
            #    f"**KERML ANNEX A Step 5** Identified {candidate_feature} as a Connector with 1-to-1 ends"
            #)

            self._builder_log[self._current_loc].append(
                f"**KERML ANNEX A Step 5** Identified {candidate_feature} as a Connector with 1-to-1 ends"
            )

            # Get the lower multiplicity of the Features the connector is bound to
            end_connected_mults = [
                get_effective_lower_multiplicity(connector_ends[i].throughReferenceSubsetting[0])
                for i in range(0, 2)
            ]

            for i, end_connected_mult in enumerate(end_connected_mults):

                #print(
                #    f"...Effective lower multiplicity of {connector_ends[i].throughReferenceSubsetting[0]}, bound to assoc feature {connector_ends[i]}, "
                #    + f"is {end_connected_mult}"
                #)

                self._builder_log[self._current_loc].append(
                    f"...Effective lower multiplicity of {connector_ends[i].throughReferenceSubsetting[0]},"
                    + f" bound to assoc feature {connector_ends[i]}, "
                    + f"is {end_connected_mult}"
                )

            # Check to see if the already built instances have a finite value

            #print(f"...Looking for connected end feature values filled in during execution.")

            self._builder_log[self._current_loc].append(
                f"...Looking for connected end feature values filled in during execution."
            )

            found_values = []

            ends_to_process = []

            # Look to the declared ends of the connector or its type
            for cf_ele in connector_ends:
                # if cf_ele.isEnd:
                ends_to_process.append(cf_ele)
            if used_typ is not None:
                for cf_ele in used_typ.throughEndFeatureMembership:
                    # if cf_ele.isEnd:
                    ends_to_process.append(cf_ele)

            for con_end in ends_to_process:
                bound_feature = con_end.throughReferenceSubsetting[0]
                bound_feature_values = self._working_map._get_atom_values_for_feature(
                    type_instance=type_instance,
                    feature_nesting=candidate_feature[0:-1] + [bound_feature],
                )

                if len(bound_feature_values) > 0:
                    #print(
                    #    f"...Found connected end feature values filled in during execution {bound_feature_values}!"
                    #)
                    self._builder_log[self._current_loc].append(
                        f"...Found connected end feature values filled in during execution {bound_feature_values}!"
                    )
                    found_values.append(len(bound_feature_values))

            number_to_make = max(end_connected_mults + found_values)

            #print(
            #    f"...Found that number of atoms to make for {candidate_feature} is {number_to_make}"
            #)
            self._builder_log[self._current_loc].append(
                f"...Found that number of atoms to make for {candidate_feature} is {number_to_make}"
            )

            feature_value_atoms = []

            used_typ = None

            for i in range(0, number_to_make):
                #print(
                #    f"**KERML ANNEX A Step 5b** Creating atom #{i + 1} to be value for {candidate_feature}"
                #)

                self._builder_log[self._current_loc].append(
                    f"**KERML ANNEX A Step 5b** Creating atom #{i + 1} to be value for {candidate_feature}"
                )

                typ = []
                used_typ = None
                used_name = candidate_feature[-1].basic_name
                used_metatype = "Association"

                if hasattr(candidate_feature[-1], "throughFeatureTyping"):
                    typ = candidate_feature[-1].throughFeatureTyping

                if len(typ) > 0:
                    used_typ = typ[0]
                    used_name = used_typ.basic_name
                    used_metatype = used_typ._metatype

                for con_end in ends_to_process:
                    #print(f"...Inspecting {con_end} for connected features.")
                    self._builder_log[self._current_loc].append(
                        f"...Inspecting {con_end} for connected features."
                    )
                    if len(con_end.throughReferenceSubsetting) > 0:
                        bound_feature = con_end.throughReferenceSubsetting[0]
                        bound_feature_type = bound_feature.throughFeatureTyping

                        #print(f"...Found connected feature {bound_feature}.")
                        self._builder_log[self._current_loc].append(
                            f"...Found connected feature {bound_feature}."
                        )

                        if len(bound_feature_type) > 0:
                            feature_used_type = bound_feature_type[0]
                            bound_feature_values = self._working_map._get_atom_values_for_feature(
                                type_instance=type_instance,
                                feature_nesting=candidate_feature[0:-1] + [bound_feature],
                            )

                            if (i + 1) > len(bound_feature_values):
                                #print(
                                #    f"Executing step 5b (1-to-1 variant). Creating atom #{i + 1} to "
                                #    + f"be value for {con_end} and also {bound_feature} to fill in rest of values."
                                #)
                                self._builder_log[self._current_loc].append(
                                    f"Executing step 5b (1-to-1 variant). Creating atom #{i + 1} to "
                                    + f"be value for {con_end} and also {bound_feature} to fill in rest of values."
                                )

                                # Case where the bound end is connected to a Write Performance
                                if has_type_named(bound_feature, "FeatureWritePerformance"):
                                    self._process_feature_write_features(
                                        type_instance=type_instance,
                                        candidate_feature=[bound_feature],
                                        passed_this=passed_this,
                                        path_to_this=path_from_this + [bound_feature],
                                    )

                                # Traverse on the bound feature (which should be compatible with the end feature)
                                # to generate values
                                else:
                                    self._inspect_feature_type_for_values(
                                        self._working_map._model,
                                        self._working_package,
                                        type_instance,
                                        i,
                                        candidate_feature[0:-1] + [bound_feature],
                                        top_portion,
                                        path_from_this + [bound_feature],
                                    )

                # Currently can only handle the binary connector case robustly
                if len(ends_to_process) == 2:
                    source_end = ends_to_process[0]
                    target_end = ends_to_process[1]

                    source_bound_feature = source_end.throughReferenceSubsetting[0]
                    target_bound_feature = target_end.throughReferenceSubsetting[0]

                    try:
                        source_atom = self._working_map._get_atom_values_for_feature(
                            type_instance=type_instance, feature_nesting=[source_bound_feature]
                        )[i]
                        # bound_features_to_atom_values_dict[source_bound_feature._id][i]
                    except KeyError:
                        raise KeyError(
                            f"...Failed to find atoms for {source_bound_feature} and connector {candidate_feature}."
                        )

                    try:
                        target_atom = self._working_map._get_atom_values_for_feature(
                            type_instance=type_instance, feature_nesting=[target_bound_feature]
                        )[i]
                    except KeyError:
                        raise KeyError(
                            f"...Failed to find atoms for {target_bound_feature} and connector {candidate_feature}."
                        )

                    #print(
                    #    f"...Typing atom association ends from {source_atom} to {target_atom} under "
                    #    + f"{used_name}{i + 1} to specialize {[ft.basic_name for ft in candidate_feature[-1].throughFeatureTyping]}"
                    #)
                    
                    self._builder_log[self._current_loc].append(
                        f"...Typing atom association ends from {source_atom} to {target_atom} under "
                        + f"{used_name}{i + 1} to specialize {[ft.basic_name for ft in candidate_feature[-1].throughFeatureTyping]}"
                    )

                    # check for typing

                    cn_types = candidate_feature[-1].throughFeatureTyping

                    if len(candidate_feature[-1].throughFeatureTyping) == 0:
                        pass

                    if used_name in ("HappensDuring", "HappensBefore"):
                        used_name = (
                            f"{used_name} for Connector between "
                            + f"{get_effective_basic_name(candidate_feature[-1].source[0])} and "
                            + f"{get_effective_basic_name(candidate_feature[-1].target[0])}"
                        )

                    new_cn_classifier = build_from_binary_assoc_pattern(
                        name=f"{used_name} {i + 1}",
                        source_role_name=connector_ends[0].basic_name,
                        target_role_name=connector_ends[1].basic_name,
                        source_type=source_atom,
                        target_type=target_atom,
                        model=self._working_map._model,
                        metatype=used_metatype,
                        owner=self._working_package,
                        superclasses=candidate_feature[-1].throughFeatureTyping,
                        specific_fields={},
                    )

                    self._working_map._add_atom_value_to_feature(
                        type_instance=type_instance,
                        feature_nesting=candidate_feature,
                        atom_value=new_cn_classifier,
                    )

    def _process_feature_write_features(
        self,
        type_instance: Element,
        candidate_feature: List[Element],
        passed_this: Element,
        path_to_this: List[Element],
    ):

        """
        Executing FeatureWritePerformances leads to the creation of new TimeSlices for the 
        occurrence that is having its Feature written to. The timeslices allow for discontinuities 
        in the value to be described in the executed model. The process will examine the FWP for 
        occurrence to slice, find the accessed feature, and find the desired value to be written.
        """

        # find onOccurrence in feature list

        features_in_fwp = candidate_feature[-1].throughFeatureMembership
        occurrence_bound = None
        accessed_type = None
        new_value = None

        #print(f"...Features found in FWP are {features_in_fwp}")

        self._builder_log[self._current_loc].append(
            f"...Features found in FWP are {features_in_fwp}"
        )

        for cf in features_in_fwp:
            if get_effective_basic_name(cf) == "onOccurrence" and len(cf.throughRedefinition) > 0:
                #print(f"...Found onOccurrence feature")
                self._builder_log[self._current_loc].append(
                    f"...Found onOccurrence feature"
                )
                occurrence_bound = get_feature_bound_values(cf)[0]
                #print(
                #    f"...Found a feature value bound to onOccurrence feature which is {occurrence_bound}"
                #)
                self._builder_log[self._current_loc].append(
                    f"...Found a feature value bound to onOccurrence feature which is {occurrence_bound}"
                )

                # expect that onOccurrence has nested features

                #print(f"...Features found in onOccurrence are {cf.feature}")
                self._builder_log[self._current_loc].append(
                    f"...Features found in onOccurrence are {cf.feature}"
                )

                for cf2 in cf.throughFeatureMembership:
                    if cf2.basic_name == "startingAt" and len(cf.throughRedefinition) > 0:
                        for cf3 in cf2.throughFeatureMembership:
                            if (
                                cf3.basic_name == "accessedFeature"
                                and len(cf.throughRedefinition) > 0
                            ):
                                accessed_type = cf3.throughSubsetting[0]
                                #print(f"...Found accessedFeature feature bound to {accessed_type}")
                                self._builder_log[self._current_loc].append(
                                    f"...Found accessedFeature feature bound to {accessed_type}"
                                )

        for cf in features_in_fwp:
            if (
                get_effective_basic_name(cf) == "replacementValues"
                and len(cf.throughRedefinition) > 0
            ):
                # TODO: Get the literal value here and then add it to the atom value for the feature using
                # nested feature. Working map also needs to be given the ability to generate bindings
                # between nested features using FeatureChainExpressions and LiteralExpressions
                for bound_val in cf.throughFeatureValue:
                    if bound_val._metatype == "FeatureReferenceExpression":
                        #print(f"...Found a feature value bound to replacement values feature {cf}")
                        self._builder_log[self._current_loc].append(
                            f"...Found a feature value bound to replacement values feature {cf}"
                        )
                        referred_item = bound_val.throughMembership[0]
                        feature_values_shared = True
                    elif "Literal" in bound_val._metatype:
                        new_value = bound_val.value
                        #print(f"...Found literal value for {cf} which is {new_value}")
                        self._builder_log[self._current_loc].append(
                            f"...Found literal value for {cf} which is {new_value}"
                        )

        # get the thing to timeslice

        time_sliced_occurrence = None

        try:
            time_sliced_occurrence = self._working_map._get_atom_values_for_feature(
                type_instance=type_instance, feature_nesting=[occurrence_bound]
            )

            #print(
            #    f"...Occurrence atom for {occurrence_bound} is found as {time_sliced_occurrence}"
            #)
            self._builder_log[self._current_loc].append(
                f"...Occurrence atom for {type_instance}.{occurrence_bound} is found as {time_sliced_occurrence}"
            )

        except KeyError:
            # need to make a new classifier here
            #print(f"...No occurrence atom for {occurrence_bound} found. Creating a new one.")

            self._builder_log[self._current_loc].append(
                f"...No occurrence atom for {occurrence_bound} found. Creating a new one."
            )

            type_to_value = get_most_specific_feature_type(occurrence_bound)
            base_name = type_to_value.basic_name
            new_classifier = build_from_classifier_pattern(
                owner=self._working_package,
                name=f"My{base_name}",
                model=self._working_map._model,
                specific_fields={},
                metatype=type_to_value._metatype,
                superclasses=[type_to_value],
            )
            time_sliced_occurrence = new_classifier
            self._working_map._add_atom_value_to_feature(
                type_instance=type_instance,
                feature_nesting=[occurrence_bound],
                atom_value=time_sliced_occurrence,
            )

        # create time slice for after this FWP is applied

        time_slice = None
        need_new_slice = True

        try:
            potential_atoms = self._working_package.throughOwningMembership
            occurrence_atoms = self._working_map._get_atom_values_for_feature(
                type_instance=type_instance, feature_nesting=[occurrence_bound]
            )
            for pa in potential_atoms:
                if (
                    pa.basic_name
                    == get_most_specific_feature_type(occurrence_bound).basic_name + "TimeSlice"
                ):
                    #print(f"Found previous time slice created for {time_sliced_occurrence}")
                    self._builder_log[self._current_loc].append(
                        f"Found previous time slice created for {time_sliced_occurrence}"
                    )
                    need_new_slice = False
                    time_slice = pa
        except KeyError:
            need_new_slice = True

        if need_new_slice and len(time_sliced_occurrence) > 0:
            #print(f"Creating a new time slice for {time_sliced_occurrence}")
            self._builder_log[self._current_loc].append(
                f"Creating a new time slice for {time_sliced_occurrence}"
            )
            new_slice = build_from_portion_pattern(
                owner=self._working_package,
                name_extension="TimeSlice",
                model=self._working_map._model,
                classifier_to_be_portioned=get_most_specific_feature_type(occurrence_bound),
                feature_to_be_set=[],
                feature_values=[],
                specific_fields={},
            )
            time_slice = new_slice
            # try:
            #    bound_features_to_atom_values_dict[occurrence_bound._id].append(new_slice)
            # except KeyError:
            #    bound_features_to_atom_values_dict.update({occurrence_bound._id: [new_slice]})

        # need to add time slice to this occurrence

        #print(
        #    f"Building feature with metatype {candidate_feature[-1]._metatype} under {time_sliced_occurrence[0]}"
        #    + f" with name after{candidate_feature[-1].basic_name.capitalize()}"
        #)
        self._builder_log[self._current_loc].append(
            f"Building feature with metatype {candidate_feature[-1]._metatype} under {time_sliced_occurrence[0]}"
            + f" with name after{candidate_feature[-1].basic_name.capitalize()}"
        )

        slice_feature = build_from_feature_pattern(
            owner=time_sliced_occurrence[0],
            name="after" + candidate_feature[-1].basic_name.capitalize(),
            model=self._working_map._model,
            specific_fields={},
            feature_type=time_slice,
            direction="",
            metatype="Feature",
            connector_end=False,
        )

        type_to_value = get_most_specific_feature_type(candidate_feature[-1])
        base_name = candidate_feature[-1].basic_name
        fwp_atom = build_from_classifier_pattern(
            owner=self._working_package,
            name=f"FeatureWritePerformanceFor{base_name.capitalize()}",
            model=self._working_map._model,
            specific_fields={},
            metatype=type_to_value._metatype,
            superclasses=[type_to_value],
        )

        self._working_map._add_atom_value_to_feature(
            type_instance=type_instance,
            feature_nesting=[candidate_feature[-1]],
            atom_value=fwp_atom,
        )

        #print(
        #    f"Writing literal value {new_value} to path {[path_to_this[0]] + [occurrence_bound] + [slice_feature] + [accessed_type]}"
        #)
        self._builder_log[self._current_loc].append(
            f"Writing literal value {new_value} to path {[path_to_this[0]] + [occurrence_bound] + [slice_feature] + [accessed_type]}"
        )

        self._working_map._add_atom_value_to_feature(
            type_instance=passed_this,
            feature_nesting=[path_to_this[0]]
            + [occurrence_bound]
            + [slice_feature]
            + [accessed_type],
            atom_value=new_value,
        )

        return None

    def _process_portion_of_life_features(
        self,
        type_instance: Element,
        candidate_feature: List[Element],
        passed_this: Element,
        passed_path_to_this: List[Element],
    ):
        
        """
        Determine of what Life the occurrence value to be generated will be a portion.
        """

        # TODO: Be sure this only applies to the top level of a decomposition of occurrences

        library_model = self._working_map._model._referenced_models[0]

        occurrence_ns = [
            library_model_ns
            for library_model_ns in library_model.ownedElement
            if library_model_ns.throughOwningMembership[0].declaredName == "Occurrences"
        ][0]

        ocurrences_eles = occurrence_ns.throughOwningMembership[0].throughOwningMembership

        life = None

        if type_instance == passed_this:

            for ocurrences_ele in ocurrences_eles:
                if ocurrences_ele._metatype in ("Class"):
                    if hasattr(ocurrences_ele, "declaredName"):
                        if ocurrences_ele.declaredName == "Life":
                            life = ocurrences_ele

            #print(
            #    f"Executing special rule for portionOfLife. Identified "
            #    + f"{get_effective_basic_name(candidate_feature[-1])} as a Feature pointing to a Life."
            #)

            self._builder_log[self._current_loc].append(
                f"Executing special rule for portionOfLife. Identified "
                + f"{get_effective_basic_name(candidate_feature[-1])} as a Feature pointing to a Life."
            )

            type_life_instance = build_from_classifier_pattern(
                owner=self._working_package,
                name="Life for " + get_effective_basic_name(type_instance),
                model=self._working_map._model,
                metatype="Class",
                superclasses=[life],
                specific_fields={},
            )

            self._working_map._add_atom_value_to_feature(
                type_instance=type_instance,
                feature_nesting=candidate_feature,
                atom_value=type_life_instance,
            )

            # will need to fill in the various fields to say the life spans indefinite time and space?

        else:
            set_of_features = self._working_map._working_dict[passed_this]
            #print(f"Looking up portionOfLife from parent division of {type_instance}")
            self._builder_log[self._current_loc].append(
                f"Looking up portionOfLife from parent division of {type_instance}"
            )
            for feat in set_of_features:
                feat_id = feat.split(".")[0]
                feat_ele = self._working_map._model.get_element(feat_id)
                if get_effective_basic_name(feat_ele) == "portionOfLife":

                    found_life = self._working_map._get_atom_values_for_feature(
                        type_instance=passed_this, feature_nesting=[feat_ele]
                    )
                    #print(f"Matched portion of life from {passed_this} for {type_instance}")
                    self._builder_log[self._current_loc].append(
                        f"Matched portion of life from {passed_this} for {type_instance}"
                    )
                    self._working_map._add_atom_value_to_feature(
                        type_instance=type_instance,
                        feature_nesting=candidate_feature,
                        atom_value=found_life[0],
                    )

        return None

    def _common_postprocess(self, featuring_type: Element, candidate_features: List[Element]):

        # Apply common steps for additional value assignments after the other approaches are 
        # applied to generate model values
        for candidate_feature in candidate_features:
            if candidate_feature._metatype == "Step":
                temp_subperformances = self._working_map._get_atom_values_for_feature(
                    type_instance=featuring_type, feature_nesting=[candidate_feature]
                )

                self._builder_log[self._current_loc].append(
                    f"Returning from dive on feature {candidate_feature}. This is a step and values that need to be added "
                    + f"to subperformances list. Values are {temp_subperformances}"
                )

                if len(temp_subperformances) > 0:
                    subperformances = []

                    for t_s in temp_subperformances:
                        subperformances.append(t_s)

                    for sub_perf in subperformances:
                        for inst_feature in featuring_type.feature:
                            if get_effective_basic_name(inst_feature) == "subperformances":
                                self._working_map._add_atom_value_to_feature(
                                    type_instance=featuring_type,
                                    feature_nesting=[inst_feature],
                                    atom_value=sub_perf,
                                )

    def _inspect_feature_type_for_values(
        self,
        input_model: Model,
        package_to_populate: Element,
        featuring_type: Element,
        atom_index: int,
        cf: List[Element],
        top_portion: Element,
        path_from_this: List[Element],
    ):
        
        """
        Look at the type of the Feature (currently expected to be a Classifier) to generate a new 
        atom for the covering values and also decide whether or not to traverse into further nested
        Features (inherited or owned) if they existing.

        :param input_model: 
        :param package_to_populate:
        :param featuring_type:
        :param atom_index:
        :param cf:
        :param top_portion:
        :parm path_from_this:
        """

        considered_type = get_most_specific_feature_type(cf[-1])

        new_ft_classifier = None

        type_name = "None"
        if considered_type is not None:
            type_name = considered_type.basic_name

        if has_type_named(cf, "Life"):
            #print("Need to implement a method to generate Life values for Occurrences.")
            self._builder_log[self._current_loc].append(
                "Need to implement a method to generate Life values for Occurrences."
            )
            return None

        cf_name = cf[-1].basic_name

        lower_features = cf[-1].feature

        # if cf._id not in values_dict_passed:

        new_ft_classifier = build_from_classifier_pattern(
            owner=package_to_populate,
            name=f"My{type_name}{atom_index + 1}",
            model=input_model,
            specific_fields={},
            metatype=considered_type._metatype,
            superclasses=[considered_type],
        )

        #print(
        #    f"**KERML ANNEX A Step 3a** Creating atom #{atom_index + 1} to be value for {cf}"
        #    + f" under {featuring_type} and specializing {considered_type}"
        #)
        self._builder_log[self._current_loc].append(
            f"**KERML ANNEX A Step 3a** Creating atom #{atom_index + 1} to be value for {cf}"
            + f" under {featuring_type} and specializing {considered_type}"
        )

        self._working_map._add_atom_value_to_feature(featuring_type, cf, new_ft_classifier)

        self._working_map._add_type_instance_to_map(new_ft_classifier)

        # the redefinitions and subsettings need to be recursively gathered

        for redef in cf[-1].throughRedefinition:
            #print(f"Found redefined feature for {cf_name}.")
            self._working_map._add_atom_value_to_feature(
                featuring_type, cf[0:-1] + [redef], new_ft_classifier
            )

        if len(lower_features) == 0:
            #print(f"...No lower features found for {cf}. Will finish descent here.")
            self._builder_log[self._current_loc].append(
                f"...No lower features found for {cf}. Will finish descent here."
            )
        elif len(lower_features) == 1:
            # check that the feature isn't just self
            if get_effective_basic_name(lower_features[0]) == "self":
                #print(f"...Only lower feature for {cf} is self. Will finish descent here.")
                self._builder_log[self._current_loc].append(
                    f"...Only lower feature for {cf} is self. Will finish descent here."
                )
                return None

        # for lf in lower_features:
        self._generate_values_for_features_in_type(
            input_model=input_model,
            package_to_populate=package_to_populate,
            type_to_value=cf[-1],
            atom_index=atom_index,
            passed_featuring_type=new_ft_classifier,
            passed_feature_path=[],
            passed_this=top_portion,
            path_from_this=path_from_this,
        )

        self._traversal_log.append(f"Stepping back after executing {str(cf[-1])}")

        return None

    def _find_model_existing_values_for_feature(
        self,
        type_instance: Element,
        feat_path: List[Element],
        passed_this: Element
    ):

        #print(f"...Looking to see if {feat_path} is bound to other feature values")
        self._builder_log[self._current_loc].append(
            f"...Looking to see if {feat_path} is bound to other feature values"
        )

        values_in_dict = []
        feature_values_shared = False

        feat = feat_path[-1]

        #print(f"...Feature values for {feat_path} are {feat.throughFeatureValue}")
        self._builder_log[self._current_loc].append(
            f"...Feature values for {feat_path} are {feat.throughFeatureValue}"
        )

        for bound_val in feat.throughFeatureValue:
            if bound_val._metatype == "FeatureReferenceExpression":
                #print(f"...Found a feature value bound to feature {feat}")
                self._builder_log[self._current_loc].append(
                    f"...Found a feature value bound to feature {feat}"
                )
                referred_item = bound_val.throughMembership[0]
                feature_values_shared = True
            elif "Literal" in bound_val._metatype:
                print(f"...Found literal value for {feat}")

        if feature_values_shared:
            #print(f"...Taking values from {referred_item} to match to values set for {feat})")
            self._builder_log[self._current_loc].append(
                f"...Taking values from {referred_item} to match to values set for {feat})"
            )
            try:
                #values_in_dict = self._working_map._get_atom_values_for_feature(
                #    type_instance, [referred_item]
                #)
                values_in_dict = self._working_map._get_atom_values_for_feature(
                    passed_this, [referred_item]
                )
                self._builder_log[self._current_loc].append(
                    f"...Looked for values in {passed_this} under {[referred_item]} and found {values_in_dict})"
                )
                if len(values_in_dict) > 0:
                    #values_in_local_dict = self._working_map._get_atom_values_for_feature(
                    #    type_instance, feat_path
                    #)
                    values_in_local_dict = self._working_map._get_atom_values_for_feature(
                        type_instance, feat_path
                    )
                    #print(f"...Found values {values_in_dict} to match to values set for {feat})")
                    self._builder_log[self._current_loc].append(
                        f"...Found values {values_in_dict} to match to values set for {type_instance}.{feat_path})"
                    )
                    # need to apply to redefined values also
                    try:
                        self._working_map._add_atom_value_to_feature(
                            type_instance=type_instance,
                            feature_nesting=feat_path,
                            atom_value=values_in_dict[0],
                        )
                        for feat_redef in feat_path[-1].throughRedefinition:
                            self._working_map._add_atom_value_to_feature(
                                type_instance=type_instance,
                                feature_nesting=[feat_redef],
                                atom_value=values_in_dict[0],
                            )
                    except KeyError:
                        self._builder_log[self._current_loc].append(
                            f"ERROR: Failed to assign {values_in_dict} to match to values set for {type_instance}.{feat_path})"
                        )
            except KeyError:
                pass

        return values_in_dict
