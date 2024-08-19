def serialize_kerml_atom(atom_to_print):
    
    atom_string = "#atom\n"
    
    if hasattr(atom_to_print, "throughUnioning"):
        atom_string = ""
    
    if atom_to_print._metatype == "Classifier":
        atom_string = atom_string + "classifier "
    elif atom_to_print._metatype == "Behavior":
        atom_string = atom_string + "behavior "
    elif atom_to_print._metatype == "Association":
        atom_string = atom_string + "association "
    elif atom_to_print._metatype == "Succession":
        atom_string = atom_string + "succession "
    elif atom_to_print._metatype == "Structure":
        atom_string = atom_string + "struct "
    elif atom_to_print._metatype == "DataType":
        atom_string = atom_string + "datatype "
    
    atom_string = atom_string + atom_to_print.basic_name
    
    if len(atom_to_print.throughSubclassification) > 0:
        atom_string = atom_string + " specializes " + \
            ", ".join([sub.basic_name for sub in atom_to_print.throughSubclassification])
        
    if hasattr(atom_to_print, "throughUnioning"):
        atom_string = atom_string + " unions " + \
            ", ".join([sub.basic_name for sub in atom_to_print.throughUnioning])
    
    if len(atom_to_print.throughFeatureMembership) > 0:
        atom_string = atom_string + " {\n"
        for feature in atom_to_print.throughFeatureMembership:
            if feature._metatype == "Feature":
                atom_string = atom_string + "   feature "
            elif feature._metatype == "Step":
                atom_string = atom_string + "   step "
            elif feature._metatype == "Connector":
                atom_string = atom_string + "   connector "
            elif feature._metatype == "Succession":
                atom_string = atom_string + "   succession "
                
            if len(feature.throughFeatureTyping) > 0:
                atom_string = atom_string + feature.basic_name + " : " + feature.throughFeatureTyping[0].basic_name
            else:
                atom_string = atom_string + feature.basic_name
                
            if len(feature.throughRedefinition) > 0:
                atom_string = atom_string + " redefines " + \
                    ", ".join([sub.basic_name for sub in feature.throughRedefinition])
                
            if hasattr(feature, "throughFeatureChaining"):
                atom_string = atom_string + " chains " + \
                    str(feature.throughFeatureChaining[0])
                
            if len(feature.throughFeatureValue) > 0:
                atom_string = atom_string + " = " + \
                    str(feature.throughFeatureValue[0])
                
            atom_string = atom_string + ";\n"
        atom_string = atom_string + "}\n"
    elif hasattr(atom_to_print, "throughEndFeatureMembership"):
        atom_string = atom_string + " {\n"
        for feature in atom_to_print.throughEndFeatureMembership:
            if feature._metatype == "Feature":
                atom_string = atom_string + "   end feature "
            elif feature._metatype == "Step":
                atom_string = atom_string + "   step "
            elif feature._metatype == "Connector":
                atom_string = atom_string + "   connector "
                
            if hasattr(feature, "throughFeatureTyping"):
                atom_string = atom_string + feature.basic_name + " : " + feature.throughFeatureTyping[0].basic_name
            else:
                atom_string = atom_string + feature.basic_name
                
            if len(feature.throughRedefinition) > 0:
                atom_string = atom_string + " redefines " + \
                    ", ".join([sub.basic_name for sub in feature.throughRedefinition])
                
            atom_string = atom_string + ";\n"
                
        atom_string = atom_string + "}\n"
    else:
        atom_string = atom_string + ";\n"
        
    return atom_string