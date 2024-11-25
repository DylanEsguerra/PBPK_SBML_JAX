# uses C_p_brain and C_IS_brain which come from brain_sbml.py
import libsbml

def create_csf_model(params):
    """
    Create CSF compartment SBML model with equations for:
    - Unbound mAb in BCSFB endosomal compartment
    - Bound mAb in BCSFB endosomal compartment
    - Lateral Ventricle
    - Third/Fourth Ventricle
    - Cisterna Magna
    - Subarachnoid Space
    
    Args:
        params (dict): Parameters including volumes, flow rates, etc.
    """
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("CSF_Model")
    model.setTimeUnits("hour")

    # Add compartments
    bcsfb_unbound = model.createCompartment()
    bcsfb_unbound.setId("BCSFB_unbound")
    bcsfb_unbound.setConstant(True)
    bcsfb_unbound.setSize(params["V_BCSFB_brain"])
    bcsfb_unbound.setUnits("litre")
    
    bcsfb_bound = model.createCompartment()
    bcsfb_bound.setId("BCSFB_bound")
    bcsfb_bound.setConstant(True)
    bcsfb_bound.setSize(params["V_BCSFB_brain"])
    bcsfb_bound.setUnits("litre")
    
    lateral_ventricle = model.createCompartment()
    lateral_ventricle.setId("LV")
    lateral_ventricle.setConstant(True)
    lateral_ventricle.setSize(params["V_LV_brain"])
    lateral_ventricle.setUnits("litre")
    
    third_fourth_ventricle = model.createCompartment()
    third_fourth_ventricle.setId("TFV")
    third_fourth_ventricle.setConstant(True)
    third_fourth_ventricle.setSize(params["V_TFV_brain"])
    third_fourth_ventricle.setUnits("litre")
    
    cisterna_magna = model.createCompartment()
    cisterna_magna.setId("CM")
    cisterna_magna.setConstant(True)
    cisterna_magna.setSize(params["V_CM_brain"])
    cisterna_magna.setUnits("litre")
    
    subarachnoid_space = model.createCompartment()
    subarachnoid_space.setId("SAS")
    subarachnoid_space.setConstant(True)
    subarachnoid_space.setSize(params["V_SAS_brain"])
    subarachnoid_space.setUnits("litre")

    # Add coupling parameters (time-varying concentrations)
    coupling_params = [
        # From brain model
        ("C_p_brain", params["C_p_brain_0"]),      # Brain plasma
        ("C_IS_brain", params["C_IS_brain_0"]),    # Brain ISF
        
        # CSF compartment concentrations
        ("C_BCSFB_unbound_brain", params["C_BCSFB_unbound_brain_0"]),  # BCSFB unbound
        ("C_BCSFB_bound_brain", params["C_BCSFB_bound_brain_0"]),      # BCSFB bound
        ("C_LV_brain", params["C_LV_brain_0"]),                        # Lateral ventricle
        ("C_TFV_brain", params["C_TFV_brain_0"]),                      # Third/Fourth ventricle
        ("C_CM_brain", params["C_CM_brain_0"]),                        # Cisterna magna
        ("C_SAS_brain", params["C_SAS_brain_0"])                       # Subarachnoid space
    ]
    
    for param_id, initial_value in coupling_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setConstant(False)  # Will be updated during simulation
        param.setValue(initial_value)
        param.setUnits("mole_per_litre")

    # Add species
    c_bcsfb_unbound = model.createSpecies()
    c_bcsfb_unbound.setId("C_BCSFB_unbound_brain")
    c_bcsfb_unbound.setCompartment("BCSFB_unbound")
    c_bcsfb_unbound.setInitialConcentration(params["C_BCSFB_unbound_brain_0"])
    c_bcsfb_unbound.setBoundaryCondition(False)
    c_bcsfb_unbound.setHasOnlySubstanceUnits(False)
    c_bcsfb_unbound.setConstant(False)
    c_bcsfb_unbound.setUnits("mole_per_litre")
    
    c_bcsfb_bound = model.createSpecies()
    c_bcsfb_bound.setId("C_BCSFB_bound_brain")
    c_bcsfb_bound.setCompartment("BCSFB_bound")
    c_bcsfb_bound.setInitialConcentration(params["C_BCSFB_bound_brain_0"])
    c_bcsfb_bound.setBoundaryCondition(False)
    c_bcsfb_bound.setHasOnlySubstanceUnits(False)
    c_bcsfb_bound.setConstant(False)
    c_bcsfb_bound.setUnits("mole_per_litre")
    
    c_lv = model.createSpecies()
    c_lv.setId("C_LV_brain")
    c_lv.setCompartment("LV")
    c_lv.setInitialConcentration(params["C_LV_brain_0"])
    c_lv.setBoundaryCondition(False)
    c_lv.setHasOnlySubstanceUnits(False)
    c_lv.setConstant(False)
    c_lv.setUnits("mole_per_litre")
    
    c_tfv = model.createSpecies()
    c_tfv.setId("C_TFV_brain")
    c_tfv.setCompartment("TFV")
    c_tfv.setInitialConcentration(params["C_TFV_brain_0"])
    c_tfv.setBoundaryCondition(False)
    c_tfv.setHasOnlySubstanceUnits(False)
    c_tfv.setConstant(False)
    c_tfv.setUnits("mole_per_litre")
    
    c_cm = model.createSpecies()
    c_cm.setId("C_CM_brain")
    c_cm.setCompartment("CM")
    c_cm.setInitialConcentration(params["C_CM_brain_0"])
    c_cm.setBoundaryCondition(False)
    c_cm.setHasOnlySubstanceUnits(False)
    c_cm.setConstant(False)
    c_cm.setUnits("mole_per_litre")
    
    c_sas = model.createSpecies()
    c_sas.setId("C_SAS_brain")
    c_sas.setCompartment("SAS")
    c_sas.setInitialConcentration(params["C_SAS_brain_0"])
    c_sas.setBoundaryCondition(False)
    c_sas.setHasOnlySubstanceUnits(False)
    c_sas.setConstant(False)
    c_sas.setUnits("mole_per_litre")

    # Add regular parameters
    required_params = [
        # Volumes
        "V_BCSFB_brain", "V_LV_brain", "V_TFV_brain", 
        "V_CM_brain", "V_SAS_brain", "V_ES_brain",
        
        # Flow rates
        "Q_CSF_brain", "Q_ISF_brain", "CLup_brain",
        
        # Fractions and coefficients
        "f_BBB", "f_BCSFB", "FR",
        "sigma_V_BCSFB", "sigma_L_SAS",
        
        # Kinetic parameters
        "kon_FcRn", "koff_FcRn", "kdeg",
        "FcRn_free_BCSFB", "f_LV",
        
        # Initial values for coupling parameters
        "C_BCSFB_unbound_brain_0", "C_BCSFB_bound_brain_0",
        "C_LV_brain_0", "C_TFV_brain_0", "C_CM_brain_0", "C_SAS_brain_0",
        "C_p_brain_0", "C_IS_brain_0"  
    ]

    # Verify all required parameters are present
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

    # Add parameters
    for param_id in required_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(params[param_id])
        param.setConstant(True)

    # Add rate rules
    # BCSFB Unbound equation
    bcsfb_unbound_rule = model.createRateRule()
    bcsfb_unbound_rule.setVariable("C_BCSFB_unbound_brain")
    
    bcsfb_unbound_eq = (
        f"CLup_brain * f_BCSFB * V_ES_brain * C_p_brain + "
        f"f_LV * CLup_brain * (1 - f_BBB) * V_ES_brain * C_LV_brain + "
        f"(1 - f_LV) * CLup_brain * (1 - f_BBB) * V_ES_brain * C_TFV_brain - "
        f"V_BCSFB_brain * kon_FcRn * C_BCSFB_unbound_brain * FcRn_free_BCSFB + "
        f"V_BCSFB_brain * koff_FcRn * C_BCSFB_bound_brain - "
        f"V_BCSFB_brain * kdeg * C_BCSFB_unbound_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/V_BCSFB_brain * ({bcsfb_unbound_eq})")
    bcsfb_unbound_rule.setMath(math_ast)

    # BCSFB Bound equation
    bcsfb_bound_rule = model.createRateRule()
    bcsfb_bound_rule.setVariable("C_BCSFB_bound_brain")
    
    bcsfb_bound_eq = (
        f"-CLup_brain * (1-f_BBB) * V_ES_brain * C_BCSFB_bound_brain + "
        f"V_BCSFB_brain * kon_FcRn * C_BCSFB_unbound_brain * FcRn_free_BCSFB - "
        f"V_BCSFB_brain * koff_FcRn * C_BCSFB_bound_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/V_BCSFB_brain * ({bcsfb_bound_eq})")
    bcsfb_bound_rule.setMath(math_ast)

    # Lateral Ventricle equation
    lv_rule = model.createRateRule()
    lv_rule.setVariable("C_LV_brain")
    
    lv_eq = (
        f"(1-sigma_V_BCSFB) * f_LV * Q_CSF_brain * C_p_brain + "
        f"f_LV * Q_ISF_brain * C_IS_brain - "
        f"(f_LV * Q_CSF_brain + f_LV * Q_ISF_brain) * C_LV_brain - "
        f"f_LV * CLup_brain * (1-f_BBB) * V_ES_brain * C_LV_brain + "
        f"f_LV * CLup_brain * (1-f_BBB) * V_ES_brain * (1-FR) * C_BCSFB_bound_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/V_LV_brain * ({lv_eq})")
    lv_rule.setMath(math_ast)

    # Third/Fourth Ventricle equation
    tfv_rule = model.createRateRule()
    tfv_rule.setVariable("C_TFV_brain")
    
    tfv_eq = (
        f"(1-sigma_V_BCSFB) * (1-f_LV) * Q_CSF_brain * C_p_brain + "
        f"(1-f_LV) * Q_ISF_brain * C_IS_brain - "
        f"(Q_CSF_brain + Q_ISF_brain) * C_TFV_brain - "
        f"(1-f_LV) * CLup_brain * (1-f_BBB) * V_ES_brain * C_TFV_brain + "
        f"(1-f_LV) * CLup_brain * (1-f_BBB) * V_ES_brain * (1-FR) * C_BCSFB_bound_brain + "
        f"(f_LV * Q_CSF_brain + f_LV * Q_ISF_brain) * C_LV_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/V_TFV_brain * ({tfv_eq})")
    tfv_rule.setMath(math_ast)

    # Cisterna Magna equation
    cm_rule = model.createRateRule()
    cm_rule.setVariable("C_CM_brain")
    
    cm_eq = (
        f"(Q_CSF_brain + Q_ISF_brain) * (C_TFV_brain - C_CM_brain)"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/V_CM_brain * ({cm_eq})")
    cm_rule.setMath(math_ast)

    # Subarachnoid Space equation
    sas_rule = model.createRateRule()
    sas_rule.setVariable("C_SAS_brain")
    
    sas_eq = (
        f"(Q_CSF_brain + Q_ISF_brain) * C_CM_brain - "
        f"(1-sigma_L_SAS) * Q_CSF_brain * C_SAS_brain - "
        f"Q_ISF_brain * C_SAS_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/V_SAS_brain * ({sas_eq})")
    sas_rule.setMath(math_ast)

    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    document = create_csf_model(params)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, "csf_sbml.xml")
        print("CSF model saved successfully!") 