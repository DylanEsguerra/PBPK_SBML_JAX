import libsbml
from pathlib import Path

def create_combined_model(params):
    """
    Create a combined Brain-CSF SBML model that includes:
    
    Brain compartments:
    - Plasma compartment
    - Unbound mAb in BBB endosomal compartment
    - Bound mAb in BBB endosomal compartment  
    - Unbound mAb in brain ISF compartment
    - Blood cells compartment
    
    CSF compartments:
    - Unbound mAb in BCSFB endosomal compartment
    - Bound mAb in BCSFB endosomal compartment
    - Lateral Ventricle
    - Third/Fourth Ventricle
    - Cisterna Magna
    - Subarachnoid Space
    """
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Combined_Brain_CSF_Model")
    model.setTimeUnits("hour")

    # Add Brain compartments
    brain_compartments = [
        ("brain_plasma", params["Vp_brain"]),
        ("BBB_unbound", params["VBBB_brain"]),
        ("BBB_bound", params["VBBB_brain"]),
        ("brain_ISF", params["VIS_brain"]),
        ("brain_blood_cells", params["VBC_brain"])
    ]
    
    # Add CSF compartments
    csf_compartments = [
        ("BCSFB_unbound", params["V_BCSFB_brain"]),
        ("BCSFB_bound", params["V_BCSFB_brain"]),
        ("LV", params["V_LV_brain"]),
        ("TFV", params["V_TFV_brain"]),
        ("CM", params["V_CM_brain"]),
        ("SAS", params["V_SAS_brain"])
    ]
    
    # Create all compartments
    for comp_id, size in brain_compartments + csf_compartments:
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setConstant(True)
        comp.setSize(size)
        comp.setUnits("millilitre")

    # Add Brain species
    brain_species = [
        ("C_p_brain", "brain_plasma", params["C_p_brain_0"]),
        ("C_BBB_unbound_brain", "BBB_unbound", params["C_BBB_unbound_brain_0"]),
        ("C_BBB_bound_brain", "BBB_bound", params["C_BBB_bound_brain_0"]),
        ("C_is_brain", "brain_ISF", params["C_is_brain_0"]),
        ("C_bc_brain", "brain_blood_cells", params["C_bc_brain_0"])
    ]
    
    # Add CSF species
    csf_species = [
        ("C_BCSFB_unbound_brain", "BCSFB_unbound", params["C_BCSFB_unbound_brain_0"]),
        ("C_BCSFB_bound_brain", "BCSFB_bound", params["C_BCSFB_bound_brain_0"]),
        ("C_LV_brain", "LV", params["C_LV_brain_0"]),
        ("C_TFV_brain", "TFV", params["C_TFV_brain_0"]),
        ("C_CM_brain", "CM", params["C_CM_brain_0"]),
        ("C_SAS_brain", "SAS", params["C_SAS_brain_0"])
    ]
    
    # Create all species
    for species_id, comp_id, init_conc in brain_species + csf_species:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment(comp_id)
        species.setInitialConcentration(init_conc)
        species.setBoundaryCondition(False)
        species.setHasOnlySubstanceUnits(False)
        species.setConstant(False)
        species.setUnits("mole_per_litre")

    # Add parameters
    required_params = [
        # Brain parameters
        # Volumes
        "Vp_brain", "VBBB_brain", "VIS_brain", "VBC_brain", "V_ES_brain",
        
        # Flow rates
        "Q_p_brain", "Q_bc_brain", "Q_ISF_brain", "Q_CSF_brain",
        
        # Rate constants
        "kon_FcRn", "koff_FcRn", "kdeg", "CLup_brain",
        
        # Brain-specific parameters
        "f_BBB", "FR", "FcRn_free_BBB",
        "sigma_V_BBB", "sigma_V_BCSFB",
        "sigma_L_brain_ISF",
        
        # Lymph flows
        "L_brain",
        
        # CSF parameters
        # Volumes
        "V_BCSFB_brain", "V_LV_brain", "V_TFV_brain", 
        "V_CM_brain", "V_SAS_brain",
        
        # Flow rates
        "Q_CSF_brain",
        
        # Fractions and coefficients
        "f_BCSFB", 
        "sigma_L_SAS",
        
        # Kinetic parameters
        "FcRn_free_BCSFB", "f_LV",
        
        # Initial values for coupling parameters
        #"C_BCSFB_unbound_brain_0", "C_BCSFB_bound_brain_0",
        "C_LV_brain_0", "C_TFV_brain_0", "C_CM_brain_0", "C_SAS_brain_0",
        "C_p_brain_0", "C_is_brain_0", "C_p_lung_0", "C_bc_lung_0"
    ]

    # Add parameters
    for param_id in required_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(params[param_id])
        param.setConstant(True)

     # Add coupling parameters This will eventually be removed when other tissues are added
    coupling_params = [
        # Lung coupling parameters
        ("C_p_lung", params["C_p_lung_0"]),
        ("C_bc_lung", params["C_bc_lung_0"]),
        
    ]

    # Create coupling parameters
    for param_id, value in coupling_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        param.setUnits("mole_per_litre")

    # Add Brain rate rules
    # Plasma equation
    plasma_rule = model.createRateRule()
    plasma_rule.setVariable("C_p_brain")
    
    plasma_eq = (
        f"(Q_p_brain * C_p_lung) - "
        f"(Q_p_brain - L_brain) * C_p_brain - "
        f"(1 - sigma_V_BBB) * Q_ISF_brain * C_p_brain - "
        f"(1 - sigma_V_BCSFB) * Q_CSF_brain * C_p_brain - "
        f"CLup_brain * V_ES_brain * C_p_brain + "
        f"CLup_brain * f_BBB * V_ES_brain * FR * C_BBB_bound_brain + "
        f"CLup_brain * (1-f_BBB) * V_ES_brain * FR * C_BCSFB_bound_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/Vp_brain * ({plasma_eq})")
    plasma_rule.setMath(math_ast)

    # BBB Unbound equation
    bbb_unbound_rule = model.createRateRule()
    bbb_unbound_rule.setVariable("C_BBB_unbound_brain")
    
    bbb_unbound_eq = (
        f"CLup_brain * f_BBB * V_ES_brain * (C_p_brain + C_is_brain) - "
        f"VBBB_brain * kon_FcRn * C_BBB_unbound_brain * FcRn_free_BBB + "
        f"VBBB_brain * koff_FcRn * C_BBB_bound_brain - "
        f"VBBB_brain * kdeg * C_BBB_unbound_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/VBBB_brain * ({bbb_unbound_eq})")
    bbb_unbound_rule.setMath(math_ast)
    
    # BBB Bound equation
    bbb_bound_rule = model.createRateRule()
    bbb_bound_rule.setVariable("C_BBB_bound_brain")
    
    bbb_bound_eq = (
        f"-CLup_brain * f_BBB * V_ES_brain * C_BBB_bound_brain + "
        f"VBBB_brain * kon_FcRn * C_BBB_unbound_brain * FcRn_free_BBB - "
        f"VBBB_brain * koff_FcRn * C_BBB_bound_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/VBBB_brain * ({bbb_bound_eq})")
    bbb_bound_rule.setMath(math_ast)
    
    # ISF equation 
    isf_rule = model.createRateRule()
    isf_rule.setVariable("C_is_brain")
    
    isf_eq = (
        f"(1 - sigma_V_BBB) * Q_ISF_brain * C_p_brain - "
        f"(1 - sigma_L_brain_ISF) * Q_ISF_brain * C_is_brain - "
        f"Q_ISF_brain * C_is_brain + Q_ISF_brain * C_SAS_brain + "
        f"CLup_brain * f_BBB * V_ES_brain * (1 - FR) * C_BBB_bound_brain - "
        f"CLup_brain * f_BBB * V_ES_brain * C_is_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/VIS_brain * ({isf_eq})")
    isf_rule.setMath(math_ast)
    
    # Blood cells equation
    bc_rule = model.createRateRule()
    bc_rule.setVariable("C_bc_brain")
    
    bc_eq = (
        f"Q_bc_brain * C_bc_lung - Q_bc_brain * C_bc_brain"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/VBC_brain * ({bc_eq})")
    bc_rule.setMath(math_ast)
    

    # Add CSF rate rules
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
        f"f_LV * Q_ISF_brain * C_is_brain - "
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
        f"(1-f_LV) * Q_ISF_brain * C_is_brain - "
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
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_brain_csf.xml"
    
    document = create_combined_model(params)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"Combined Brain-CSF model saved successfully to {output_path}!")

if __name__ == "__main__":
    # Load parameters and run main
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')
    params_dict = dict(zip(params_df['name'], params_df['value']))
    main(params_dict) 