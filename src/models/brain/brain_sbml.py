# uses C_BCSFB and C_SAS_brain which come from csf_sbml.py
# uses C_p_lung and C_bc_lung which come from lung_sbml.py
# creates C_p_brain and C_bc_brain which are used in blood_sbml.py
import libsbml
from pathlib import Path

def create_brain_model(params):
    """
    Create Brain compartment SBML model with equations for:
    - Plasma compartment
    - Unbound mAb in BBB endosomal compartment
    - Bound mAb in BBB endosomal compartment  
    - Unbound mAb in brain ISF compartment
    - Blood cells compartment
    
    Args:
        params (dict): Parameters including volumes, flow rates, etc.
    """
   
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Brain_Model")
    model.setTimeUnits("hour")

    # Add compartments
    brain_plasma = model.createCompartment()
    brain_plasma.setId("brain_plasma")
    brain_plasma.setConstant(True)
    brain_plasma.setSize(params["Vp_brain"])
    brain_plasma.setUnits("millilitre")
    
    bbb_unbound = model.createCompartment()
    bbb_unbound.setId("BBB_unbound") 
    bbb_unbound.setConstant(True)
    bbb_unbound.setSize(params["VBBB_brain"])
    bbb_unbound.setUnits("millilitre")
    
    bbb_bound = model.createCompartment()
    bbb_bound.setId("BBB_bound")
    bbb_bound.setConstant(True) 
    bbb_bound.setSize(params["VBBB_brain"])
    bbb_bound.setUnits("millilitre")
    
    brain_isf = model.createCompartment()
    brain_isf.setId("brain_ISF")
    brain_isf.setConstant(True)
    brain_isf.setSize(params["VIS_brain"])
    brain_isf.setUnits("millilitre")
    
    brain_bc = model.createCompartment()
    brain_bc.setId("brain_blood_cells")
    brain_bc.setConstant(True)
    brain_bc.setSize(params["VBC_brain"])
    brain_bc.setUnits("millilitre")
    
    # Add coupling parameters (time-varying concentrations)
    coupling_params = [
        # Lung coupling parameters
        ("C_p_lung", params["C_p_lung_0"]),
        ("C_bc_lung", params["C_bc_lung_0"]),
        
        # CSF coupling parameter
        ("C_SAS_brain", params["C_SAS_brain_0"]),          # From CSF
        ("C_BCSFB_bound_brain", params["C_BCSFB_bound_brain_0"])  # From CSF
    ]

    # Create coupling parameters
    for param_id, value in coupling_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(False)
        param.setUnits("mole_per_litre")

    # Add regular parameters
    required_params = [
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
        "L_brain"
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

    # Add species with ODEs
    species_list = [
        ("C_p_brain", "brain_plasma", params["C_p_brain_0"]),
        ("C_BBB_unbound_brain", "BBB_unbound", params["C_BBB_unbound_brain_0"]),
        ("C_BBB_bound_brain", "BBB_bound", params["C_BBB_bound_brain_0"]),
        ("C_is_brain", "brain_ISF", params["C_is_brain_0"]),
        ("C_bc_brain", "brain_blood_cells", params["C_bc_brain_0"])
    ]

    for species_id, compartment_id, initial_value in species_list:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment(compartment_id)
        species.setInitialConcentration(initial_value)
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setConstant(False)
        species.setUnits("mole_per_litre")

    # Add rate rules
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
    
    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "brain_sbml.xml"
    
    document = create_brain_model(params)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"Brain model saved successfully to {output_path}!") 