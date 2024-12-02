# uses C_p_brain and C_bc_brain which come from brain_sbml.py
# uses C_SAS_brain which comes from csf_sbml.py
# uses C_is_brain which comes from brain_sbml.py
# uses C_is_lung which come from lung_sbml.py

# creates C_p, C_bc and C_ln 
import libsbml
from pathlib import Path

def create_blood_model(params):
    """
    Create Blood compartment SBML model with plasma equation
    
    Args:
        params (dict): Parameters including:
            - Vp: Plasma volume
            - Q_p_{organ}: Plasma flow rates for each organ
            - L_{organ}: Lymph flow rates for each organ
    """
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Blood_Model")
    model.setTimeUnits("hour")
    
    # Add core compartments
    plasma = model.createCompartment()
    plasma.setId("plasma")
    plasma.setConstant(True)
    plasma.setSize(params["Vp"])
    plasma.setUnits("millilitre")
    
    blood_cells = model.createCompartment()
    blood_cells.setId("blood_cells")
    blood_cells.setConstant(True)
    blood_cells.setSize(params["Vbc"])
    blood_cells.setUnits("millilitre")
    
    lymph_node = model.createCompartment()
    lymph_node.setId("lymph_node")
    lymph_node.setConstant(True)
    lymph_node.setSize(params["Vlymphnode"])
    lymph_node.setUnits("millilitre")
    
    # Add main species (only those with ODEs)
    c_plasma = model.createSpecies()
    c_plasma.setId("C_p")
    c_plasma.setCompartment("plasma")
    c_plasma.setInitialConcentration(params["C_p_0"])
    c_plasma.setBoundaryCondition(False)
    c_plasma.setHasOnlySubstanceUnits(False)
    c_plasma.setConstant(False)
    c_plasma.setUnits("mole_per_litre")
    
    c_bc = model.createSpecies()
    c_bc.setId("C_bc")
    c_bc.setCompartment("blood_cells")
    c_bc.setInitialConcentration(params["C_bc_0"])
    c_bc.setBoundaryCondition(False)
    c_bc.setHasOnlySubstanceUnits(False)
    c_bc.setConstant(False)
    c_bc.setUnits("mole_per_litre")
    
    c_ln = model.createSpecies()
    c_ln.setId("C_ln")
    c_ln.setCompartment("lymph_node")
    c_ln.setInitialConcentration(params["C_ln_0"])
    c_ln.setBoundaryCondition(False)
    c_ln.setHasOnlySubstanceUnits(False)
    c_ln.setConstant(False)
    c_ln.setUnits("mole_per_litre")

    # Add coupling parameters (time-varying concentrations)
    coupling_params = [
        # Currently Implemented Tissues

        # Lung coupling parameters
        ("C_is_lung", params["C_is_lung_0"]),
        
        # CSF coupling
        ("C_SAS_brain", params["C_SAS_brain_0"]),

        # Liver coupling 
        ("C_is_liver", params["C_is_liver_0"]),
        ("C_p_liver", params["C_p_liver_0"]),
        ("C_bc_liver", params["C_bc_liver_0"]),


        # Tissues need to be added 

        
        # Other organ plasma concentrations
        ("C_p_heart", params["C_p_heart_0"]),
        ("C_p_kidney", params["C_p_kidney_0"]),
        ("C_p_brain", params["C_p_brain_0"]),
        ("C_p_muscle", params["C_p_muscle_0"]),
        ("C_p_marrow", params["C_p_marrow_0"]),
        ("C_p_thymus", params["C_p_thymus_0"]),
        ("C_p_skin", params["C_p_skin_0"]),
        ("C_p_fat", params["C_p_fat_0"]),
        ("C_p_other", params["C_p_other_0"]),
        
        # Organ blood cell concentrations
        ("C_bc_heart", params["C_bc_heart_0"]),
        ("C_bc_kidney", params["C_bc_kidney_0"]),
        ("C_bc_brain", params["C_bc_brain_0"]),
        ("C_bc_muscle", params["C_bc_muscle_0"]),
        ("C_bc_marrow", params["C_bc_marrow_0"]),
        ("C_bc_thymus", params["C_bc_thymus_0"]),
        ("C_bc_skin", params["C_bc_skin_0"]),
        ("C_bc_fat", params["C_bc_fat_0"]),
        ("C_bc_other", params["C_bc_other_0"]),
        
        # ISF concentrations
        ("C_is_heart", params["C_is_heart_0"]),
        ("C_is_kidney", params["C_is_kidney_0"]),
        ("C_is_brain", params["C_is_brain_0"]),
        ("C_is_muscle", params["C_is_muscle_0"]),
        ("C_is_marrow", params["C_is_marrow_0"]),
        ("C_is_thymus", params["C_is_thymus_0"]),
        ("C_is_skin", params["C_is_skin_0"]),
        ("C_is_fat", params["C_is_fat_0"]),
        ("C_is_SI", params["C_is_SI_0"]),
        ("C_is_LI", params["C_is_LI_0"]),
        ("C_is_spleen", params["C_is_spleen_0"]),
        ("C_is_pancreas", params["C_is_pancreas_0"]),
        ("C_is_other", params["C_is_other_0"]),

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
        "Vp", "Vbc", "Vlymphnode",
        
        # Flow rates
        "Q_p_heart", "Q_p_lung", "Q_p_muscle", "Q_p_skin", "Q_p_fat",
        "Q_p_marrow", "Q_p_kidney", "Q_p_liver", "Q_p_SI", "Q_p_LI",
        "Q_p_pancreas", "Q_p_thymus", "Q_p_spleen", "Q_p_other", "Q_p_brain", "Q_CSF_brain", "Q_ECF_brain",
        
        "Q_bc_heart", "Q_bc_lung", "Q_bc_muscle", "Q_bc_skin", "Q_bc_fat",
        "Q_bc_marrow", "Q_bc_kidney", "Q_bc_liver", "Q_bc_SI", "Q_bc_LI",
        "Q_bc_pancreas", "Q_bc_thymus", "Q_bc_spleen", "Q_bc_other", "Q_bc_brain",
        
        # Lymph flows
        "L_lung", "L_heart", "L_kidney", "L_brain", "L_muscle",
        "L_marrow", "L_thymus", "L_skin", "L_fat", "L_SI",
        "L_LI", "L_spleen", "L_pancreas", "L_liver", "L_other",
        "L_LN",
        
        # Reflection coefficients
        "sigma_L_lung", "sigma_L_heart", "sigma_L_kidney", "sigma_L_brain_ISF",
        "sigma_L_muscle", "sigma_L_marrow", "sigma_L_thymus", "sigma_L_skin",
        "sigma_L_fat", "sigma_L_SI", "sigma_L_LI", "sigma_L_spleen",
        "sigma_L_pancreas", "sigma_L_liver", "sigma_L_other", "sigma_L_SAS",
        
        # Initial conditions
        "C_p_0", "C_bc_0", "C_ln_0"
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

    # Add plasma rate rule
    plasma_rule = model.createRateRule()
    plasma_rule.setVariable("C_p")
    
    # Updated plasma equation to split liver system
    plasma_eq = (
        f"-(Q_p_lung + L_lung) * C_p + "
        f"(Q_p_heart - L_heart) * C_p_heart + "
        f"(Q_p_kidney - L_kidney) * C_p_kidney + "
        f"(Q_p_brain - L_brain) * C_p_brain + "
        f"(Q_p_muscle - L_muscle) * C_p_muscle + "
        f"(Q_p_marrow - L_marrow) * C_p_marrow + "
        f"(Q_p_thymus - L_thymus) * C_p_thymus + "
        f"(Q_p_skin - L_skin) * C_p_skin + "
        f"(Q_p_fat - L_fat) * C_p_fat + "
        f"((Q_p_SI - L_SI) + (Q_p_LI - L_LI) + "
        f"(Q_p_spleen - L_spleen) + "
        f"(Q_p_pancreas - L_pancreas) + "
        f"(Q_p_liver - L_liver)) * C_p_liver + "
        f"(Q_p_other - L_other) * C_p_other + "
        f"L_LN * C_ln"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/Vp * ({plasma_eq})")
    plasma_rule.setMath(math_ast)
    
    # Add blood cells rate rule
    bc_rule = model.createRateRule()
    bc_rule.setVariable("C_bc")
    
    # Updated blood cells equation
    bc_eq = (
        f"-Q_bc_lung * C_bc + "
        f"Q_bc_heart * C_bc_heart + "
        f"Q_bc_kidney * C_bc_kidney + "
        f"Q_bc_brain * C_bc_brain + "
        f"Q_bc_muscle * C_bc_muscle + "
        f"Q_bc_marrow * C_bc_marrow + "
        f"Q_bc_thymus * C_bc_thymus + "
        f"Q_bc_skin * C_bc_skin + "
        f"Q_bc_fat * C_bc_fat + "
        f"(Q_bc_SI + Q_bc_LI + Q_bc_spleen + "
        f"Q_bc_pancreas + Q_bc_liver) * C_bc_liver + "
        f"Q_bc_other * C_bc_other"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/Vbc * ({bc_eq})")
    bc_rule.setMath(math_ast)
    
    # Add lymph node rate rule
    ln_rule = model.createRateRule()
    ln_rule.setVariable("C_ln")
    
    # Lymph node equation
    ln_eq = (
        f"(1-sigma_L_lung) * L_lung * C_is_lung + "
        f"(1-sigma_L_heart) * L_heart * C_is_heart + "
        f"(1-sigma_L_kidney) * L_kidney * C_is_kidney + "
        f"(1-sigma_L_SAS) * Q_CSF_brain * C_SAS_brain + "
        f"(1-sigma_L_brain_ISF) * Q_ECF_brain * C_is_brain + "
        f"(1-sigma_L_muscle) * L_muscle * C_is_muscle + "
        f"(1-sigma_L_marrow) * L_marrow * C_is_marrow + "
        f"(1-sigma_L_thymus) * L_thymus * C_is_thymus + "
        f"(1-sigma_L_skin) * L_skin * C_is_skin + "
        f"(1-sigma_L_fat) * L_fat * C_is_fat + "
        f"(1-sigma_L_SI) * L_SI * C_is_SI + "
        f"(1-sigma_L_LI) * L_LI * C_is_LI + "
        f"(1-sigma_L_spleen) * L_spleen * C_is_spleen + "
        f"(1-sigma_L_pancreas) * L_pancreas * C_is_pancreas + "
        f"(1-sigma_L_liver) * L_liver * C_is_liver + "
        f"(1-sigma_L_other) * L_other * C_is_other - "
        f"L_LN * C_ln"
    )
    
    math_ast = libsbml.parseL3Formula(f"1/Vlymphnode * ({ln_eq})")
    ln_rule.setMath(math_ast)
    
    
    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    # Create output directory if it doesn't exist
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to the correct location
    output_path = output_dir / "blood_sbml.xml"
    
    document = create_blood_model(params)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"Blood model saved successfully to {output_path}!") 