import libsbml
from pathlib import Path
import pandas as pd

def create_liver_model(params):
    """
    Create Liver compartment SBML model
    
    Args:
        params (dict): Parameters including volumes, flow rates, etc.
    """
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Liver_Model")
    model.setTimeUnits("hour")
    
    # Add regular parameters
    required_params = [
        # Volumes
        "Vp_liver", "VBC_liver", "VIS_liver", "VES_liver",
        
        # Flow rates
        "Q_p_liver", "Q_bc_liver",
        "Q_p_spleen", "Q_bc_spleen", "Q_p_pancreas", "Q_bc_pancreas",
        "Q_p_SI", "Q_bc_SI", "Q_p_LI", "Q_bc_LI",
        
        # Lymph flows
        "L_liver", "L_spleen", "L_pancreas", "L_SI", "L_LI",
        
        # Kinetic parameters
        "kon_FcRn", "koff_FcRn", "kdeg", "CLup_liver",
        "sigma_V_liver", "sigma_L_liver", "FR",
        
        # Initial concentrations
        "C_p_liver_0", "C_bc_liver_0", "C_is_liver_0",
        "C_e_unbound_liver_0", "C_e_bound_liver_0", "FcRn_free_liver_0",
        "C_p_lung_0", "C_bc_lung_0",
        "C_p_spleen_0", "C_bc_spleen_0",
        "C_p_pancreas_0", "C_bc_pancreas_0", 
        "C_p_SI_0", "C_bc_SI_0",
        "C_p_LI_0", "C_bc_LI_0"
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

    # Add compartments
    vp_liver = model.createCompartment()
    vp_liver.setId("Vp_liver")
    vp_liver.setConstant(True)
    vp_liver.setSize(params["Vp_liver"])
    vp_liver.setUnits("millilitre")
    
    vbc_liver = model.createCompartment()
    vbc_liver.setId("VBC_liver")
    vbc_liver.setConstant(True)
    vbc_liver.setSize(params["VBC_liver"])
    vbc_liver.setUnits("millilitre")
    
    vis_liver = model.createCompartment()
    vis_liver.setId("VIS_liver")
    vis_liver.setConstant(True)
    vis_liver.setSize(params["VIS_liver"])
    vis_liver.setUnits("millilitre")
    
    ves_liver = model.createCompartment()
    ves_liver.setId("VES_liver")
    ves_liver.setConstant(True)
    ves_liver.setSize(params["VES_liver"])
    ves_liver.setUnits("millilitre")

    # Add coupling parameters (time-varying concentrations)
    coupling_params = [
        # current 
        ("C_p_lung", params["C_p_lung_0"]),      # From lung
        ("C_bc_lung", params["C_bc_lung_0"]),    # From lung
        
        # future

        ("C_p_spleen", params["C_p_spleen_0"]),  # From spleen
        ("C_bc_spleen", params["C_bc_spleen_0"]),# From spleen
        ("C_p_pancreas", params["C_p_pancreas_0"]), # From pancreas
        ("C_bc_pancreas", params["C_bc_pancreas_0"]), # From pancreas
        ("C_p_SI", params["C_p_SI_0"]),         # From small intestine
        ("C_bc_SI", params["C_bc_SI_0"]),       # From small intestine
        ("C_p_LI", params["C_p_LI_0"]),         # From large intestine
        ("C_bc_LI", params["C_bc_LI_0"])        # From large intestine
    ]

    for param_id, value in coupling_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(False)
        param.setUnits("mole_per_litre")

    # Add species with ODEs
    species_list = [
        ("C_p_liver", "Vp_liver", params["C_p_liver_0"]),
        ("C_bc_liver", "VBC_liver", params["C_bc_liver_0"]),
        ("C_is_liver", "VIS_liver", params["C_is_liver_0"]),
        ("C_e_unbound_liver", "VES_liver", params["C_e_unbound_liver_0"]),
        ("C_e_bound_liver", "VES_liver", params["C_e_bound_liver_0"]),
        ("FcRn_free_liver", "VES_liver", params["FcRn_free_liver_0"])
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

    # Add plasma rate rule
    plasma_rule = model.createRateRule()
    plasma_rule.setVariable("C_p_liver")
    plasma_eq = (
        f"Q_p_liver * C_p_lung + "
        f"(Q_p_spleen - L_spleen) * C_p_spleen + "
        f"(Q_p_pancreas - L_pancreas) * C_p_pancreas + "
        f"(Q_p_SI - L_SI) * C_p_SI + "
        f"(Q_p_LI - L_LI) * C_p_LI - "
        f"((Q_p_liver - L_liver) + "
        f"(Q_p_spleen - L_spleen) + "
        f"(Q_p_pancreas - L_pancreas) + "
        f"(Q_p_SI - L_SI) + "
        f"(Q_p_LI - L_LI)) * C_p_liver - "
        f"(1 - sigma_V_liver) * L_liver * C_p_liver - "
        f"CLup_liver * C_p_liver + "
        f"CLup_liver * FR * C_e_bound_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_liver * ({plasma_eq})")
    plasma_rule.setMath(math_ast)

    # Add blood cells rate rule
    bc_rule = model.createRateRule()
    bc_rule.setVariable("C_bc_liver")
    bc_eq = (
        f"Q_bc_liver * C_bc_lung + "
        f"Q_bc_spleen * C_bc_spleen + "
        f"Q_bc_pancreas * C_bc_pancreas + "
        f"Q_bc_SI * C_bc_SI + "
        f"Q_bc_LI * C_bc_LI - "
        f"(Q_bc_liver + Q_bc_spleen + Q_bc_pancreas + Q_bc_SI + Q_bc_LI) * C_bc_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_liver * ({bc_eq})")
    bc_rule.setMath(math_ast)

    # Add ISF rate rule
    is_rule = model.createRateRule()
    is_rule.setVariable("C_is_liver")
    is_eq = (
        f"(1 - sigma_V_liver) * L_liver * C_p_liver - "
        f"(1 - sigma_L_liver) * L_liver * C_is_liver + "
        f"CLup_liver * (1 - FR) * C_e_bound_liver - "
        f"CLup_liver * C_is_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_liver * ({is_eq})")
    is_rule.setMath(math_ast)

    # Add endosomal unbound rate rule
    e_unbound_rule = model.createRateRule()
    e_unbound_rule.setVariable("C_e_unbound_liver")
    e_unbound_eq = (
        f"CLup_liver * (C_p_liver + C_is_liver) - "
        f"VES_liver * kon_FcRn * C_e_unbound_liver * FcRn_free_liver + "
        f"VES_liver * koff_FcRn * C_e_bound_liver - "
        f"kdeg * C_e_unbound_liver * VES_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_liver * ({e_unbound_eq})")
    e_unbound_rule.setMath(math_ast)

    # Add endosomal bound rate rule
    e_bound_rule = model.createRateRule()
    e_bound_rule.setVariable("C_e_bound_liver")
    e_bound_eq = (
        f"VES_liver * kon_FcRn * C_e_unbound_liver * FcRn_free_liver - "
        f"VES_liver * koff_FcRn * C_e_bound_liver - "
        f"CLup_liver * C_e_bound_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_liver * ({e_bound_eq})")
    e_bound_rule.setMath(math_ast)

    # Add FcRn free rate rule
    fcrn_free_rule = model.createRateRule()
    fcrn_free_rule.setVariable("FcRn_free_liver")
    fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_liver * VES_liver - "
        f"kon_FcRn * C_e_unbound_liver * FcRn_free_liver * VES_liver + "
        f"CLup_liver * C_e_bound_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_liver * ({fcrn_free_eq})")
    fcrn_free_rule.setMath(math_ast)

    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "liver_sbml.xml"
    
    document = create_liver_model(params)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"Liver model saved successfully to {output_path}!")

if __name__ == "__main__":
    # Load parameters from Excel when run directly
    params = {}
    excel_file = "parameters/liver_params.xlsx"
    
    for sheet in ['Volumes', 'Flows', 'Kinetics', 'Concentrations']:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    main(params) 