import libsbml
from pathlib import Path
import pandas as pd

def create_lung_model(params):
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Lung_Model")
    model.setTimeUnits("hour")
    
    # Add compartments
    vp_lung = model.createCompartment()
    vp_lung.setId("Vp_lung")
    vp_lung.setConstant(True)
    vp_lung.setSize(params["Vp_lung"])
    vp_lung.setUnits("millilitre")
    
    vbc_lung = model.createCompartment()
    vbc_lung.setId("VBC_lung")
    vbc_lung.setConstant(True)
    vbc_lung.setSize(params["VBC_lung"])
    vbc_lung.setUnits("millilitre")
    
    vis_lung = model.createCompartment()
    vis_lung.setId("VIS_lung")
    vis_lung.setConstant(True)
    vis_lung.setSize(params["VIS_lung"])
    vis_lung.setUnits("millilitre")
    
    ves_lung = model.createCompartment()
    ves_lung.setId("VES_lung")
    ves_lung.setConstant(True)
    ves_lung.setSize(params["VES_lung"])
    ves_lung.setUnits("millilitre")

    # Add coupling parameters (time-varying concentrations from blood)
    coupling_params = [
        ("C_p", params["C_p_0"]),  # From blood
        ("C_bc", params["C_bc_0"]) # From blood
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
        "Vp_lung", "VBC_lung", "VIS_lung", "VES_lung",
        
        # Flow rates
        "Q_p_lung", "Q_bc_lung", "L_lung",
        
        # Rate constants
        "kon_FcRn", "koff_FcRn", "kdeg", "CLup_lung",
        
        # Lung-specific parameters
        "FR",
        "sigma_V_lung", "sigma_L_lung",

        # initial concentrations
        "C_p_lung_0", "C_bc_lung_0", "C_is_lung_0", "C_e_unbound_lung_0", "C_e_bound_lung_0", "C_p_0", "C_bc_0", "FcRn_free_lung_0"
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
        param.setUnits("dimensionless")  # Update units as needed

    # Add species with ODEs
    species_list = [
        ("C_p_lung", "Vp_lung", params["C_p_lung_0"]),
        ("C_bc_lung", "VBC_lung", params["C_bc_lung_0"]),
        ("C_is_lung", "VIS_lung", params["C_is_lung_0"]),
        ("C_e_unbound_lung", "VES_lung", params["C_e_unbound_lung_0"]),
        ("C_e_bound_lung", "VES_lung", params["C_e_bound_lung_0"]),
        ("FcRn_free_lung", "VES_lung", params["FcRn_free_lung_0"])
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
    plasma_rule.setVariable("C_p_lung")
    plasma_eq = (
        f"Q_p_lung * C_p - "  # Now uses coupling parameter C_p
        f"(Q_p_lung - L_lung) * C_p_lung - "
        f"(1 - sigma_V_lung) * L_lung * C_p_lung - "
        f"CLup_lung * C_p_lung + "
        f"CLup_lung * FR * C_e_bound_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_lung * ({plasma_eq})")
    plasma_rule.setMath(math_ast)

    # Blood cells equation
    bc_rule = model.createRateRule()
    bc_rule.setVariable("C_bc_lung")
    bc_eq = (
        f"Q_bc_lung * C_bc - "  # lung and liver use C_bc from blood plasma but other tissues use lung C_bc
        f"Q_bc_lung * C_bc_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_lung * ({bc_eq})")
    bc_rule.setMath(math_ast)

    # Endosomal unbound equation
    e_unbound_rule = model.createRateRule()
    e_unbound_rule.setVariable("C_e_unbound_lung")
    e_unbound_eq = (
        f"CLup_lung * (C_p_lung + C_is_lung) - "
        f"VES_lung * kon_FcRn * C_e_unbound_lung * FcRn_free_lung + "
        f"VES_lung * koff_FcRn * C_e_bound_lung - "
        f"kdeg * C_e_unbound_lung * VES_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_lung * ({e_unbound_eq})")
    e_unbound_rule.setMath(math_ast)

    # Endosomal bound equation
    e_bound_rule = model.createRateRule()
    e_bound_rule.setVariable("C_e_bound_lung")
    e_bound_eq = (
        f"VES_lung * kon_FcRn * C_e_unbound_lung * FcRn_free_lung - "
        f"VES_lung * koff_FcRn * C_e_bound_lung - "
        f"CLup_lung * C_e_bound_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_lung * ({e_bound_eq})")
    e_bound_rule.setMath(math_ast)

    # Interstitial space equation
    is_rule = model.createRateRule()
    is_rule.setVariable("C_is_lung")
    is_eq = (
        f"(1 - sigma_V_lung) * L_lung * C_p_lung - "
        f"(1 - sigma_L_lung) * L_lung * C_is_lung + "
        f"CLup_lung * (1 - FR) * C_e_bound_lung - "
        f"CLup_lung * C_is_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_lung * ({is_eq})")
    is_rule.setMath(math_ast)

    # Add FcRn_free equation
    fcrn_free_rule = model.createRateRule()
    fcrn_free_rule.setVariable("FcRn_free_lung")
    fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_lung * VES_lung - "
        f"kon_FcRn * C_e_unbound_lung * FcRn_free_lung * VES_lung + "
        f"CLup_lung * C_e_bound_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_lung * ({fcrn_free_eq})")
    fcrn_free_rule.setMath(math_ast)

    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lung_sbml.xml"
    
    document = create_lung_model(params)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"Lung model saved successfully to {output_path}!")

if __name__ == "__main__":
    # This would only run if lung_sbml.py is run directly
    # Load parameters from Excel
    params = {}
    excel_file = "parameters/lung_params.xlsx"
    
    for sheet in ['Volumes', 'Flows', 'Kinetics', 'Concentrations']:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    main(params) 