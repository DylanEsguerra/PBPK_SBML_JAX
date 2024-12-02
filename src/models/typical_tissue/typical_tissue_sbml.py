import libsbml
from pathlib import Path
import pandas as pd

def create_typical_tissue_model(params, organ_name):
    """Create SBML model for a typical tissue (heart, muscle, etc.)"""
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel(f"{organ_name}_Model")
    model.setTimeUnits("hour")
    
    # Add compartments
    compartments = {
        f"Vp_{organ_name}": params[f"Vp_{organ_name}"],
        f"VBC_{organ_name}": params[f"VBC_{organ_name}"],
        f"VIS_{organ_name}": params[f"VIS_{organ_name}"],
        f"VES_{organ_name}": params[f"VES_{organ_name}"]
    }
    
    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setConstant(True)
        comp.setSize(size)
        comp.setUnits("millilitre")

    # Add parameters
    required_params = [
        f"Q_p_{organ_name}", f"Q_bc_{organ_name}", f"L_{organ_name}",
        "kon_FcRn", "koff_FcRn", "kdeg", f"CLup_{organ_name}",
        "FR", f"sigma_V_{organ_name}", f"sigma_L_{organ_name}"
    ]
    
    for param_id in required_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(params[param_id])
        param.setConstant(True)

    # Add coupling parameters
    coupling_params = [
        ("C_p_lung", params["C_p_lung_0"]),
        ("C_bc_lung", params["C_bc_lung_0"])
    ]
    
    for param_id, value in coupling_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(False)
        param.setUnits("mole_per_litre")

    # Add species
    species_list = [
        (f"C_p_{organ_name}", f"Vp_{organ_name}", params[f"C_p_{organ_name}_0"]),
        (f"C_bc_{organ_name}", f"VBC_{organ_name}", params[f"C_bc_{organ_name}_0"]),
        (f"C_is_{organ_name}", f"VIS_{organ_name}", params[f"C_is_{organ_name}_0"]),
        (f"C_e_unbound_{organ_name}", f"VES_{organ_name}", params[f"C_e_unbound_{organ_name}_0"]),
        (f"C_e_bound_{organ_name}", f"VES_{organ_name}", params[f"C_e_bound_{organ_name}_0"]),
        (f"FcRn_free_{organ_name}", f"VES_{organ_name}", params[f"FcRn_free_{organ_name}_0"])
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
    plasma_rule.setVariable(f"C_p_{organ_name}")
    plasma_eq = (
        f"Q_p_{organ_name} * C_p_lung - "
        f"Q_p_{organ_name} * C_p_{organ_name} - "
        f"(1 - sigma_V_{organ_name}) * L_{organ_name} * C_p_{organ_name} - "
        f"CLup_{organ_name} * C_p_{organ_name} + "
        f"CLup_{organ_name} * FR * C_e_bound_{organ_name}"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_{organ_name} * ({plasma_eq})")
    plasma_rule.setMath(math_ast)

    # Blood cells equation
    bc_rule = model.createRateRule()
    bc_rule.setVariable(f"C_bc_{organ_name}")
    bc_eq = (
        f"Q_bc_{organ_name} * C_bc_lung - "
        f"Q_bc_{organ_name} * C_bc_{organ_name}"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_{organ_name} * ({bc_eq})")
    bc_rule.setMath(math_ast)

    # ISF equation
    is_rule = model.createRateRule()
    is_rule.setVariable(f"C_is_{organ_name}")
    is_eq = (
        f"(1 - sigma_V_{organ_name}) * L_{organ_name} * C_p_{organ_name} - "
        f"(1 - sigma_L_{organ_name}) * L_{organ_name} * C_is_{organ_name} + "
        f"CLup_{organ_name} * (1 - FR) * C_e_bound_{organ_name} - "
        f"CLup_{organ_name} * C_is_{organ_name}"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_{organ_name} * ({is_eq})")
    is_rule.setMath(math_ast)

    # Endosomal unbound equation
    e_unbound_rule = model.createRateRule()
    e_unbound_rule.setVariable(f"C_e_unbound_{organ_name}")
    e_unbound_eq = (
        f"CLup_{organ_name} * (C_p_{organ_name} + C_is_{organ_name}) - "
        f"VES_{organ_name} * kon_FcRn * C_e_unbound_{organ_name} * FcRn_free_{organ_name} + "
        f"VES_{organ_name} * koff_FcRn * C_e_bound_{organ_name} - "
        f"kdeg * C_e_unbound_{organ_name} * VES_{organ_name}"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_{organ_name} * ({e_unbound_eq})")
    e_unbound_rule.setMath(math_ast)

    # Endosomal bound equation
    e_bound_rule = model.createRateRule()
    e_bound_rule.setVariable(f"C_e_bound_{organ_name}")
    e_bound_eq = (
        f"VES_{organ_name} * kon_FcRn * C_e_unbound_{organ_name} * FcRn_free_{organ_name} - "
        f"VES_{organ_name} * koff_FcRn * C_e_bound_{organ_name} - "
        f"CLup_{organ_name} * C_e_bound_{organ_name}"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_{organ_name} * ({e_bound_eq})")
    e_bound_rule.setMath(math_ast)

    # FcRn free equation
    fcrn_free_rule = model.createRateRule()
    fcrn_free_rule.setVariable(f"FcRn_free_{organ_name}")
    fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_{organ_name} * VES_{organ_name} - "
        f"kon_FcRn * C_e_unbound_{organ_name} * FcRn_free_{organ_name} * VES_{organ_name} + "
        f"CLup_{organ_name} * C_e_bound_{organ_name}"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_{organ_name} * ({fcrn_free_eq})")
    fcrn_free_rule.setMath(math_ast)

    return document

def create_typical_tissues_model(params):
    """Create SBML model for all typical tissue organs"""
    typical_tissues = [
        'heart', 'muscle', 'kidney', 'skin', 'fat', 'marrow', 'thymus',
        'SI', 'LI', 'spleen', 'pancreas', 'other'
    ]
    
    # Create a document for each organ
    organ_documents = {}
    for organ in typical_tissues:
        organ_documents[organ] = create_typical_tissue_model(params, organ)
    
    return organ_documents

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    output_dir = Path("generated/sbml/typical_tissue")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    organ_documents = create_typical_tissues_model(params)
    
    for organ, document in organ_documents.items():
        output_path = output_dir / f"{organ}_sbml.xml"
        if document.getNumErrors() != 0:
            print(f"\nValidation errors for {organ}:")
            document.printErrors()
        else:
            save_model(document, str(output_path))
            print(f"{organ.capitalize()} model saved successfully to {output_path}!")

if __name__ == "__main__":
    # Load parameters from Excel when run directly
    params = {}
    excel_file = "parameters/typical_tissue_params.xlsx"
    
    for sheet in ['Volumes', 'Flows', 'Kinetics', 'Concentrations']:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    main(params) 