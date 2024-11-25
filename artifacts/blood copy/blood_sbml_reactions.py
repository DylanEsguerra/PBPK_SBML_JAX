import libsbml

def create_blood_model(params):
    """
    Create Blood compartment SBML model using reactions for flows between compartments
    
    Args:
        params (dict): Model parameters including volumes, flow rates, and reflection coefficients
    """
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Blood_Model")
    model.setTimeUnits("hour")
    
    # Create compartments
    compartments = {
        "plasma": params["Vp"],
        "blood_cells": params["VBC"],
        "lymph_node": params["V_LN"]
    }
    
    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setConstant(True)
        comp.setSize(size)
        comp.setUnits("litre")
    
    # Create species
    species = {
        "C_p": "plasma",
        "C_BC": "blood_cells",
        "C_LN": "lymph_node"
    }
    
    for species_id, comp_id in species.items():
        spec = model.createSpecies()
        spec.setId(species_id)
        spec.setCompartment(comp_id)
        spec.setInitialConcentration(0.0)
        spec.setBoundaryCondition(False)
        spec.setHasOnlySubstanceUnits(False)
        spec.setConstant(False)
        spec.setUnits("mole_per_litre")
    
    # Add parameters
    for param_id, value in params.items():
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
    
    # Create plasma flow reactions for each organ
    organs = ["lung", "heart", "kidney", "brain", "muscle", "marrow", 
              "thymus", "skin", "fat", "liver", "other"]
    
    for organ in organs:
        # Plasma flow from organ to central
        organ_to_plasma = model.createReaction()
        organ_to_plasma.setId(f"{organ}_to_plasma")
        organ_to_plasma.setReversible(False)
        
        # Reactant (organ)
        reactant = organ_to_plasma.createReactant()
        reactant.setSpecies(f"C_p_{organ}")  # Organ plasma concentration
        reactant.setStoichiometry(1.0)
        
        # Product (plasma)
        product = organ_to_plasma.createProduct()
        product.setSpecies("C_p")
        product.setStoichiometry(1.0)
        
        # Kinetic law: Q_p_organ * C_p_organ
        if organ == "kidney":
            flow_param = "Q_plasma_kidney"
        else:
            flow_param = f"Q_p_{organ}"
            
        math_ast = libsbml.parseL3Formula(f"{flow_param} * C_p_{organ}")
        kinetic_law = organ_to_plasma.createKineticLaw()
        kinetic_law.setMath(math_ast)
        
        # Plasma flow from central to organ
        plasma_to_organ = model.createReaction()
        plasma_to_organ.setId(f"plasma_to_{organ}")
        plasma_to_organ.setReversible(False)
        
        # Reactant (plasma)
        reactant = plasma_to_organ.createReactant()
        reactant.setSpecies("C_p")
        reactant.setStoichiometry(1.0)
        
        # Product (organ)
        product = plasma_to_organ.createProduct()
        product.setSpecies(f"C_p_{organ}")
        product.setStoichiometry(1.0)
        
        # Kinetic law: Q_p_organ * C_p
        math_ast = libsbml.parseL3Formula(f"{flow_param} * C_p")
        kinetic_law = plasma_to_organ.createKineticLaw()
        kinetic_law.setMath(math_ast)
    
    # Create blood cell flow reactions
    for organ in organs:
        # Blood cell flow from organ to central
        organ_to_bc = model.createReaction()
        organ_to_bc.setId(f"{organ}_to_bc")
        organ_to_bc.setReversible(False)
        
        # Reactant (organ)
        reactant = organ_to_bc.createReactant()
        reactant.setSpecies(f"C_BC_{organ}")
        reactant.setStoichiometry(1.0)
        
        # Product (blood cells)
        product = organ_to_bc.createProduct()
        product.setSpecies("C_BC")
        product.setStoichiometry(1.0)
        
        # Kinetic law: Q_BC_organ * C_BC_organ
        math_ast = libsbml.parseL3Formula(f"Q_BC_{organ} * C_BC_{organ}")
        kinetic_law = organ_to_bc.createKineticLaw()
        kinetic_law.setMath(math_ast)
        
        # Blood cell flow from central to organ
        bc_to_organ = model.createReaction()
        bc_to_organ.setId(f"bc_to_{organ}")
        bc_to_organ.setReversible(False)
        
        # Reactant (blood cells)
        reactant = bc_to_organ.createReactant()
        reactant.setSpecies("C_BC")
        reactant.setStoichiometry(1.0)
        
        # Product (organ)
        product = bc_to_organ.createProduct()
        product.setSpecies(f"C_BC_{organ}")
        product.setStoichiometry(1.0)
        
        # Kinetic law: Q_BC_organ * C_BC
        math_ast = libsbml.parseL3Formula(f"Q_BC_{organ} * C_BC")
        kinetic_law = bc_to_organ.createKineticLaw()
        kinetic_law.setMath(math_ast)
    
    # Create lymph flow reactions
    organs_lymph = organs + ["SI", "LI", "spleen", "pancreas"]
    
    for organ in organs_lymph:
        # Lymph flow from organ to lymph node
        lymph_flow = model.createReaction()
        lymph_flow.setId(f"lymph_flow_{organ}")
        lymph_flow.setReversible(False)
        
        # Reactant (organ interstitial space)
        reactant = lymph_flow.createReactant()
        reactant.setSpecies(f"C_IS_{organ}")
        reactant.setStoichiometry(1.0)
        
        # Product (lymph node)
        product = lymph_flow.createProduct()
        product.setSpecies("C_LN")
        product.setStoichiometry(1.0)
        
        # Kinetic law: (1 - sigma_L_organ) * L_organ * C_IS_organ
        math_ast = libsbml.parseL3Formula(
            f"(1 - sigma_L_{organ}) * L_{organ} * C_IS_{organ}"
        )
        kinetic_law = lymph_flow.createKineticLaw()
        kinetic_law.setMath(math_ast)
    
    # Add special brain lymph flows (CSF and ISF)
    brain_csf_flow = model.createReaction()
    brain_csf_flow.setId("brain_csf_flow")
    brain_csf_flow.setReversible(False)
    
    reactant = brain_csf_flow.createReactant()
    reactant.setSpecies("C_SAS_brain")
    reactant.setStoichiometry(1.0)
    
    product = brain_csf_flow.createProduct()
    product.setSpecies("C_LN")
    product.setStoichiometry(1.0)
    
    math_ast = libsbml.parseL3Formula(
        "(1 - sigma_L_SAS) * Q_CSF_brain * C_SAS_brain"
    )
    kinetic_law = brain_csf_flow.createKineticLaw()
    kinetic_law.setMath(math_ast)
    
    # Lymph flow from lymph node to plasma
    ln_to_plasma = model.createReaction()
    ln_to_plasma.setId("ln_to_plasma")
    ln_to_plasma.setReversible(False)
    
    reactant = ln_to_plasma.createReactant()
    reactant.setSpecies("C_LN")
    reactant.setStoichiometry(1.0)
    
    product = ln_to_plasma.createProduct()
    product.setSpecies("C_p")
    product.setStoichiometry(1.0)
    
    math_ast = libsbml.parseL3Formula("L_LN * C_LN")
    kinetic_law = ln_to_plasma.createKineticLaw()
    kinetic_law.setMath(math_ast)
    
    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    document = create_blood_model(params)
    
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, "blood_sbml_reactions.xml")
        print("Blood model saved successfully!")

if __name__ == "__main__":
    # Example parameters (same as before)
    params = {
        "Vp": 3.0,
        "VBC": 2.0,
        "V_LN": 0.1,
        # ... other parameters
    }
    main(params) 