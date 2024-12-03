import libsbml



def create_pk_model(params, sc_doses, iv_doses):
    """
    Create a PK SBML model with specified parameters and dosing schedules
    
    Args:
        params (dict): PK parameters (F, D1, KA, CL, Vc, Q, Vp)
        sc_doses (list): List of tuples (time, amount) for subcutaneous doses
        iv_doses (list): List of tuples (time, amount) for IV doses
    """
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("PK_Model")
    model.setTimeUnits("day")

    # Add parameters to model
    for param_id, value in params.items():
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
    
    # Add compartments with proper volumes
    absorption = model.createCompartment()
    absorption.setId("absorption_site")
    absorption.setConstant(True)
    absorption.setSize(1.0)
    absorption.setUnits("litre")
    
    central = model.createCompartment()
    central.setId("central")
    central.setConstant(True)
    central.setSize(params["Vc"])
    central.setUnits("litre")
    
    peripheral = model.createCompartment()
    peripheral.setId("peripheral")
    peripheral.setConstant(True)
    peripheral.setSize(params["Vp"])
    peripheral.setUnits("litre")
    
    # Add species with proper units
    a = model.createSpecies()
    a.setId("A")
    a.setCompartment("absorption_site")
    a.setInitialAmount(0.0)
    a.setBoundaryCondition(False)
    a.setHasOnlySubstanceUnits(True)
    a.setConstant(False)
    a.setUnits("milligram")
    
    c = model.createSpecies()
    c.setId("C")
    c.setCompartment("central")
    c.setInitialConcentration(0.0)
    c.setBoundaryCondition(False)
    c.setHasOnlySubstanceUnits(False)
    c.setConstant(False)
    c.setUnits("milligram_per_litre")
    
    cp = model.createSpecies()
    cp.setId("Cp")
    cp.setCompartment("peripheral")
    cp.setInitialConcentration(0.0)
    cp.setBoundaryCondition(False)
    cp.setHasOnlySubstanceUnits(False)
    cp.setConstant(False)
    cp.setUnits("milligram_per_litre")
    
    
    # Define the dosing schedule using an assignment rule for Dsc
    dsc_param = model.createParameter()
    dsc_param.setId("Dsc")
    dsc_param.setConstant(False)
    dsc_param.setUnits("milligram")
    
    # Build the piecewise function for SC dosing
    pieces = []
    for t_dose, amount in sc_doses:
        condition = f"(time >= {t_dose}) && (time < {t_dose} + D1)"
        pieces.append(f"{amount}, {condition}")
    
    # Combine all pieces into a piecewise function
    piecewise_expression = "piecewise(" + ", ".join(pieces) + ", 0)"
    
    # Create an assignment rule for Dsc
    dsc_rule = model.createAssignmentRule()
    dsc_rule.setVariable("Dsc")
    math_ast = libsbml.parseL3Formula(piecewise_expression)
    dsc_rule.setMath(math_ast)
    
    # Add IV dosing parameter
    div_param = model.createParameter()
    div_param.setId("Div")
    div_param.setConstant(False)
    div_param.setUnits("milligram")
    
    # Build the piecewise function for IV dosing
    iv_pieces = []
    for t_dose, amount in iv_doses:
        condition = f"(time >= {t_dose}) && (time < {t_dose} + 0.001)"
        iv_pieces.append(f"{amount}, {condition}")
    
    # Combine all IV pieces into a piecewise function
    iv_piecewise_expression = "piecewise(" + ", ".join(iv_pieces) + ", 0)"
    
    # Create an assignment rule for Div
    div_rule = model.createAssignmentRule()
    div_rule.setVariable("Div")
    math_ast = libsbml.parseL3Formula(iv_piecewise_expression)
    div_rule.setMath(math_ast)

    
    
    # Now define the reactions
    # Absorption Reaction: A -> C
    r1 = model.createReaction()
    r1.setId("absorption")
    r1.setReversible(False)
    
    r1_react = r1.createReactant()
    r1_react.setSpecies("A")
    r1_react.setStoichiometry(1.0)
    r1_react.setConstant(False)
    
    r1_prod = r1.createProduct()
    r1_prod.setSpecies("C")
    r1_prod.setStoichiometry(1.0)
    r1_prod.setConstant(False)
    
    # Kinetic Law: Rate of absorption 
    math_ast = libsbml.parseL3Formula("KA * A")
    r1_kl = r1.createKineticLaw()
    r1_kl.setMath(math_ast)
    
    # Dosing Reaction: -> A
    r_sc_dose = model.createReaction()
    r_sc_dose.setId("sc_dosing")
    r_sc_dose.setReversible(False)
    
    # Product: A
    r_sc_dose_prod = r_sc_dose.createProduct()
    r_sc_dose_prod.setSpecies("A")
    r_sc_dose_prod.setStoichiometry(1.0)
    r_sc_dose_prod.setConstant(False)
    
    # Kinetic Law: Rate of dosing input
    math_ast = libsbml.parseL3Formula("F * Dsc / D1")
    r_sc_dose_kl = r_sc_dose.createKineticLaw()
    r_sc_dose_kl.setMath(math_ast)

    # IV Dosing Reaction: -> C
    r_iv_dose = model.createReaction()
    r_iv_dose.setId("iv_dosing")
    r_iv_dose.setReversible(False)
    
    # Product: C
    r_iv_dose_prod = r_iv_dose.createProduct()
    r_iv_dose_prod.setSpecies("C")
    r_iv_dose_prod.setStoichiometry(1.0)
    r_iv_dose_prod.setConstant(False)
    
    # Kinetic Law: Rate of dosing input
    math_ast = libsbml.parseL3Formula("Div")
    r_iv_dose_kl = r_iv_dose.createKineticLaw()
    r_iv_dose_kl.setMath(math_ast)
    
    # Distribution from central to peripheral: C -> Cp
    r2 = model.createReaction()
    r2.setId("central_to_peripheral")
    r2.setReversible(False)
    
    r2_react = r2.createReactant()
    r2_react.setSpecies("C")
    r2_react.setStoichiometry(1.0)
    r2_react.setConstant(False)
    
    r2_prod = r2.createProduct()
    r2_prod.setSpecies("Cp")
    r2_prod.setStoichiometry(1.0)
    r2_prod.setConstant(False)
    
    # Kinetic Law: Rate of distribution (using amounts)
    math_ast = libsbml.parseL3Formula("Q * C")
    r2_kl = r2.createKineticLaw()
    r2_kl.setMath(math_ast)
    
    # Distribution from peripheral to central: Cp -> C
    r3 = model.createReaction()
    r3.setId("peripheral_to_central")
    r3.setReversible(False)
    
    r3_react = r3.createReactant()
    r3_react.setSpecies("Cp")
    r3_react.setStoichiometry(1.0)
    r3_react.setConstant(False)
    
    r3_prod = r3.createProduct()
    r3_prod.setSpecies("C")
    r3_prod.setStoichiometry(1.0)
    r3_prod.setConstant(False)
    
    # Kinetic Law: Rate of redistribution (using amounts)
    math_ast = libsbml.parseL3Formula("Q * Cp")
    r3_kl = r3.createKineticLaw()
    r3_kl.setMath(math_ast)
    
    # Elimination from central compartment: C ->
    r4 = model.createReaction()
    r4.setId("elimination")
    r4.setReversible(False)
    
    r4_react = r4.createReactant()
    r4_react.setSpecies("C")
    r4_react.setStoichiometry(1.0)
    r4_react.setConstant(False)
    
    # Kinetic Law: Rate of elimination
    math_ast = libsbml.parseL3Formula("CL * C")
    r4_kl = r4.createKineticLaw()
    r4_kl.setMath(math_ast)
    
    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)


def main(params, sc_doses, iv_doses):
    # Create the model
    document = create_pk_model(params, sc_doses, iv_doses)
    
    # Always save a new XML file, overwriting any existing one
    save_model(document, "pk_sbml.xml")
    
    # Validate after saving
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
        return False
    else:
        print("Model saved successfully!")
        return True

if __name__ == "__main__":
    main(params, sc_doses, iv_doses)