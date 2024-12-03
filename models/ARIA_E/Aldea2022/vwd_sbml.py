import libsbml


def create_vwd_model(params):
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("VWD_Model")
    model.setTimeUnits("day")

    # Add parameters to model
    for param_id, value in params.items():
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)

    # Create compartments 
    local_amyloid = model.createCompartment()
    local_amyloid.setId("local_amyloid")
    local_amyloid.setConstant(True)
    local_amyloid.setSize(1.0)
    local_amyloid.setUnits("dimensionless")

    vwd_compartment = model.createCompartment()
    vwd_compartment.setId("VWD_compartment")
    vwd_compartment.setConstant(True)
    vwd_compartment.setSize(1.0)
    vwd_compartment.setUnits("dimensionless")

    # Add species
    abeta = model.createSpecies()
    abeta.setId("A_beta")
    abeta.setCompartment("local_amyloid")
    abeta.setInitialAmount(params["A_beta0"])
    abeta.setBoundaryCondition(False)
    abeta.setHasOnlySubstanceUnits(True)
    abeta.setConstant(False)
    abeta.setUnits("dimensionless")
    
    vwd = model.createSpecies()
    vwd.setId("VWD")
    vwd.setCompartment("VWD_compartment")
    vwd.setInitialAmount(params["VWD0"])
    vwd.setBoundaryCondition(False)
    vwd.setHasOnlySubstanceUnits(True)
    vwd.setConstant(False)
    vwd.setUnits("dimensionless")

    # Add central concentration as a parameter (will be updated from PK model)
    central_conc = model.createParameter()
    central_conc.setId("central_conc")
    central_conc.setValue(0.0)
    central_conc.setConstant(False)

    # PD Model reactions
    # Amyloid degradation: A_beta ->
    r1 = model.createReaction()
    r1.setId("amyloid_degradation")
    r1.setReversible(False)
    
    r1_react = r1.createReactant()
    r1_react.setSpecies("A_beta")
    r1_react.setStoichiometry(1.0)
    r1_react.setConstant(False)
    
    # Kinetic Law: Rate of amyloid degradation
    math_ast = libsbml.parseL3Formula("alpha_removal * central_conc * A_beta")
    r1_kl = r1.createKineticLaw()
    r1_kl.setMath(math_ast)
    
    # VWD production: -> VWD
    r2 = model.createReaction()
    r2.setId("vwd_production")
    r2.setReversible(False)
    
    r2_prod = r2.createProduct()
    r2_prod.setSpecies("VWD")
    r2_prod.setStoichiometry(1.0)
    r2_prod.setConstant(False)
    
    # Kinetic Law: Rate of VWD production
    math_ast = libsbml.parseL3Formula("alpha_removal * central_conc * A_beta")
    r2_kl = r2.createKineticLaw()
    r2_kl.setMath(math_ast)
    
    # VWD degradation: VWD ->
    r3 = model.createReaction()
    r3.setId("vwd_degradation")
    r3.setReversible(False)
    
    r3_react = r3.createReactant()
    r3_react.setSpecies("VWD")
    r3_react.setStoichiometry(1.0)
    r3_react.setConstant(False)
    
    # Kinetic Law: Rate of VWD degradation
    math_ast = libsbml.parseL3Formula("k_repair * VWD")
    r3_kl = r3.createKineticLaw()
    r3_kl.setMath(math_ast)

    # Add BGTS calculation
    bgts = model.createSpecies()
    bgts.setId("BGTS")
    bgts.setCompartment("VWD_compartment")
    bgts.setInitialAmount(0.0)
    bgts.setBoundaryCondition(False)
    bgts.setHasOnlySubstanceUnits(True)
    bgts.setConstant(False)
    
    # Add an assignment rule for BGTS
    rule = model.createAssignmentRule()
    rule.setVariable("BGTS")
    math_ast = libsbml.parseL3Formula("BGTS_max * ((VWD / EG50) ^ pow) / (1 + (VWD / EG50) ^ pow)")
    rule.setMath(math_ast)

    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    document = create_vwd_model(params)
    
    # Validate the model
    if document.getNumErrors() > 0:
        print("Validation errors:")
        document.printErrors()
    else:
        # Save the model to file
        save_model(document, "vwd_sbml.xml")
        print("VWD Model saved successfully!")

if __name__ == "__main__":
    main(params) 