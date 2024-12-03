import libsbml


def create_suvr_model(params):
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("PET_Model")
    model.setTimeUnits("day")

    # Add parameters to model
    for param_id, value in params.items():
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)


    # Create compartment
    effect_comp = model.createCompartment()
    effect_comp.setId("effect_compartment")
    effect_comp.setConstant(True)
    effect_comp.setSize(1.0)
    effect_comp.setUnits("dimensionless")

    # Add species for effect compartment only
    ce = model.createSpecies()
    ce.setId("Ce")
    ce.setCompartment("effect_compartment")
    ce.setInitialAmount(0.0)
    ce.setBoundaryCondition(False)
    ce.setHasOnlySubstanceUnits(True)
    ce.setConstant(False)
    ce.setUnits("dimensionless")

    # Add central concentration as a parameter (will be updated from PK model)
    central_conc = model.createParameter()
    central_conc.setId("central_conc")
    central_conc.setValue(0.0)
    central_conc.setConstant(False)

    
    # Effect compartment kinetics with proper scaling
    r1 = model.createReaction()
    r1.setId("effect_compartment_kinetics")
    r1.setReversible(False)
    
    r1_prod = r1.createProduct()
    r1_prod.setSpecies("Ce")
    r1_prod.setStoichiometry(1.0)
    r1_prod.setConstant(False)
    
    # Kinetic Law: dCe/dt = Ke0 * (central_conc - Ce)
    math_ast = libsbml.parseL3Formula("Ke0 * (central_conc - Ce)")
    r1_kl = r1.createKineticLaw()
    r1_kl.setMath(math_ast)
    
    # SUVR assignment rule with proper scaling
    # Add SUVR as a species 
    suvr = model.createSpecies()
    suvr.setId("SUVR")
    suvr.setCompartment("effect_compartment")
    suvr.setInitialAmount(params["SUVR_0"])
    suvr.setBoundaryCondition(False)
    suvr.setHasOnlySubstanceUnits(True)
    suvr.setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("SUVR")
    math_ast = libsbml.parseL3Formula("SUVR_0 * (1 - SLOP * Ce ^ power)")
    rule.setMath(math_ast)

    # Add ARIA hazard as a parameter 
    aria = model.createSpecies()
    aria.setId("ARIA_hazard")
    aria.setCompartment("effect_compartment")
    aria.setInitialAmount(0.0)
    aria.setBoundaryCondition(False)
    aria.setHasOnlySubstanceUnits(True)
    aria.setConstant(False)
    aria.setUnits("dimensionless")

    # Add ARIA hazard assignment rule
    aria_rule = model.createAssignmentRule()
    aria_rule.setVariable("ARIA_hazard")

    # Create the mathematical expression for the hazard function
    math_ast = libsbml.parseL3Formula(
        "ln(BSAPOE4_carrier) + (Emax * central_conc)/(central_conc + EC50) * (T50^gamma)/(T50^gamma + time^gamma)"
    )
    aria_rule.setMath(math_ast)

    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params):
    document = create_suvr_model(params)
    
    # Validate the model
    if document.getNumErrors() > 0:
        print("Validation errors:")
        document.printErrors()
    else:
        # Save the model to file
        save_model(document, "pet_sbml.xml")
        print("PET Model saved successfully!")

if __name__ == "__main__":
    main(params)