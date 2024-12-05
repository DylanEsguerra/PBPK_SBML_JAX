import libsbml

def create_ab_production_model(params):
    """
    Create an amyloid-beta production model using specific Michaelis-Menten kinetics
    """
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("AB_Production_Model")
    model.setTimeUnits("day")

    # Add parameters
    for param_id, value in params.items():
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)

    # Create compartment
    local_amyloid = model.createCompartment()
    local_amyloid.setId("local_amyloid")
    local_amyloid.setConstant(True)
    local_amyloid.setSize(1.0)
    local_amyloid.setUnits("dimensionless")

    # Add all species
    species_list = [
        ("APP", params["APP0"]),
        ("C83", params["C83_0"]),
        ("C99", params["C99_0"]),
        ("p3", params["p3_0"]),
        ("A_beta", params["A_beta0"])
    ]

    for species_id, initial_amount in species_list:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment("local_amyloid")
        species.setInitialAmount(initial_amount)
        species.setBoundaryCondition(False)
        species.setHasOnlySubstanceUnits(True)
        species.setConstant(False)
        species.setUnits("dimensionless")

    # r0: APP production (constant)
    r0 = model.createReaction()
    r0.setId("app_production")
    r0.setReversible(False)

    r0_prod = r0.createProduct()
    r0_prod.setSpecies("APP")
    r0_prod.setStoichiometry(1.0)
    r0_prod.setConstant(False)

    math_ast = libsbml.parseL3Formula("vr0")
    r0_kl = r0.createKineticLaw()
    r0_kl.setMath(math_ast)

    # r1: APP -> C83 (alpha-secretase)
    r1 = model.createReaction()
    r1.setId("app_to_c83")
    r1.setReversible(False)

    r1_react = r1.createReactant()
    r1_react.setSpecies("APP")
    r1_react.setStoichiometry(1.0)
    r1_react.setConstant(False)

    r1_prod = r1.createProduct()
    r1_prod.setSpecies("C83")
    r1_prod.setStoichiometry(1.0)
    r1_prod.setConstant(False)

    math_ast = libsbml.parseL3Formula("(Vm1 * APP / Km1) / (1 + APP/Km1 + C99/Km5)")
    r1_kl = r1.createKineticLaw()
    r1_kl.setMath(math_ast)

    # r2: APP -> C99 (beta-secretase)
    r2 = model.createReaction()
    r2.setId("app_to_c99")
    r2.setReversible(False)

    r2_react = r2.createReactant()
    r2_react.setSpecies("APP")
    r2_react.setStoichiometry(1.0)
    r2_react.setConstant(False)

    r2_prod = r2.createProduct()
    r2_prod.setSpecies("C99")
    r2_prod.setStoichiometry(1.0)
    r2_prod.setConstant(False)

    math_ast = libsbml.parseL3Formula("(Vm2 * APP / Km2) / (1 + APP/Km2)")
    r2_kl = r2.createKineticLaw()
    r2_kl.setMath(math_ast)

    # r3: C83 -> p3 (gamma-secretase) ***
    r3 = model.createReaction()
    r3.setId("c83_to_p3")
    r3.setReversible(False)

    r3_react = r3.createReactant()
    r3_react.setSpecies("C83")
    r3_react.setStoichiometry(1.0)
    r3_react.setConstant(False)

    r3_prod = r3.createProduct()
    r3_prod.setSpecies("p3")
    r3_prod.setStoichiometry(1.0)
    r3_prod.setConstant(False)

    math_ast = libsbml.parseL3Formula("(Vm3 * C83 / Km3) / (1 + C83/Km3 + C99/Km4)")
    r3_kl = r3.createKineticLaw()
    r3_kl.setMath(math_ast)

    # r4: C99 -> A_beta (gamma-secretase)
    r4 = model.createReaction()
    r4.setId("c99_to_abeta")
    r4.setReversible(False)

    r4_react = r4.createReactant()
    r4_react.setSpecies("C99")
    r4_react.setStoichiometry(1.0)
    r4_react.setConstant(False)

    r4_prod = r4.createProduct()
    r4_prod.setSpecies("A_beta")
    r4_prod.setStoichiometry(1.0)
    r4_prod.setConstant(False)

    math_ast = libsbml.parseL3Formula("(Vm4 * C99 / Km4) / (1 + C83/Km3 + C99/Km4)")
    r4_kl = r4.createKineticLaw()
    r4_kl.setMath(math_ast)

    # r5: C99 -> C83 (alpha-secretase)
    r5 = model.createReaction()
    r5.setId("c99_to_c83")
    r5.setReversible(False)

    r5_react = r5.createReactant()
    r5_react.setSpecies("C99")
    r5_react.setStoichiometry(1.0)
    r5_react.setConstant(False)

    r5_prod = r5.createProduct()
    r5_prod.setSpecies("C83")
    r5_prod.setStoichiometry(1.0)
    r5_prod.setConstant(False)

    math_ast = libsbml.parseL3Formula("(Vm5 * C99 / Km5) / (1 + APP/Km1 + C99/Km5)")
    r5_kl = r5.createKineticLaw()
    r5_kl.setMath(math_ast)

    return document

def save_model(document, filename):
    libsbml.writeSBMLToFile(document, filename)

def main(params=None):
        
    document = create_ab_production_model(params)
    
    if document.getNumErrors() > 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, "ab_production_sbml.xml")
        print("AB Production Model saved successfully!")

if __name__ == "__main__":
    main() 