import libsbml
from pathlib import Path
import sys
from src.models.pk_sbml import create_pk_model
from src.models.vwd.vwd_sbml import create_vwd_model

def create_master_model(params):
    """Create a master SBML model that combines PK and VWD models"""
    
    # Extract parameters and dosing schedules
    pk_params = params['pk']
    vwd_params = params['vwd']
    sc_doses = params['sc_doses']
    iv_doses = params['iv_doses']
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Aldea_Model")
    model.setTimeUnits("day")

    # Create individual models
    pk_doc = create_pk_model(pk_params, sc_doses, iv_doses)
    vwd_doc = create_vwd_model(vwd_params)
    
    # Get models
    pk_model = pk_doc.getModel()
    vwd_model = vwd_doc.getModel()
    
    # Transfer all compartments, parameters, species, rules, and reactions
    all_models = [pk_model, vwd_model]
    for source_model in all_models:
        for i in range(source_model.getNumCompartments()):
            comp = source_model.getCompartment(i)
            if not model.getCompartment(comp.getId()):  # Check if compartment already exists
                new_comp = model.createCompartment()
                new_comp.setId(comp.getId())
                new_comp.setConstant(comp.getConstant())
                new_comp.setSize(comp.getSize())
                new_comp.setUnits(comp.getUnits())

    # Transfer all parameters from all models
    for source_model in all_models:
        for i in range(source_model.getNumParameters()):
            param = source_model.getParameter(i)
            if not model.getParameter(param.getId()):  # Check if parameter already exists
                new_param = model.createParameter()
                new_param.setId(param.getId())
                new_param.setValue(param.getValue())
                new_param.setConstant(param.getConstant())
                if param.isSetUnits():
                    new_param.setUnits(param.getUnits())

    # Transfer all species from all models
    for source_model in all_models:
        for i in range(source_model.getNumSpecies()):
            species = source_model.getSpecies(i)
            if not model.getSpecies(species.getId()):  # Check if species already exists
                new_species = model.createSpecies()
                new_species.setId(species.getId())
                new_species.setConstant(species.getConstant())
                new_species.setBoundaryCondition(species.getBoundaryCondition())
                new_species.setCompartment(species.getCompartment())
                new_species.setHasOnlySubstanceUnits(species.getHasOnlySubstanceUnits())
                
                # Only set initial amount if it's set in the source species
                if species.isSetInitialAmount():
                    new_species.setInitialAmount(species.getInitialAmount())
                # Otherwise set initial concentration if it's set
                elif species.isSetInitialConcentration():
                    new_species.setInitialConcentration(species.getInitialConcentration())
                
                new_species.setSubstanceUnits(species.getSubstanceUnits())
                new_species.setUnits(species.getUnits())

    # Transfer all rules from all models
    for source_model in all_models:
        for i in range(source_model.getNumRules()):
            rule = source_model.getRule(i)
            if rule.isAssignment():
                new_rule = model.createAssignmentRule()
                new_rule.setVariable(rule.getVariable())
                new_rule.setMath(rule.getMath().deepCopy())
            elif rule.isRate():
                new_rule = model.createRateRule()
                new_rule.setVariable(rule.getVariable())
                new_rule.setMath(rule.getMath().deepCopy())

    # Transfer all reactions from all models
    for source_model in all_models:
        for i in range(source_model.getNumReactions()):
            reaction = source_model.getReaction(i)
            new_reaction = model.createReaction()
            new_reaction.setId(reaction.getId())
            new_reaction.setReversible(reaction.getReversible())

            for j in range(reaction.getNumReactants()):
                ref = reaction.getReactant(j)
                new_ref = new_reaction.createReactant()
                new_ref.setSpecies(ref.getSpecies())
                new_ref.setStoichiometry(ref.getStoichiometry())
                new_ref.setConstant(ref.getConstant())

            for j in range(reaction.getNumProducts()):
                ref = reaction.getProduct(j)
                new_ref = new_reaction.createProduct()
                new_ref.setSpecies(ref.getSpecies())
                new_ref.setStoichiometry(ref.getStoichiometry())
                new_ref.setConstant(ref.getConstant())

            if reaction.isSetKineticLaw():
                kl = reaction.getKineticLaw()
                new_kl = new_reaction.createKineticLaw()
                new_kl.setMath(kl.getMath().deepCopy())

    return document

def save_model(document, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    libsbml.writeSBMLToFile(document, str(output_dir / "aldea_sbml.xml")) 