import libsbml
from pathlib import Path
import sys

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))
from pk_module.pk_sbml import create_pk_model
from Retout2022.src.models.hazard.hazard_sbml import create_hazard_model

def create_master_model(params):
    """Create a master SBML model that combines PK and PET models"""
    
    # Extract parameters and dosing schedules
    pk_params = params['pk']
    pet_params = params['pet']
    sc_doses = params['sc_doses']
    iv_doses = params['iv_doses']
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Retout_Model")
    model.setTimeUnits("day")

    # Create individual models
    pk_doc = create_pk_model(pk_params, sc_doses, iv_doses)
    hazard_doc = create_hazard_model(pet_params)
    
    # Get models
    pk_model = pk_doc.getModel()
    hazard_model = hazard_doc.getModel()
    
    # Transfer all compartments, parameters, species, rules, and reactions
    all_models = [pk_model, hazard_model]
    for source_model in all_models:
        for i in range(source_model.getNumCompartments()):
            comp = source_model.getCompartment(i)
            if not model.getCompartment(comp.getId()):
                new_comp = model.createCompartment()
                new_comp.setId(comp.getId())
                new_comp.setConstant(comp.getConstant())
                new_comp.setSize(comp.getSize())
                new_comp.setUnits(comp.getUnits())

        for i in range(source_model.getNumParameters()):
            param = source_model.getParameter(i)
            if not model.getParameter(param.getId()):
                new_param = model.createParameter()
                new_param.setId(param.getId())
                new_param.setValue(param.getValue())
                new_param.setConstant(param.getConstant())
                if param.isSetUnits():
                    new_param.setUnits(param.getUnits())

        for i in range(source_model.getNumSpecies()):
            species = source_model.getSpecies(i)
            if not model.getSpecies(species.getId()):
                new_species = model.createSpecies()
                new_species.setId(species.getId())
                new_species.setConstant(species.getConstant())
                new_species.setBoundaryCondition(species.getBoundaryCondition())
                new_species.setCompartment(species.getCompartment())
                new_species.setHasOnlySubstanceUnits(species.getHasOnlySubstanceUnits())
                
                if species.isSetInitialAmount():
                    new_species.setInitialAmount(species.getInitialAmount())
                elif species.isSetInitialConcentration():
                    new_species.setInitialConcentration(species.getInitialConcentration())
                
                new_species.setSubstanceUnits(species.getSubstanceUnits())
                new_species.setUnits(species.getUnits())

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
    libsbml.writeSBMLToFile(document, str(output_dir / "retout_sbml.xml")) 