import libsbml
from pathlib import Path
from src.models.csf.csf_sbml import create_csf_model
from src.models.brain.brain_sbml import create_brain_model
from src.models.blood.blood_sbml import create_blood_model
from src.models.lung.lung_sbml import create_lung_model
from src.models.liver.liver_sbml import create_liver_model

def check(value, message):
    """If 'value' is None, prints an error message constructed using
    'message' and then exits with status code 1. If 'value' is an integer,
    it assumes it is a libSBML return status code. If the code value is
    LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
    prints an error message constructed using 'message' along with text from
    libSBML explaining the meaning of the code, and exits with status code 1.
    """
    if value is None:
        raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
    elif type(value) is int:
        if value == libsbml.LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = 'Error encountered trying to ' + message + '.' \
                    + 'LibSBML returned error code ' + str(value) + ': "' \
                    + libsbml.OperationReturnValue_toString(value).strip() + '"'
            raise SystemExit(err_msg)
    else:
        return

def create_master_model(params):
    """Create a master SBML model that combines all organ models"""
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Master_Model")
    model.setTimeUnits("hour")

    # Create individual models
    blood_doc = create_blood_model(params)
    lung_doc = create_lung_model(params)
    brain_doc = create_brain_model(params)
    csf_doc = create_csf_model(params)
    liver_doc = create_liver_model(params)
    
    # Get models
    blood_model = blood_doc.getModel()
    lung_model = lung_doc.getModel()
    brain_model = brain_doc.getModel()
    csf_model = csf_doc.getModel()
    liver_model = liver_doc.getModel()

    # Transfer all compartments from all models
    for source_model in [blood_model, lung_model, brain_model, csf_model, liver_model]:
        for i in range(source_model.getNumCompartments()):
            comp = source_model.getCompartment(i)
            new_comp = model.createCompartment()
            new_comp.setId(comp.getId())
            new_comp.setConstant(comp.getConstant())
            new_comp.setSize(comp.getSize())
            new_comp.setUnits(comp.getUnits())

    # Transfer all parameters from all models
    for source_model in [blood_model, lung_model, brain_model, csf_model, liver_model]:
        for i in range(source_model.getNumParameters()):
            param = source_model.getParameter(i)
            # Skip if parameter already exists (shared between models)
            if not model.getParameter(param.getId()):
                new_param = model.createParameter()
                new_param.setId(param.getId())
                new_param.setValue(param.getValue())
                new_param.setConstant(param.getConstant())
                if param.isSetUnits():
                    new_param.setUnits(param.getUnits())

    # Transfer all species from all models
    for source_model in [blood_model, lung_model, brain_model, csf_model, liver_model]:
        for i in range(source_model.getNumSpecies()):
            species = source_model.getSpecies(i)
            new_species = model.createSpecies()
            new_species.setId(species.getId())
            new_species.setConstant(species.getConstant())
            new_species.setBoundaryCondition(species.getBoundaryCondition())
            new_species.setCompartment(species.getCompartment())
            new_species.setHasOnlySubstanceUnits(species.getHasOnlySubstanceUnits())
            new_species.setInitialAmount(species.getInitialAmount())
            new_species.setInitialConcentration(species.getInitialConcentration())
            new_species.setSubstanceUnits(species.getSubstanceUnits())
            new_species.setUnits(species.getUnits())

    # Transfer all rules from all models
    for source_model in [blood_model, lung_model, brain_model, csf_model, liver_model]:
        for i in range(source_model.getNumRules()):
            rule = source_model.getRule(i)
            if rule.isRate():  # This should be true for all our rules
                new_rule = model.createRateRule()
                new_rule.setVariable(rule.getVariable())
                new_rule.setMath(rule.getMath().deepCopy())

    # Validate the model
    if document.getNumErrors() > 0:
        print("Validation errors:")
        document.printErrors()
        return None
        
    return document
    
    