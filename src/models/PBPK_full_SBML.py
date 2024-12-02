import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path):
    """Load parameters from CSV file into dictionary with values and units"""
    print(f"Loading parameters from {csv_path}")
    df = pd.read_csv(csv_path)
    print("\nFirst few rows of parameter file:")
    print(df.head())
    
    params = dict(zip(df['name'], df['value']))
    params_with_units = dict(zip(df['name'], zip(df['value'], df['units'])))
    
    print("\nFirst few parameters loaded:")
    for i, (key, value) in enumerate(params.items()):
        if i < 5:  # Print first 5 parameters
            print(f"{key}: {value} ({params_with_units[key][1]})")
    
    return params, params_with_units

def create_pbpk_model(params, params_with_units):
    """Create unified PBPK SBML model combining all compartments"""
    print("\nCreating PBPK model...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("PBPK_Model")
    model.setTimeUnits("hour")

    # Debug print for required parameters
    print("\nChecking if key parameters exist:")
    key_params = ["Vp", "Vbc", "Vlymphnode", "Q_p_lung"]
    for param in key_params:
        print(f"{param}: {'✓' if param in params else '✗'}")

    # Initialize compartments (all are constant)
    compartments = {
        "plasma": params["Vp"],
        "blood_cells": params["Vbc"],
        "lymph_node": params["Vlymphnode"],
        
        # Lung (special case - always keep separate)
        "Vp_lung": params["Vp_lung"],
        "VBC_lung": params["VBC_lung"],
        "VIS_lung": params["VIS_lung"],
        "VES_lung": params["VES_lung"],
        
        # Brain (special case - always keep separate)
        "Vp_brain": params["Vp_brain"],
        "VIS_brain": params["VIS_brain"],
        "VBBB_brain": params["VBBB_brain"],
        "VBC_brain": params["VBC_brain"],
        "V_ES_brain": params["V_ES_brain"],
        "V_BCSFB_brain": params["V_BCSFB_brain"],
        "V_LV_brain": params["V_LV_brain"],
        "V_TFV_brain": params["V_TFV_brain"],
        "V_CM_brain": params["V_CM_brain"],
        "V_SAS_brain": params["V_SAS_brain"],
        
        # Liver (special case - always keep separate)
        "Vp_liver": params["Vp_liver"],
        "VBC_liver": params["VBC_liver"],
        "VIS_liver": params["VIS_liver"],
        "VES_liver": params["VES_liver"],
    }

    # Define standard organs (moved to top of file)
    standard_organs = [
        'heart', 'muscle', 'kidney', 'skin', 'fat', 'marrow', 'thymus', 
        'SI', 'LI', 'spleen', 'pancreas', 'other'
    ]
    
    # Add standard organs to compartments dictionary
    for organ in standard_organs:
        compartments.update({
            f"Vp_{organ}": params[f"Vp_{organ}"],
            f"VBC_{organ}": params[f"VBC_{organ}"],
            f"VIS_{organ}": params[f"VIS_{organ}"],
            f"VES_{organ}": params[f"VES_{organ}"]
        })

    # Create all parameters first (before any species or rules)
    print("\nCreating parameters...")
    required_params = [
        # Core volumes (always keep)
        ("Vp", params["Vp"]), 
        ("Vbc", params["Vbc"]), 
        ("Vlymphnode", params["Vlymphnode"]),
        

        # Lung parameters
        ("Vp_lung", params["Vp_lung"]),
        ("VBC_lung", params["VBC_lung"]),
        ("VIS_lung", params["VIS_lung"]),
        ("VES_lung", params["VES_lung"]),
        ("Q_p_lung", params["Q_p_lung"]),
        ("Q_bc_lung", params["Q_bc_lung"]),
        ("L_lung", params["L_lung"]),
        ("sigma_V_lung", params["sigma_V_lung"]),
        ("sigma_L_lung", params["sigma_L_lung"]),
        ("CLup_lung", params["CLup_lung"]),
        
        # Brain and CSF parameters (including BBB and CSF specific)
        ("Vp_brain", params["Vp_brain"]),
        ("VIS_brain", params["VIS_brain"]),
        ("VBBB_brain", params["VBBB_brain"]),
        ("VBC_brain", params["VBC_brain"]),
        ("V_ES_brain", params["V_ES_brain"]),
        ("V_BCSFB_brain", params["V_BCSFB_brain"]),
        ("V_LV_brain", params["V_LV_brain"]),
        ("V_TFV_brain", params["V_TFV_brain"]),
        ("V_CM_brain", params["V_CM_brain"]),
        ("V_SAS_brain", params["V_SAS_brain"]),
        ("Q_p_brain", params["Q_p_brain"]),
        ("Q_bc_brain", params["Q_bc_brain"]),
        ("L_brain", params["L_brain"]),
        ("sigma_V_brain", params["sigma_V_brain"]),
        ("sigma_L_brain", params["sigma_L_brain"]),
        ("sigma_L_SAS", params["sigma_L_SAS"]),
        ("sigma_L_brain_ISF", params["sigma_L_brain_ISF"]),
        ("CLup_brain", params["CLup_brain"]),
        ("Q_CSF_brain", params["Q_CSF_brain"]),    
        ("Q_ISF_brain", params["Q_ISF_brain"]),
        ("Q_ECF_brain", params["Q_ECF_brain"]),
        ("sigma_V_BBB", params["sigma_V_BBB"]),
        ("sigma_V_BCSFB", params["sigma_V_BCSFB"]),
       
        
        # Liver parameters
        ("Vp_liver", params["Vp_liver"]),
        ("VBC_liver", params["VBC_liver"]),
        ("VIS_liver", params["VIS_liver"]),
        ("VES_liver", params["VES_liver"]),
        ("Q_p_liver", params["Q_p_liver"]),
        ("Q_bc_liver", params["Q_bc_liver"]),
        ("L_liver", params["L_liver"]),
        ("sigma_V_liver", params["sigma_V_liver"]),
        ("sigma_L_liver", params["sigma_L_liver"]),
        ("CLup_liver", params["CLup_liver"]),

        # Lymph flows
        ("L_lung", params["L_lung"]),
        ("L_brain", params["L_brain"]),
        ("L_heart", params["L_heart"]),
        ("L_liver", params["L_liver"]),
        ("L_kidney", params["L_kidney"]),
        ("L_muscle", params["L_muscle"]),
        ("L_skin", params["L_skin"]),
        ("L_fat", params["L_fat"]),
        ("L_marrow", params["L_marrow"]),
        ("L_thymus", params["L_thymus"]),
        ("L_SI", params["L_SI"]),
        ("L_LI", params["L_LI"]),
        ("L_spleen", params["L_spleen"]),
        ("L_pancreas", params["L_pancreas"]),
        ("L_other", params["L_other"]),
        ("L_LN", params["L_LN"]),
        
        # Global kinetic parameters
        ("kon_FcRn", params["kon_FcRn"]),       # FcRn binding rate
        ("koff_FcRn", params["koff_FcRn"]),     # FcRn unbinding rate
        ("kdeg", params["kdeg"]),               # Degradation rate
        ("FR", params["FR"]),                   # Global recycling fraction
        ("f_BBB", params["f_BBB"]),
        ("f_LV", params["f_LV"]),
        ("f_BCSFB", params["f_BCSFB"]),
        ("FcRn_free_BBB", params["FcRn_free_BBB"]),
        ("FcRn_free_BCSFB", params["FcRn_free_BCSFB"]),
    ]

    # Add standard organ parameters through loop
    for organ in standard_organs:
        required_params.extend([
            # Volumes
            (f"Vp_{organ}", params[f"Vp_{organ}"]),
            (f"VBC_{organ}", params[f"VBC_{organ}"]),
            (f"VIS_{organ}", params[f"VIS_{organ}"]),
            (f"VES_{organ}", params[f"VES_{organ}"]),
            # Flows
            (f"Q_p_{organ}", params[f"Q_p_{organ}"]),
            (f"Q_bc_{organ}", params[f"Q_bc_{organ}"]),
            (f"L_{organ}", params[f"L_{organ}"]),
            # Transport parameters
            (f"sigma_V_{organ}", params[f"sigma_V_{organ}"]),
            (f"sigma_L_{organ}", params[f"sigma_L_{organ}"]),
            (f"CLup_{organ}", params[f"CLup_{organ}"]),
           
        ])

    # Create parameters
    for param_id, value in required_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        if param_id in params_with_units:
            param.setUnits(params_with_units[param_id][1])
            print(f"Created parameter {param_id}: {value} {params_with_units[param_id][1]}")


    # Initialize species list - only include variables (with ODEs)
    variable_species = [
        # Blood/Lymph species (have ODEs)
        ("C_p", "plasma", params["C_p_0"]),
        ("C_bc", "blood_cells", params["C_bc_0"]),
        ("C_ln", "lymph_node", params["C_ln_0"]),
        
        # Lung species (have ODEs)
        ("C_p_lung", "Vp_lung", params["C_p_lung_0"]),
        ("C_bc_lung", "VBC_lung", params["C_bc_lung_0"]),
        ("C_is_lung", "VIS_lung", params["C_is_lung_0"]),
        ("C_e_unbound_lung", "VES_lung", params["C_e_unbound_lung_0"]),
        ("C_e_bound_lung", "VES_lung", params["C_e_bound_lung_0"]),
        ("FcRn_free_lung", "VES_lung", params["FcRn_free_lung_0"]),
        
        # Brain species (have ODEs)
        ("C_p_brain", "Vp_brain", params["C_p_brain_0"]),
        ("C_BBB_unbound_brain", "VBBB_brain", params["C_BBB_unbound_brain_0"]),
        ("C_BBB_bound_brain", "VBBB_brain", params["C_BBB_bound_brain_0"]),
        ("C_is_brain", "VIS_brain", params["C_is_brain_0"]),
        ("C_bc_brain", "VBC_brain", params["C_bc_brain_0"]),
        ("C_BCSFB_unbound_brain", "V_BCSFB_brain", params["C_BCSFB_unbound_brain_0"]),
        ("C_BCSFB_bound_brain", "V_BCSFB_brain", params["C_BCSFB_bound_brain_0"]),
        ("C_LV_brain", "V_LV_brain", params["C_LV_brain_0"]),
        ("C_TFV_brain", "V_TFV_brain", params["C_TFV_brain_0"]),
        ("C_CM_brain", "V_CM_brain", params["C_CM_brain_0"]),
        ("C_SAS_brain", "V_SAS_brain", params["C_SAS_brain_0"]),
        
        # Liver species (have ODEs)
        ("C_p_liver", "Vp_liver", params["C_p_liver_0"]),
        ("C_bc_liver", "VBC_liver", params["C_bc_liver_0"]),
        ("C_is_liver", "VIS_liver", params["C_is_liver_0"]),
        ("C_e_unbound_liver", "VES_liver", params["C_e_unbound_liver_0"]),
        ("C_e_bound_liver", "VES_liver", params["C_e_bound_liver_0"]),
        ("FcRn_free_liver", "VES_liver", params["FcRn_free_liver_0"]),
        
    ]


    # Add compartments for standard organs
    for organ in standard_organs:
        
        compartments.update({
            f"Vp_{organ}": params[f"Vp_{organ}"],
            f"VBC_{organ}": params[f"VBC_{organ}"],
            f"VIS_{organ}": params[f"VIS_{organ}"],
            f"VES_{organ}": params.get(f"VES_{organ}", 0.1)  # Default endosomal volume if not specified
        })

        variable_species.extend([
            (f"C_p_{organ}", f"Vp_{organ}", params[f"C_p_{organ}_0"]),
            (f"C_bc_{organ}", f"VBC_{organ}", params[f"C_bc_{organ}_0"]),
            (f"C_is_{organ}", f"VIS_{organ}", params[f"C_is_{organ}_0"]),
            (f"C_e_unbound_{organ}", f"VES_{organ}", params[f"C_e_unbound_{organ}_0"]),
            (f"C_e_bound_{organ}", f"VES_{organ}", params[f"C_e_bound_{organ}_0"]),
            (f"FcRn_free_{organ}", f"VES_{organ}", params[f"FcRn_free_{organ}_0"])
        ])
    #'''

     # Create compartments
    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setConstant(True)
        comp.setSize(size)
        comp.setUnits("millilitre")

    # Create variable species
    for species_id, compartment_id, initial_value in variable_species:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment(compartment_id)
        species.setInitialConcentration(initial_value)
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setConstant(False)
        species.setUnits("mole_per_litre")


    # Blood equations
    plasma_rule = model.createRateRule()
    plasma_rule.setVariable("C_p")
    plasma_eq = (
        f"-(Q_p_lung + L_lung) * C_p + "
        f"(Q_p_heart - L_heart) * C_p_heart + "  # Constant for now
        f"(Q_p_kidney - L_kidney) * C_p_kidney + "  # Constant for now
        f"(Q_p_brain - L_brain) * C_p_brain + "
        f"(Q_p_muscle - L_muscle) * C_p_muscle + "  # Constant for now
        f"(Q_p_marrow - L_marrow) * C_p_marrow + "  # Constant for now
        f"(Q_p_thymus - L_thymus) * C_p_thymus + "  # Constant for now
        f"(Q_p_skin - L_skin) * C_p_skin + "  # Constant for now
        f"(Q_p_fat - L_fat) * C_p_fat + "  # Constant for now
        f"((Q_p_SI - L_SI) + (Q_p_LI - L_LI) + "
        f"(Q_p_spleen - L_spleen) + "
        f"(Q_p_pancreas - L_pancreas) + "
        f"(Q_p_liver - L_liver)) * C_p_liver + "
        f"(Q_p_other - L_other) * C_p_other + "  # Constant for now
        f"L_LN * C_ln"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp * ({plasma_eq})")
    plasma_rule.setMath(math_ast)

    # Blood cells equation
    bc_rule = model.createRateRule()
    bc_rule.setVariable("C_bc")
    bc_eq = (
        f"-Q_bc_lung * C_bc + "
        f"Q_bc_heart * C_bc_heart + "  # Constant for now
        f"Q_bc_kidney * C_bc_kidney + "  # Constant for now
        f"Q_bc_brain * C_bc_brain + "
        f"Q_bc_muscle * C_bc_muscle + "  # Constant for now
        f"Q_bc_marrow * C_bc_marrow + "  # Constant for now
        f"Q_bc_thymus * C_bc_thymus + "  # Constant for now
        f"Q_bc_skin * C_bc_skin + "  # Constant for now
        f"Q_bc_fat * C_bc_fat + "  # Constant for now
        f"(Q_bc_SI + Q_bc_LI + Q_bc_spleen + "
        f"Q_bc_pancreas + Q_bc_liver) * C_bc_liver + "
        f"Q_bc_other * C_bc_other"  # Constant for now
    )
    math_ast = libsbml.parseL3Formula(f"1/Vbc * ({bc_eq})")
    bc_rule.setMath(math_ast)

    # Lymph node equation
    ln_rule = model.createRateRule()
    ln_rule.setVariable("C_ln")

    ln_eq = (
        f"(1-sigma_L_lung) * L_lung * C_is_lung + "
        f"(1-sigma_L_heart) * L_heart * C_is_heart + "
        f"(1-sigma_L_kidney) * L_kidney * C_is_kidney + "
        f"(1-sigma_L_SAS) * Q_CSF_brain * C_SAS_brain + "
        f"(1-sigma_L_brain_ISF) * Q_ECF_brain * C_is_brain + "
        f"(1-sigma_L_muscle) * L_muscle * C_is_muscle + "
        f"(1-sigma_L_marrow) * L_marrow * C_is_marrow + "
        f"(1-sigma_L_thymus) * L_thymus * C_is_thymus + "
        f"(1-sigma_L_skin) * L_skin * C_is_skin + "
        f"(1-sigma_L_fat) * L_fat * C_is_fat + "
        f"(1-sigma_L_SI) * L_SI * C_is_SI + "
        f"(1-sigma_L_LI) * L_LI * C_is_LI + "
        f"(1-sigma_L_spleen) * L_spleen * C_is_spleen + "
        f"(1-sigma_L_pancreas) * L_pancreas * C_is_pancreas + "
        f"(1-sigma_L_liver) * L_liver * C_is_liver + "
        f"(1-sigma_L_other) * L_other * C_is_other - "
        f"L_LN * C_ln"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vlymphnode * ({ln_eq})")
    ln_rule.setMath(math_ast)

    # Lung equations
    # Plasma equation
    lung_plasma_rule = model.createRateRule()
    lung_plasma_rule.setVariable("C_p_lung")
    lung_plasma_eq = (
        f"Q_p_lung * C_p - "
        f"Q_p_lung * C_p_lung - "
        f"(1 - sigma_V_lung) * L_lung * C_p_lung - "
        f"CLup_lung * C_p_lung + "
        f"CLup_lung * FR * C_e_bound_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_lung * ({lung_plasma_eq})")
    lung_plasma_rule.setMath(math_ast)

    # Lung blood cells equation
    lung_bc_rule = model.createRateRule()
    lung_bc_rule.setVariable("C_bc_lung")
    lung_bc_eq = (
        f"Q_bc_lung * C_bc - Q_bc_lung * C_bc_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_lung * ({lung_bc_eq})")
    lung_bc_rule.setMath(math_ast)

    # Lung endosomal unbound equation
    lung_e_unbound_rule = model.createRateRule()
    lung_e_unbound_rule.setVariable("C_e_unbound_lung")
    lung_e_unbound_eq = (
        f"CLup_lung * (C_p_lung + C_is_lung) - "
        f"VES_lung * kon_FcRn * C_e_unbound_lung * FcRn_free_lung + "
        f"VES_lung * koff_FcRn * C_e_bound_lung - "
        f"kdeg * C_e_unbound_lung * VES_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_lung * ({lung_e_unbound_eq})")
    lung_e_unbound_rule.setMath(math_ast)

    # Lung endosomal bound equation
    lung_e_bound_rule = model.createRateRule()
    lung_e_bound_rule.setVariable("C_e_bound_lung")
    lung_e_bound_eq = (
        f"VES_lung * kon_FcRn * C_e_unbound_lung * FcRn_free_lung - "
        f"VES_lung * koff_FcRn * C_e_bound_lung - "
        f"CLup_lung * C_e_bound_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_lung * ({lung_e_bound_eq})")
    lung_e_bound_rule.setMath(math_ast)

    # Lung ISF equation
    lung_is_rule = model.createRateRule()
    lung_is_rule.setVariable("C_is_lung")
    lung_is_eq = (
        f"(1 - sigma_V_lung) * L_lung * C_p_lung - "
        f"(1 - sigma_L_lung) * L_lung * C_is_lung + "
        f"CLup_lung * (1 - FR) * C_e_bound_lung - "
        f"CLup_lung * C_is_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_lung * ({lung_is_eq})")
    lung_is_rule.setMath(math_ast)

    # Lung FcRn free equation
    lung_fcrn_free_rule = model.createRateRule()
    lung_fcrn_free_rule.setVariable("FcRn_free_lung")
    lung_fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_lung * VES_lung - "
        f"kon_FcRn * C_e_unbound_lung * FcRn_free_lung * VES_lung + "
        f"CLup_lung * C_e_bound_lung"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_lung * ({lung_fcrn_free_eq})")
    lung_fcrn_free_rule.setMath(math_ast)

    # Brain equations
    # Plasma equation
    brain_plasma_rule = model.createRateRule()
    brain_plasma_rule.setVariable("C_p_brain")
    brain_plasma_eq = (
        f"(Q_p_brain * C_p_lung) - "
        f"(Q_p_brain - L_brain) * C_p_brain - "
        f"(1 - sigma_V_BBB) * Q_ISF_brain * C_p_brain - "
        f"(1 - sigma_V_BCSFB) * Q_CSF_brain * C_p_brain - "
        f"CLup_brain * V_ES_brain * C_p_brain + "
        f"CLup_brain * f_BBB * V_ES_brain * FR * C_BBB_bound_brain + "
        f"CLup_brain * (1-f_BBB) * V_ES_brain * FR * C_BCSFB_bound_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_brain * ({brain_plasma_eq})")
    brain_plasma_rule.setMath(math_ast)

    # BBB Unbound equation
    bbb_unbound_rule = model.createRateRule()
    bbb_unbound_rule.setVariable("C_BBB_unbound_brain")
    bbb_unbound_eq = (
        f"CLup_brain * f_BBB * V_ES_brain * (C_p_brain + C_is_brain) - "
        f"VBBB_brain * kon_FcRn * C_BBB_unbound_brain * FcRn_free_BBB + "
        f"VBBB_brain * koff_FcRn * C_BBB_bound_brain - "
        f"VBBB_brain * kdeg * C_BBB_unbound_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBBB_brain * ({bbb_unbound_eq})")
    bbb_unbound_rule.setMath(math_ast)

    # BBB Bound equation
    bbb_bound_rule = model.createRateRule()
    bbb_bound_rule.setVariable("C_BBB_bound_brain")
    bbb_bound_eq = (
        f"-CLup_brain * f_BBB * V_ES_brain * C_BBB_bound_brain + "
        f"VBBB_brain * kon_FcRn * C_BBB_unbound_brain * FcRn_free_BBB - "
        f"VBBB_brain * koff_FcRn * C_BBB_bound_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBBB_brain * ({bbb_bound_eq})")
    bbb_bound_rule.setMath(math_ast)

    # Brain ISF equation
    brain_isf_rule = model.createRateRule()
    brain_isf_rule.setVariable("C_is_brain")
    brain_isf_eq = (
        f"(1 - sigma_V_BBB) * Q_ISF_brain * C_p_brain - "
        f"(1 - sigma_L_brain_ISF) * Q_ISF_brain * C_is_brain - "
        f"Q_ISF_brain * C_is_brain + Q_ISF_brain * C_SAS_brain + "
        f"CLup_brain * f_BBB * V_ES_brain * (1 - FR) * C_BBB_bound_brain - "
        f"CLup_brain * f_BBB * V_ES_brain * C_is_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_brain * ({brain_isf_eq})")
    brain_isf_rule.setMath(math_ast)

    # CSF equations
    # BCSFB Unbound equation
    bcsfb_unbound_rule = model.createRateRule()
    bcsfb_unbound_rule.setVariable("C_BCSFB_unbound_brain")
    

    bcsfb_unbound_eq = (
        f"CLup_brain * f_BCSFB * V_ES_brain * C_p_brain + "
        f"f_LV * CLup_brain * (1 - f_BBB) * V_ES_brain * C_LV_brain + "
        f"(1 - f_LV) * CLup_brain * (1 - f_BBB) * V_ES_brain * C_TFV_brain - "
        f"V_BCSFB_brain * kon_FcRn * C_BCSFB_unbound_brain * FcRn_free_BCSFB + "
        f"V_BCSFB_brain * koff_FcRn * C_BCSFB_bound_brain - "
        f"V_BCSFB_brain * kdeg * C_BCSFB_unbound_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/V_BCSFB_brain * ({bcsfb_unbound_eq})")
    bcsfb_unbound_rule.setMath(math_ast)

    # BCSFB Bound equation
    bcsfb_bound_rule = model.createRateRule()
    bcsfb_bound_rule.setVariable("C_BCSFB_bound_brain")
    bcsfb_bound_eq = (
        f"-CLup_brain * (1-f_BBB) * V_ES_brain * C_BCSFB_bound_brain + "
        f"V_BCSFB_brain * kon_FcRn * C_BCSFB_unbound_brain * FcRn_free_BCSFB - "
        f"V_BCSFB_brain * koff_FcRn * C_BCSFB_bound_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/V_BCSFB_brain * ({bcsfb_bound_eq})")
    bcsfb_bound_rule.setMath(math_ast)

    # Lateral Ventricle equation
    lv_rule = model.createRateRule()
    lv_rule.setVariable("C_LV_brain")
    lv_eq = (
        f"(1-sigma_V_BCSFB) * f_LV * Q_CSF_brain * C_p_brain + "
        f"f_LV * Q_ISF_brain * C_is_brain - "
        f"(f_LV * Q_CSF_brain + f_LV * Q_ISF_brain) * C_LV_brain - "
        f"f_LV * CLup_brain * (1-f_BBB) * V_ES_brain * C_LV_brain + "
        f"f_LV * CLup_brain * (1-f_BBB) * V_ES_brain * (1-FR) * C_BCSFB_bound_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/V_LV_brain * ({lv_eq})")
    lv_rule.setMath(math_ast)

    # Third/Fourth Ventricle equation
    tfv_rule = model.createRateRule()
    tfv_rule.setVariable("C_TFV_brain")
    tfv_eq = (
        f"(1-sigma_V_BCSFB) * (1-f_LV) * Q_CSF_brain * C_p_brain + "
        f"(1-f_LV) * Q_ISF_brain * C_is_brain - "
        f"(Q_CSF_brain + Q_ISF_brain) * C_TFV_brain - "
        f"(1-f_LV) * CLup_brain * (1-f_BBB) * V_ES_brain * C_TFV_brain + "
        f"(1-f_LV) * CLup_brain * (1-f_BBB) * V_ES_brain * (1-FR) * C_BCSFB_bound_brain + "
        f"(f_LV * Q_CSF_brain + f_LV * Q_ISF_brain) * C_LV_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/V_TFV_brain * ({tfv_eq})")
    tfv_rule.setMath(math_ast)

    # Cisterna Magna equation
    cm_rule = model.createRateRule()
    cm_rule.setVariable("C_CM_brain")
    cm_eq = (
        f"(Q_CSF_brain + Q_ISF_brain) * (C_TFV_brain - C_CM_brain)"
    )
    math_ast = libsbml.parseL3Formula(f"1/V_CM_brain * ({cm_eq})")
    cm_rule.setMath(math_ast)

    # Subarachnoid Space equation
    sas_rule = model.createRateRule()
    sas_rule.setVariable("C_SAS_brain")
    sas_eq = (
        f"(Q_CSF_brain + Q_ISF_brain) * C_CM_brain - "
        f"(1-sigma_L_SAS) * Q_CSF_brain * C_SAS_brain - "
        f"Q_ISF_brain * C_SAS_brain"
    )
    math_ast = libsbml.parseL3Formula(f"1/V_SAS_brain * ({sas_eq})")
    sas_rule.setMath(math_ast)

    # Liver equations
    # Plasma equation
    liver_plasma_rule = model.createRateRule()
    liver_plasma_rule.setVariable("C_p_liver")
    liver_plasma_eq = (
        f"Q_p_liver * C_p_lung + "
        f"Q_p_spleen * C_p_spleen + "  # Constant for now
        f"Q_p_pancreas * C_p_pancreas + "  # Constant for now
        f"Q_p_SI * C_p_SI + "  # Constant for now
        f"Q_p_LI * C_p_LI - "  # Constant for now
        f"((Q_p_liver - L_liver) + "
        f"(Q_p_spleen - L_spleen) + "
        f"(Q_p_pancreas - L_pancreas) + "
        f"(Q_p_SI - L_SI) + "
        f"(Q_p_LI - L_LI)) * C_p_liver - "
        f"(1 - sigma_V_liver) * L_liver * C_p_liver - "
        f"CLup_liver * C_p_liver + "
        f"CLup_liver * FR * C_e_bound_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_liver * ({liver_plasma_eq})")
    liver_plasma_rule.setMath(math_ast)

    # Blood cells equation
    liver_bc_rule = model.createRateRule()
    liver_bc_rule.setVariable("C_bc_liver")
    liver_bc_eq = (
        f"Q_bc_liver * C_bc_lung + "
        f"Q_bc_spleen * C_bc_spleen + "  # Constant for now
        f"Q_bc_pancreas * C_bc_pancreas + "  # Constant for now
        f"Q_bc_SI * C_bc_SI + "  # Constant for now
        f"Q_bc_LI * C_bc_LI - "  # Constant for now
        f"(Q_bc_liver + Q_bc_spleen + Q_bc_pancreas + Q_bc_SI + Q_bc_LI) * C_bc_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_liver * ({liver_bc_eq})")
    liver_bc_rule.setMath(math_ast)

    # ISF equation
    liver_is_rule = model.createRateRule()
    liver_is_rule.setVariable("C_is_liver")
    liver_is_eq = (
        f"(1 - sigma_V_liver) * L_liver * C_p_liver - "
        f"(1 - sigma_L_liver) * L_liver * C_is_liver + "
        f"CLup_liver * (1 - FR) * C_e_bound_liver - "
        f"CLup_liver * C_is_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_liver * ({liver_is_eq})")
    liver_is_rule.setMath(math_ast)

    # Endosomal unbound equation
    liver_e_unbound_rule = model.createRateRule()
    liver_e_unbound_rule.setVariable("C_e_unbound_liver")
    liver_e_unbound_eq = (
        f"CLup_liver * (C_p_liver + C_is_liver) - "
        f"VES_liver * kon_FcRn * C_e_unbound_liver * FcRn_free_liver + "
        f"VES_liver * koff_FcRn * C_e_bound_liver - "
        f"kdeg * C_e_unbound_liver * VES_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_liver * ({liver_e_unbound_eq})")
    liver_e_unbound_rule.setMath(math_ast)

    # Endosomal bound equation
    liver_e_bound_rule = model.createRateRule()
    liver_e_bound_rule.setVariable("C_e_bound_liver")
    liver_e_bound_eq = (
        f"VES_liver * kon_FcRn * C_e_unbound_liver * FcRn_free_liver - "
        f"VES_liver * koff_FcRn * C_e_bound_liver - "
        f"CLup_liver * C_e_bound_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_liver * ({liver_e_bound_eq})")
    liver_e_bound_rule.setMath(math_ast)

    # FcRn free equation
    liver_fcrn_free_rule = model.createRateRule()
    liver_fcrn_free_rule.setVariable("FcRn_free_liver")
    liver_fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_liver * VES_liver - "
        f"kon_FcRn * C_e_unbound_liver * FcRn_free_liver * VES_liver + "
        f"CLup_liver * C_e_bound_liver"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_liver * ({liver_fcrn_free_eq})")
    liver_fcrn_free_rule.setMath(math_ast)

    # Heart plasma equation
    heart_plasma_rule = model.createRateRule()
    heart_plasma_rule.setVariable("C_p_heart")
    heart_plasma_eq = (
        f"Q_p_heart * C_p_lung - "
        f"Q_p_heart * C_p_heart - "
        f"(1 - sigma_V_heart) * L_heart * C_p_heart - "
        f"CLup_heart * C_p_heart + "
        f"CLup_heart * FR * C_e_bound_heart"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_heart * ({heart_plasma_eq})")
    heart_plasma_rule.setMath(math_ast)

    # Heart blood cells equation
    heart_bc_rule = model.createRateRule()
    heart_bc_rule.setVariable("C_bc_heart")
    heart_bc_eq = (
        f"Q_bc_heart * C_bc_lung - Q_bc_heart * C_bc_heart"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_heart * ({heart_bc_eq})")
    heart_bc_rule.setMath(math_ast)

    # Heart endosomal unbound equation
    heart_e_unbound_rule = model.createRateRule()
    heart_e_unbound_rule.setVariable("C_e_unbound_heart")
    heart_e_unbound_eq = (
        f"CLup_heart * (C_p_heart + C_is_heart) - "
        f"VES_heart * kon_FcRn * C_e_unbound_heart * FcRn_free_heart + "
        f"VES_heart * koff_FcRn * C_e_bound_heart - "
        f"kdeg * C_e_unbound_heart * VES_heart"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_heart * ({heart_e_unbound_eq})")
    heart_e_unbound_rule.setMath(math_ast)

    # Heart endosomal bound equation
    heart_e_bound_rule = model.createRateRule()
    heart_e_bound_rule.setVariable("C_e_bound_heart")
    heart_e_bound_eq = (
        f"VES_heart * kon_FcRn * C_e_unbound_heart * FcRn_free_heart - "
        f"VES_heart * koff_FcRn * C_e_bound_heart - "
        f"CLup_heart * C_e_bound_heart"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_heart * ({heart_e_bound_eq})")
    heart_e_bound_rule.setMath(math_ast)

    # Heart ISF equation
    heart_is_rule = model.createRateRule()
    heart_is_rule.setVariable("C_is_heart")
    heart_is_eq = (
        f"(1 - sigma_V_heart) * L_heart * C_p_heart - "
        f"(1 - sigma_L_heart) * L_heart * C_is_heart + "
        f"CLup_heart * (1 - FR) * C_e_bound_heart - "
        f"CLup_heart * C_is_heart"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_heart * ({heart_is_eq})")
    heart_is_rule.setMath(math_ast)

    # Heart FcRn free equation
    heart_fcrn_free_rule = model.createRateRule()
    heart_fcrn_free_rule.setVariable("FcRn_free_heart")
    heart_fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_heart * VES_heart - "
        f"kon_FcRn * C_e_unbound_heart * FcRn_free_heart * VES_heart + "
        f"CLup_heart * C_e_bound_heart"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_heart * ({heart_fcrn_free_eq})")
    heart_fcrn_free_rule.setMath(math_ast)

    # Muscle plasma equation
    muscle_plasma_rule = model.createRateRule()
    muscle_plasma_rule.setVariable("C_p_muscle")
    muscle_plasma_eq = (
        f"Q_p_muscle * C_p_lung - "
        f"Q_p_muscle * C_p_muscle - "
        f"(1 - sigma_V_muscle) * L_muscle * C_p_muscle - "
        f"CLup_muscle * C_p_muscle + "
        f"CLup_muscle * FR * C_e_bound_muscle"
    )
    math_ast = libsbml.parseL3Formula(f"1/Vp_muscle * ({muscle_plasma_eq})")
    muscle_plasma_rule.setMath(math_ast)

    # Muscle blood cells equation
    muscle_bc_rule = model.createRateRule()
    muscle_bc_rule.setVariable("C_bc_muscle")
    muscle_bc_eq = (
        f"Q_bc_muscle * C_bc_lung - Q_bc_muscle * C_bc_muscle"
    )
    math_ast = libsbml.parseL3Formula(f"1/VBC_muscle * ({muscle_bc_eq})")
    muscle_bc_rule.setMath(math_ast)

    # Muscle ISF equation
    muscle_is_rule = model.createRateRule()
    muscle_is_rule.setVariable("C_is_muscle")
    muscle_is_eq = (
        f"(1 - sigma_V_muscle) * L_muscle * C_p_muscle - "
        f"(1 - sigma_L_muscle) * L_muscle * C_is_muscle + "
        f"CLup_muscle * (1 - FR) * C_e_bound_muscle - "
        f"CLup_muscle * C_is_muscle"
    )
    math_ast = libsbml.parseL3Formula(f"1/VIS_muscle * ({muscle_is_eq})")
    muscle_is_rule.setMath(math_ast)

    # Muscle endosomal unbound equation
    muscle_e_unbound_rule = model.createRateRule()
    muscle_e_unbound_rule.setVariable("C_e_unbound_muscle")
    muscle_e_unbound_eq = (
        f"CLup_muscle * (C_p_muscle + C_is_muscle) - "
        f"VES_muscle * kon_FcRn * C_e_unbound_muscle * FcRn_free_muscle + "
        f"VES_muscle * koff_FcRn * C_e_bound_muscle - "
        f"kdeg * C_e_unbound_muscle * VES_muscle"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_muscle * ({muscle_e_unbound_eq})")
    muscle_e_unbound_rule.setMath(math_ast)

    # Muscle endosomal bound equation
    muscle_e_bound_rule = model.createRateRule()
    muscle_e_bound_rule.setVariable("C_e_bound_muscle")
    muscle_e_bound_eq = (
        f"VES_muscle * kon_FcRn * C_e_unbound_muscle * FcRn_free_muscle - "
        f"VES_muscle * koff_FcRn * C_e_bound_muscle - "
        f"CLup_muscle * C_e_bound_muscle"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_muscle * ({muscle_e_bound_eq})")
    muscle_e_bound_rule.setMath(math_ast)

    # Muscle FcRn free equation
    muscle_fcrn_free_rule = model.createRateRule()
    muscle_fcrn_free_rule.setVariable("FcRn_free_muscle")
    muscle_fcrn_free_eq = (
        f"koff_FcRn * C_e_bound_muscle * VES_muscle - "
        f"kon_FcRn * C_e_unbound_muscle * FcRn_free_muscle * VES_muscle + "
        f"CLup_muscle * C_e_bound_muscle"
    )
    math_ast = libsbml.parseL3Formula(f"1/VES_muscle * ({muscle_fcrn_free_eq})")
    muscle_fcrn_free_rule.setMath(math_ast)

    # Add equations for standard organs
    #'''
    for organ in standard_organs:
    
        # Plasma equation
        plasma_rule = model.createRateRule()
        plasma_rule.setVariable(f"C_p_{organ}")
        plasma_eq = (
            f"Q_p_{organ} * C_p_lung - "
            f"Q_p_{organ} * C_p_{organ} - "
            f"(1 - sigma_V_{organ}) * L_{organ} * C_p_{organ} - "
            f"CLup_{organ} * C_p_{organ} + "
            f"CLup_{organ} * FR * C_e_bound_{organ}"
        )
        math_ast = libsbml.parseL3Formula(f"1/Vp_{organ} * ({plasma_eq})")
        plasma_rule.setMath(math_ast)

        # Blood cells equation
        bc_rule = model.createRateRule()
        bc_rule.setVariable(f"C_bc_{organ}")
        bc_eq = f"Q_bc_{organ} * C_bc_lung - Q_bc_{organ} * C_bc_{organ}"
        math_ast = libsbml.parseL3Formula(f"1/VBC_{organ} * ({bc_eq})")
        bc_rule.setMath(math_ast)

        # ISF equation
        is_rule = model.createRateRule()
        is_rule.setVariable(f"C_is_{organ}")
        is_eq = (
            f"(1 - sigma_V_{organ}) * L_{organ} * C_p_{organ} - "
            f"(1 - sigma_L_{organ}) * L_{organ} * C_is_{organ} + "
            f"CLup_{organ} * (1 - FR) * C_e_bound_{organ} - "
            f"CLup_{organ} * C_is_{organ}"
        )
        math_ast = libsbml.parseL3Formula(f"1/VIS_{organ} * ({is_eq})")
        is_rule.setMath(math_ast)

        # Endosomal unbound equation
        e_unbound_rule = model.createRateRule()
        e_unbound_rule.setVariable(f"C_e_unbound_{organ}")
        e_unbound_eq = (
            f"CLup_{organ} * (C_p_{organ} + C_is_{organ}) - "
            f"VES_{organ} * kon_FcRn * C_e_unbound_{organ} * FcRn_free_{organ} + "
            f"VES_{organ} * koff_FcRn * C_e_bound_{organ} - "
            f"kdeg * C_e_unbound_{organ} * VES_{organ}"
        )
        math_ast = libsbml.parseL3Formula(f"1/VES_{organ} * ({e_unbound_eq})")
        e_unbound_rule.setMath(math_ast)

        # Endosomal bound equation
        e_bound_rule = model.createRateRule()
        e_bound_rule.setVariable(f"C_e_bound_{organ}")
        e_bound_eq = (
            f"VES_{organ} * kon_FcRn * C_e_unbound_{organ} * FcRn_free_{organ} - "
            f"VES_{organ} * koff_FcRn * C_e_bound_{organ} - "
            f"CLup_{organ} * C_e_bound_{organ}"
        )
        math_ast = libsbml.parseL3Formula(f"1/VES_{organ} * ({e_bound_eq})")
        e_bound_rule.setMath(math_ast)

        # FcRn free equation
        fcrn_free_rule = model.createRateRule()
        fcrn_free_rule.setVariable(f"FcRn_free_{organ}")
        fcrn_free_eq = (
            f"koff_FcRn * C_e_bound_{organ} * VES_{organ} - "
            f"kon_FcRn * C_e_unbound_{organ} * FcRn_free_{organ} * VES_{organ} + "
            f"CLup_{organ} * C_e_bound_{organ}"
        )
        math_ast = libsbml.parseL3Formula(f"1/VES_{organ} * ({fcrn_free_eq})")
        fcrn_free_rule.setMath(math_ast)
    #'''



    return document

def save_model(document, filename):
    """Save SBML model to file with validation"""
    # Check for errors
    if document.getNumErrors() > 0:
        print("\nValidation errors:")
        document.printErrors()
        return False
        
    # Additional validation
    print("\nValidating SBML model...")
    print(f"Number of compartments: {document.getModel().getNumCompartments()}")
    print(f"Number of species: {document.getModel().getNumSpecies()}")
    print(f"Number of parameters: {document.getModel().getNumParameters()}")
    print(f"Number of rules: {document.getModel().getNumRules()}")
    
    # Save the file
    result = libsbml.writeSBMLToFile(document, filename)
    if result:
        print(f"\nPBPK model saved successfully to {filename}!")
        return True
    else:
        print(f"\nError: Unable to save SBML file to {filename}")
        return False

def main():
    # Load parameters
    params_path = Path("parameters/pbpk_parameters.csv")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found at {params_path}")
    
    params, params_with_units = load_parameters(params_path)
    
    # Create output directory
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pbpk_model.xml"
    
    # Create and save model
    document = create_pbpk_model(params, params_with_units)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"PBPK model saved successfully to {output_path}!")

if __name__ == "__main__":
    main() 