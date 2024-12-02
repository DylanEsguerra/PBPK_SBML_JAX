import pandas as pd
from pathlib import Path

def create_consolidated_params_csv():
    """Create a single CSV file containing all PBPK model parameters"""
    
    # Core volumes
    volumes = {
        'name': [
            # Blood volumes
            'Vp', 'Vbc', 'Vlymphnode',
            # Lung volumes
            'Vp_lung', 'VBC_lung', 'VIS_lung', 'VES_lung',
            # Brain volumes
            'Vp_brain', 'VIS_brain', 'VBBB_brain', 'VBC_brain', 'V_ES_brain',
            # CSF volumes
            'V_BCSFB_brain', 'V_LV_brain', 'V_TFV_brain', 'V_CM_brain', 'V_SAS_brain',
            # Liver volumes
            'Vp_liver', 'VBC_liver', 'VIS_liver', 'VES_liver',
            # Heart volumes
            'Vp_heart', 'VBC_heart', 'VIS_heart', 'VES_heart',
            # Kidney volumes
            'Vp_kidney', 'VBC_kidney', 'VIS_kidney', 'VES_kidney',
            # Muscle volumes
            'Vp_muscle', 'VBC_muscle', 'VIS_muscle', 'VES_muscle',
            # Skin volumes
            'Vp_skin', 'VBC_skin', 'VIS_skin', 'VES_skin',
            # Fat volumes
            'Vp_fat', 'VBC_fat', 'VIS_fat', 'VES_fat',
            # Bone/Marrow volumes
            'Vp_marrow', 'VBC_marrow', 'VIS_marrow', 'VES_marrow',
            # Thymus volumes
            'Vp_thymus', 'VBC_thymus', 'VIS_thymus', 'VES_thymus',
            # GI volumes
            'Vp_SI', 'VBC_SI', 'VIS_SI', 'VES_SI',
            'Vp_LI', 'VBC_LI', 'VIS_LI', 'VES_LI',
            # Other organ volumes
            'Vp_spleen', 'VBC_spleen', 'VIS_spleen', 'VES_spleen',
            'Vp_pancreas', 'VBC_pancreas', 'VIS_pancreas', 'VES_pancreas',
            'Vp_other', 'VBC_other', 'VIS_other', 'VES_other'
        ],
        'value': [
            # Blood volumes
            3126, 2558, 274,
            # Lung volumes
            1000, 55.0, 45.0, 5.00,
            # Brain volumes
            31.9, 261.0, 0.1, 26.1, 7.25,
            # CSF volumes
            0.1, 22.5, 22.5, 7.5, 90.0,
            # Liver volumes
            2143, 183, 149, 10.7,
            # Heart volumes (from table)
            341, 13.1, 10.8, 1.71,
            # Kidney volumes (from table)
            332, 18.2, 14.9, 1.66,
            # Muscle volumes (from table)
            30078, 662, 541, 150,
            # Skin volumes (from table)
            3408, 127, 104, 17.0,
            # Fat/Adipose volumes (from table)
            13465, 148, 121, 67.3,
            # Bone/Marrow volumes (from table)
            10165, 224, 183, 50.8,
            # Thymus volumes (from table)
            6.41, 0.353, 0.288, 0.0321,
            # GI volumes (from table)
            385, 6.15, 5.03, 1.93,  # SI
            548, 8.74, 7.15, 2.74,  # LI
            # Other organ volumes (from table)
            221, 26.8, 21.9, 1.11,  # Spleen
            104, 5.70, 4.66, 0.518,  # Pancreas
            4852, 204, 167, 24.3     # Other
        ],
        'units': ['mL'] * 69,  # Updated count
        'description': [
            # Blood volumes
            'Total plasma volume', 'Total blood cell volume', 'Lymph node volume',
            # Lung volumes
            'Lung plasma volume', 'Lung blood cell volume', 'Lung ISF volume', 'Lung endosomal volume',
            # Brain volumes
            'Brain plasma volume', 'Brain ISF volume', 'BBB endosomal volume', 'Brain blood cell volume', 'Brain endosomal volume',
            # CSF volumes
            'BCSFB endosomal volume', 'Lateral ventricle volume', 'Third/Fourth ventricle volume',
            'Cisterna magna volume', 'Subarachnoid space volume',
            # Liver volumes
            'Liver plasma volume', 'Liver blood cell volume', 'Liver ISF volume', 'Liver endosomal volume',
            # Heart volumes
            'Heart plasma volume', 'Heart blood cell volume', 'Heart ISF volume', 'Heart endosomal volume',
            # Kidney volumes
            'Kidney plasma volume', 'Kidney blood cell volume', 'Kidney ISF volume', 'Kidney endosomal volume',
            # Muscle volumes
            'Muscle plasma volume', 'Muscle blood cell volume', 'Muscle ISF volume', 'Muscle endosomal volume',
            # Skin volumes
            'Skin plasma volume', 'Skin blood cell volume', 'Skin ISF volume', 'Skin endosomal volume',
            # Fat volumes
            'Fat plasma volume', 'Fat blood cell volume', 'Fat ISF volume', 'Fat endosomal volume',
            # Bone/Marrow volumes
            'Bone marrow plasma volume', 'Bone marrow blood cell volume', 'Bone marrow ISF volume', 'Bone marrow endosomal volume',
            # Thymus volumes
            'Thymus plasma volume', 'Thymus blood cell volume', 'Thymus ISF volume', 'Thymus endosomal volume',
            # GI volumes
            'Small intestine plasma volume', 'Small intestine blood cell volume', 'Small intestine ISF volume', 'Small intestine endosomal volume',
            'Large intestine plasma volume', 'Large intestine blood cell volume', 'Large intestine ISF volume', 'Large intestine endosomal volume',
            # Other organ volumes
            'Spleen plasma volume', 'Spleen blood cell volume', 'Spleen ISF volume', 'Spleen endosomal volume',
            'Pancreas plasma volume', 'Pancreas blood cell volume', 'Pancreas ISF volume', 'Pancreas endosomal volume',
            'Other plasma volume', 'Other blood cell volume', 'Other ISF volume', 'Other endosomal volume'
        ]
    }

    # Let's verify the lengths match
    print(f"Lengths of arrays in volumes dictionary:")
    print(f"name: {len(volumes['name'])}")
    print(f"value: {len(volumes['value'])}")
    print(f"units: {len(volumes['units'])}")
    print(f"description: {len(volumes['description'])}")

    # Flow rates
    flows = {
        'name': [
            # Plasma flows
            'Q_p_lung', 'Q_p_brain', 'Q_p_heart', 'Q_p_liver', 'Q_p_kidney',
            'Q_p_muscle', 'Q_p_skin', 'Q_p_fat', 'Q_p_marrow', 'Q_p_thymus',
            'Q_p_SI', 'Q_p_LI', 'Q_p_spleen', 'Q_p_pancreas', 'Q_p_other',
            
            # Blood cell flows
            'Q_bc_lung', 'Q_bc_brain', 'Q_bc_heart', 'Q_bc_liver', 'Q_bc_kidney',
            'Q_bc_muscle', 'Q_bc_skin', 'Q_bc_fat', 'Q_bc_marrow', 'Q_bc_thymus',
            'Q_bc_SI', 'Q_bc_LI', 'Q_bc_spleen', 'Q_bc_pancreas', 'Q_bc_other',
            
            # CSF and ISF flows
            'Q_CSF_brain', 'Q_ISF_brain', 'Q_ECF_brain',
            
            # Lymph flows
            'L_lung', 'L_brain', 'L_heart', 'L_liver', 'L_kidney',
            'L_muscle', 'L_skin', 'L_fat', 'L_marrow', 'L_thymus',
            'L_SI', 'L_LI', 'L_spleen', 'L_pancreas', 'L_other', 'L_LN'
        ],
        'value': [
            # Plasma flows (from table, mL/h)
            181913, 36402, 7752, 13210, 32402,
            33469, 11626, 8343, 2591, 353,
            12368, 12867, 6343, 3056, 5521,
            
            # Blood cell flows (calculated as plasma flow * (Vbc/Vp))
            148920, 29810, 6350, 10820, 26530,
            27410, 9520, 6830, 2120, 289,
            10130, 10530, 5190, 2500, 4520,
            
            # CSF and ISF flows
            21.0, 10.5, 10.5,  # table 2
            
            # Lymph flows (typically 0.2% of plasma flow)
            364, 73, 16, 26, 65,
            67, 23, 17, 5, 1,
            25, 26, 13, 6, 11, 500  # L_LN value may need adjustment
        ],
        'units': ['mL/h'] * 49,
        'description': [
            # Plasma flows
            'Lung plasma flow rate', 'Brain plasma flow rate', 'Heart plasma flow rate',
            'Liver plasma flow rate', 'Kidney plasma flow rate', 'Muscle plasma flow rate',
            'Skin plasma flow rate', 'Fat plasma flow rate', 'Bone marrow plasma flow rate',
            'Thymus plasma flow rate', 'Small intestine plasma flow rate',
            'Large intestine plasma flow rate', 'Spleen plasma flow rate',
            'Pancreas plasma flow rate', 'Other organs plasma flow rate',
            
            # Blood cell flows
            'Lung blood cell flow rate', 'Brain blood cell flow rate',
            'Heart blood cell flow rate', 'Liver blood cell flow rate',
            'Kidney blood cell flow rate', 'Muscle blood cell flow rate',
            'Skin blood cell flow rate', 'Fat blood cell flow rate',
            'Bone marrow blood cell flow rate', 'Thymus blood cell flow rate',
            'Small intestine blood cell flow rate', 'Large intestine blood cell flow rate',
            'Spleen blood cell flow rate', 'Pancreas blood cell flow rate',
            'Other organs blood cell flow rate',
            
            # CSF and ISF flows
            'CSF flow rate', 'Brain ISF flow rate', 'Cerebral ISF production rate',
            
            # Lymph flows
            'Lung lymph flow rate', 'Brain lymph flow rate', 'Heart lymph flow rate',
            'Liver lymph flow rate', 'Kidney lymph flow rate', 'Muscle lymph flow rate',
            'Skin lymph flow rate', 'Fat lymph flow rate', 'Bone marrow lymph flow rate',
            'Thymus lymph flow rate', 'Small intestine lymph flow rate',
            'Large intestine lymph flow rate', 'Spleen lymph flow rate',
            'Pancreas lymph flow rate', 'Other organs lymph flow rate',
            'Lymph node flow rate'
        ]
    }

    # Let's verify the lengths match
    print(f"\nLengths of arrays in flows dictionary:")
    print(f"name: {len(flows['name'])}")
    print(f"value: {len(flows['value'])}")
    print(f"units: {len(flows['units'])}")
    print(f"description: {len(flows['description'])}")

    # Kinetic parameters
    kinetics = {
        'name': [
            # FcRn kinetics
            'kon_FcRn', 'koff_FcRn', 'kdeg',
            
            # Clearance parameters for each organ
            'CLup_lung', 'CLup_brain', 'CLup_heart', 'CLup_liver', 'CLup_kidney',
            'CLup_muscle', 'CLup_skin', 'CLup_fat', 'CLup_marrow', 'CLup_thymus',
            'CLup_SI', 'CLup_LI', 'CLup_spleen', 'CLup_pancreas', 'CLup_other',
            
            # Reflection coefficients (vascular)
            'sigma_V_lung', 'sigma_V_brain', 'sigma_V_heart', 'sigma_V_liver', 'sigma_V_kidney',
            'sigma_V_muscle', 'sigma_V_skin', 'sigma_V_fat', 'sigma_V_marrow', 'sigma_V_thymus',
            'sigma_V_SI', 'sigma_V_LI', 'sigma_V_spleen', 'sigma_V_pancreas', 'sigma_V_other', 
            'sigma_V_BCSFB', 'sigma_V_BBB',
            
            # Reflection coefficients (lymphatic)
            'sigma_L_lung', 'sigma_L_brain', 'sigma_L_heart', 'sigma_L_liver', 'sigma_L_kidney',
            'sigma_L_muscle', 'sigma_L_skin', 'sigma_L_fat', 'sigma_L_marrow', 'sigma_L_thymus',
            'sigma_L_SI', 'sigma_L_LI', 'sigma_L_spleen', 'sigma_L_pancreas', 'sigma_L_brain_ISF',
            'sigma_L_SAS', 'sigma_L_other',
            
            # Other parameters
            'FR', 'f_BBB', 'f_LV', 'f_BCSFB', 'FcRn_free_BBB', 'FcRn_free_BCSFB'
        ],
        'value': [
            # FcRn kinetics
            5.59e8, 23.9, 26.6,  # kon_FcRn and koff_FcRn from table (human values)
            
            # Clearance parameters 
            0.55, 0.3, 0.55, 0.55, 0.55,  # CLup = 0.55 L/h/L from sup table 6, CLup_brain from table 3
            0.55, 0.55, 0.55, 0.55, 0.55,
            0.55, 0.55, 0.55, 0.55, 0.55,
            
            # Vascular reflection coefficients
            0.95, 0.95, 0.95, 0.95, 0.90,  # 0.95 for lung/heart/muscle/skin/adipose/LI/other
            0.95, 0.95, 0.95, 0.95, 0.90,  # 0.90 for kidney/thymus/SI/pancreas
            0.90, 0.95, 0.85, 0.90, 0.95,  # 0.85 for spleen/liver/bone
            0.9974, 0.95,
            
            # Lymphatic reflection coefficients
            0.2, 0.2, 0.2, 0.2, 0.2,  # σL = 0.2 from table for all tissues
            0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2,
                
            # Other parameters
            0.715, 0.95, 0.2, 0.2, 1.0, 1.0  # FR = 0.715 from table
        ],
        'units': ['1/M/h'] + ['1/h'] * 2 + ['L/h/L'] * 15 + ['dimensionless'] * 40,
        'description': [
            # FcRn kinetics
            'FcRn binding rate constant', 'FcRn unbinding rate constant',
            'Degradation rate constant',
            
            # Clearance parameters
            'Lung uptake clearance', 'Brain uptake clearance', 'Heart uptake clearance',
            'Liver uptake clearance', 'Kidney uptake clearance', 'Muscle uptake clearance',
            'Skin uptake clearance', 'Fat uptake clearance', 'Bone marrow uptake clearance',
            'Thymus uptake clearance', 'Small intestine uptake clearance',
            'Large intestine uptake clearance', 'Spleen uptake clearance',
            'Pancreas uptake clearance', 'Other organs uptake clearance',
            
            # Vascular reflection coefficients
            'Lung vascular reflection coefficient', 'Brain vascular reflection coefficient',
            'Heart vascular reflection coefficient', 'Liver vascular reflection coefficient',
            'Kidney vascular reflection coefficient', 'Muscle vascular reflection coefficient',
            'Skin vascular reflection coefficient', 'Fat vascular reflection coefficient',
            'Bone marrow vascular reflection coefficient', 'Thymus vascular reflection coefficient',
            'Small intestine vascular reflection coefficient',
            'Large intestine vascular reflection coefficient',
            'Spleen vascular reflection coefficient', 'Pancreas vascular reflection coefficient',
            'Other organs vascular reflection coefficient', 'BCSFB vascular reflection coefficient',
            'BBB vascular reflection coefficient',
            
            # Lymphatic reflection coefficients
            'Lung lymphatic reflection coefficient', 'Brain lymphatic reflection coefficient',
            'Heart lymphatic reflection coefficient', 'Liver lymphatic reflection coefficient',
            'Kidney lymphatic reflection coefficient', 'Muscle lymphatic reflection coefficient',
            'Skin lymphatic reflection coefficient', 'Fat lymphatic reflection coefficient',
            'Bone marrow lymphatic reflection coefficient',
            'Thymus lymphatic reflection coefficient',
            'Small intestine lymphatic reflection coefficient',
            'Large intestine lymphatic reflection coefficient',
            'Spleen lymphatic reflection coefficient',
            'Pancreas lymphatic reflection coefficient',
            'Brain ISF lymphatic reflection coefficient',
            'Subarachnoid space lymphatic reflection coefficient',
            'Other organs lymphatic reflection coefficient',
            
            # Other parameters
            'Recycling fraction', 'BBB permeability', 
            'Lateral ventricle flow fraction',  # Added description
            'BCSFB permiability',
            'Initial BBB FcRn free concentration',  # Added description
            'Initial BCSFB FcRn free concentration'  # Added description
        ]
    }

    # Let's verify the lengths match
    print(f"\nLengths of arrays in kinetics dictionary:")
    print(f"name: {len(kinetics['name'])}")
    print(f"value: {len(kinetics['value'])}")
    print(f"units: {len(kinetics['units'])}")
    print(f"description: {len(kinetics['description'])}")

    # Initial concentrations
    concentrations = {
        'name': [
            # Blood and lymph (3)
            'C_p_0', 'C_bc_0', 'C_ln_0',
            
            # Plasma concentrations (15)
            'C_p_lung_0', 'C_p_brain_0', 'C_p_heart_0', 'C_p_liver_0', 'C_p_kidney_0',
            'C_p_muscle_0', 'C_p_skin_0', 'C_p_fat_0', 'C_p_marrow_0', 'C_p_thymus_0',
            'C_p_SI_0', 'C_p_LI_0', 'C_p_spleen_0', 'C_p_pancreas_0', 'C_p_other_0',
            
            # Blood cell concentrations (15)
            'C_bc_lung_0', 'C_bc_brain_0', 'C_bc_heart_0', 'C_bc_liver_0', 'C_bc_kidney_0',
            'C_bc_muscle_0', 'C_bc_skin_0', 'C_bc_fat_0', 'C_bc_marrow_0', 'C_bc_thymus_0',
            'C_bc_SI_0', 'C_bc_LI_0', 'C_bc_spleen_0', 'C_bc_pancreas_0', 'C_bc_other_0',
            
            # ISF concentrations (15)
            'C_is_lung_0', 'C_is_brain_0', 'C_is_heart_0', 'C_is_liver_0', 'C_is_kidney_0',
            'C_is_muscle_0', 'C_is_skin_0', 'C_is_fat_0', 'C_is_marrow_0', 'C_is_thymus_0',
            'C_is_SI_0', 'C_is_LI_0', 'C_is_spleen_0', 'C_is_pancreas_0', 'C_is_other_0',
            
            # Endosomal concentrations (42: 3 per organ * 14 organs)
            'C_e_unbound_lung_0', 'C_e_bound_lung_0', 'FcRn_free_lung_0',
            'C_e_unbound_heart_0', 'C_e_bound_heart_0', 'FcRn_free_heart_0',
            'C_e_unbound_liver_0', 'C_e_bound_liver_0', 'FcRn_free_liver_0',
            'C_e_unbound_kidney_0', 'C_e_bound_kidney_0', 'FcRn_free_kidney_0',
            'C_e_unbound_muscle_0', 'C_e_bound_muscle_0', 'FcRn_free_muscle_0',
            'C_e_unbound_skin_0', 'C_e_bound_skin_0', 'FcRn_free_skin_0',
            'C_e_unbound_fat_0', 'C_e_bound_fat_0', 'FcRn_free_fat_0',
            'C_e_unbound_marrow_0', 'C_e_bound_marrow_0', 'FcRn_free_marrow_0',
            'C_e_unbound_thymus_0', 'C_e_bound_thymus_0', 'FcRn_free_thymus_0',
            'C_e_unbound_SI_0', 'C_e_bound_SI_0', 'FcRn_free_SI_0',
            'C_e_unbound_LI_0', 'C_e_bound_LI_0', 'FcRn_free_LI_0',
            'C_e_unbound_spleen_0', 'C_e_bound_spleen_0', 'FcRn_free_spleen_0',
            'C_e_unbound_pancreas_0', 'C_e_bound_pancreas_0', 'FcRn_free_pancreas_0',
            'C_e_unbound_other_0', 'C_e_bound_other_0', 'FcRn_free_other_0',
            
            # BBB concentrations (2)
            'C_BBB_unbound_brain_0', 'C_BBB_bound_brain_0',
            
            # CSF concentrations (6)
            'C_BCSFB_unbound_brain_0', 'C_BCSFB_bound_brain_0',
            'C_LV_brain_0', 'C_TFV_brain_0', 'C_CM_brain_0', 'C_SAS_brain_0'
        ],
        'value': [
            # Blood and lymph (3)
            14118/3126, 0.0, 0.0,   # Original value was: 14118/3126 (36 mg/kg for 60kg person, Igg1 153 Kda 153000 g/mol, divided by Vp)
            
            # Plasma concentrations (15)
            # Set C_p_lung_0 to original plasma value, rest to 0
            0.0,       # lung - using original plasma concentration value
            0.0,              # brain
            *[0.0] * 13,      # remaining organs
            
            # Blood cell concentrations (15)
            *[0.0] * 15,
            
            # ISF concentrations (15)
            *[0.0] * 15,
            
            # Endosomal concentrations (42: 3 per organ * 14 organs)
            # For each organ: [C_e_unbound = 0.0, C_e_bound = 0.0, FcRn_free = 4.982e-5]
            *[val for _ in range(14) for val in [0.0, 0.0, 4.982e-5]],
            
            # BBB concentrations (2)
            *[0.0] * 2,
            
            # CSF concentrations (6)
            *[0.0] * 6
        ],
        'units': ['nM/mL'] * 98,
        'description': [
            # Blood and lymph (3)
            'Initial plasma concentration (C_p_0)', 
            'Initial blood cell concentration (C_bc_0)', 
            'Initial lymph node concentration (C_ln_0)',
            
            # Plasma concentrations (15)
            'Initial lung plasma concentration (C_p_lung_0)',
            'Initial brain plasma concentration (C_p_brain_0)',
            'Initial heart plasma concentration (C_p_heart_0)',
            'Initial liver plasma concentration (C_p_liver_0)',
            'Initial kidney plasma concentration (C_p_kidney_0)',
            'Initial muscle plasma concentration (C_p_muscle_0)',
            'Initial skin plasma concentration (C_p_skin_0)',
            'Initial fat plasma concentration (C_p_fat_0)',
            'Initial bone marrow plasma concentration (C_p_marrow_0)',
            'Initial thymus plasma concentration (C_p_thymus_0)',
            'Initial SI plasma concentration (C_p_SI_0)',
            'Initial LI plasma concentration (C_p_LI_0)',
            'Initial spleen plasma concentration (C_p_spleen_0)',
            'Initial pancreas plasma concentration (C_p_pancreas_0)',
            'Initial other plasma concentration (C_p_other_0)',
            
            # Blood cell concentrations (15)
            'Initial lung blood cell concentration (C_bc_lung_0)',
            'Initial brain blood cell concentration (C_bc_brain_0)',
            'Initial heart blood cell concentration (C_bc_heart_0)',
            'Initial liver blood cell concentration (C_bc_liver_0)',
            'Initial kidney blood cell concentration (C_bc_kidney_0)',
            'Initial muscle blood cell concentration (C_bc_muscle_0)',
            'Initial skin blood cell concentration (C_bc_skin_0)',
            'Initial fat blood cell concentration (C_bc_fat_0)',
            'Initial bone marrow blood cell concentration (C_bc_marrow_0)',
            'Initial thymus blood cell concentration (C_bc_thymus_0)',
            'Initial SI blood cell concentration (C_bc_SI_0)',
            'Initial LI blood cell concentration (C_bc_LI_0)',
            'Initial spleen blood cell concentration (C_bc_spleen_0)',
            'Initial pancreas blood cell concentration (C_bc_pancreas_0)',
            'Initial other blood cell concentration (C_bc_other_0)',
            
            # ISF concentrations (15)
            'Initial lung ISF concentration (C_is_lung_0)',
            'Initial brain ISF concentration (C_is_brain_0)',
            'Initial heart ISF concentration (C_is_heart_0)',
            'Initial liver ISF concentration (C_is_liver_0)',
            'Initial kidney ISF concentration (C_is_kidney_0)',
            'Initial muscle ISF concentration (C_is_muscle_0)',
            'Initial skin ISF concentration (C_is_skin_0)',
            'Initial fat ISF concentration (C_is_fat_0)',
            'Initial bone marrow ISF concentration (C_is_marrow_0)',
            'Initial thymus ISF concentration (C_is_thymus_0)',
            'Initial SI ISF concentration (C_is_SI_0)',
            'Initial LI ISF concentration (C_is_LI_0)',
            'Initial spleen ISF concentration (C_is_spleen_0)',
            'Initial pancreas ISF concentration (C_is_pancreas_0)',
            'Initial other ISF concentration (C_is_other_0)',
            
            # Endosomal concentrations (42: 3 per organ * 14 organs)
            *[desc for organ in ['lung', 'heart', 'liver', 'kidney', 
                               'muscle', 'skin', 'fat', 'marrow', 'thymus',
                               'SI', 'LI', 'spleen', 'pancreas', 'other']
              for desc in [
                  f'Initial {organ} endosomal unbound concentration (C_e_unbound_{organ}_0)',
                  f'Initial {organ} endosomal bound concentration (C_e_bound_{organ}_0)',
                  f'Initial {organ} FcRn free concentration (FcRn_free_{organ}_0)'
              ]],
            
            # BBB concentrations (2)
            'Initial BBB unbound concentration (C_BBB_unbound_brain_0)',
            'Initial BBB bound concentration (C_BBB_bound_brain_0)',
            
            # CSF concentrations (6)
            'Initial BCSFB unbound concentration (C_BCSFB_unbound_brain_0)',
            'Initial BCSFB bound concentration (C_BCSFB_bound_brain_0)',
            'Initial lateral ventricle concentration (C_LV_brain_0)',
            'Initial third/fourth ventricle concentration (C_TFV_brain_0)',
            'Initial cisterna magna concentration (C_CM_brain_0)',
            'Initial subarachnoid space concentration (C_SAS_brain_0)'
        ]
    }

    # Let's count the items in each section of 'value' array:
    value_counts = [
        3,   # Blood and lymph
        15,  # Plasma concentrations
        15,  # Blood cell concentrations
        15,  # ISF concentrations
        42,  # Endosomal concentrations (3 per organ * 14 organs)
        2,   # BBB concentrations (brain: unbound, bound)
        6    # CSF concentrations
    ]
    total_expected = sum(value_counts)
    print(f"Expected total values: {total_expected}")  # Should be 98
    
    # Verify each section length matches comments
    blood_lymph = concentrations['value'][:3]
    plasma = concentrations['value'][3:18]
    blood_cell = concentrations['value'][18:33]
    isf = concentrations['value'][33:48]
    endosomal = concentrations['value'][48:90]  # 14 organs * 3 params
    bbb = concentrations['value'][90:92]        # Brain BBB: 2 params
    csf = concentrations['value'][92:]          # CSF: 6 params
    
    print(f"\nActual section lengths:")
    print(f"Blood and lymph: {len(blood_lymph)} (expected 3)")
    print(f"Plasma: {len(plasma)} (expected 15)")
    print(f"Blood cell: {len(blood_cell)} (expected 15)")
    print(f"ISF: {len(isf)} (expected 15)")
    print(f"Endosomal: {len(endosomal)} (expected 42)")
    print(f"BBB: {len(bbb)} (expected 2)")
    print(f"CSF: {len(csf)} (expected 6)")

    # Update descriptions to handle brain separately
    concentrations['description'] = [
        # Blood and lymph (3)
        'Initial plasma concentration (C_p_0)', 
        'Initial blood cell concentration (C_bc_0)', 
        'Initial lymph node concentration (C_ln_0)',
        
        # Plasma concentrations (15)
        'Initial lung plasma concentration (C_p_lung_0)',
        'Initial brain plasma concentration (C_p_brain_0)',
        'Initial heart plasma concentration (C_p_heart_0)',
        'Initial liver plasma concentration (C_p_liver_0)',
        'Initial kidney plasma concentration (C_p_kidney_0)',
        'Initial muscle plasma concentration (C_p_muscle_0)',
        'Initial skin plasma concentration (C_p_skin_0)',
        'Initial fat plasma concentration (C_p_fat_0)',
        'Initial bone marrow plasma concentration (C_p_marrow_0)',
        'Initial thymus plasma concentration (C_p_thymus_0)',
        'Initial SI plasma concentration (C_p_SI_0)',
        'Initial LI plasma concentration (C_p_LI_0)',
        'Initial spleen plasma concentration (C_p_spleen_0)',
        'Initial pancreas plasma concentration (C_p_pancreas_0)',
        'Initial other plasma concentration (C_p_other_0)',
        
        # Blood cell concentrations (15)
        'Initial lung blood cell concentration (C_bc_lung_0)',
        'Initial brain blood cell concentration (C_bc_brain_0)',
        'Initial heart blood cell concentration (C_bc_heart_0)',
        'Initial liver blood cell concentration (C_bc_liver_0)',
        'Initial kidney blood cell concentration (C_bc_kidney_0)',
        'Initial muscle blood cell concentration (C_bc_muscle_0)',
        'Initial skin blood cell concentration (C_bc_skin_0)',
        'Initial fat blood cell concentration (C_bc_fat_0)',
        'Initial bone marrow blood cell concentration (C_bc_marrow_0)',
        'Initial thymus blood cell concentration (C_bc_thymus_0)',
        'Initial SI blood cell concentration (C_bc_SI_0)',
        'Initial LI blood cell concentration (C_bc_LI_0)',
        'Initial spleen blood cell concentration (C_bc_spleen_0)',
        'Initial pancreas blood cell concentration (C_bc_pancreas_0)',
        'Initial other blood cell concentration (C_bc_other_0)',
        
        # ISF concentrations (15)
        'Initial lung ISF concentration (C_is_lung_0)',
        'Initial brain ISF concentration (C_is_brain_0)',
        'Initial heart ISF concentration (C_is_heart_0)',
        'Initial liver ISF concentration (C_is_liver_0)',
        'Initial kidney ISF concentration (C_is_kidney_0)',
        'Initial muscle ISF concentration (C_is_muscle_0)',
        'Initial skin ISF concentration (C_is_skin_0)',
        'Initial fat ISF concentration (C_is_fat_0)',
        'Initial bone marrow ISF concentration (C_is_marrow_0)',
        'Initial thymus ISF concentration (C_is_thymus_0)',
        'Initial SI ISF concentration (C_is_SI_0)',
        'Initial LI ISF concentration (C_is_LI_0)',
        'Initial spleen ISF concentration (C_is_spleen_0)',
        'Initial pancreas ISF concentration (C_is_pancreas_0)',
        'Initial other ISF concentration (C_is_other_0)',
        
        # Endosomal concentrations (42: 3 per organ * 14 organs)
        *[desc for organ in ['lung', 'heart', 'liver', 'kidney', 
                           'muscle', 'skin', 'fat', 'marrow', 'thymus',
                           'SI', 'LI', 'spleen', 'pancreas', 'other']
          for desc in [
              f'Initial {organ} endosomal unbound concentration (C_e_unbound_{organ}_0)',
              f'Initial {organ} endosomal bound concentration (C_e_bound_{organ}_0)',
              f'Initial {organ} FcRn free concentration (FcRn_free_{organ}_0)'
          ]],
        
        # BBB concentrations (2)
        'Initial BBB unbound concentration (C_BBB_unbound_brain_0)',
        'Initial BBB bound concentration (C_BBB_bound_brain_0)',
        
        # CSF concentrations (6)
        'Initial BCSFB unbound concentration (C_BCSFB_unbound_brain_0)',
        'Initial BCSFB bound concentration (C_BCSFB_bound_brain_0)',
        'Initial lateral ventricle concentration (C_LV_brain_0)',
        'Initial third/fourth ventricle concentration (C_TFV_brain_0)',
        'Initial cisterna magna concentration (C_CM_brain_0)',
        'Initial subarachnoid space concentration (C_SAS_brain_0)'
    ]

    # Verify all arrays have the same length
    print("\nVerifying array lengths:")
    print(f"name: {len(concentrations['name'])} (expected {total_expected})")
    print(f"value: {len(concentrations['value'])} (expected {total_expected})")
    print(f"units: {len(concentrations['units'])} (expected {total_expected})")
    print(f"description: {len(concentrations['description'])} (expected {total_expected})")

    # Combine all parameters into one DataFrame
    all_params = pd.concat([
        pd.DataFrame(volumes),
        pd.DataFrame(flows),
        pd.DataFrame(kinetics),
        pd.DataFrame(concentrations)
    ]).reset_index(drop=True)

    # Save to CSV
    output_dir = Path('parameters')
    output_dir.mkdir(exist_ok=True)
    all_params.to_csv(output_dir / 'pbpk_parameters.csv', index=False)

def main():
    print("Creating consolidated PBPK parameters CSV file...")
    create_consolidated_params_csv()
    print("Parameters file created successfully!")

if __name__ == "__main__":
    main()