# create_parameter_sheets.py
import pandas as pd
from pathlib import Path

def create_blood_params_excel():
    # Core volumes
    volumes = {
        'name': [
            'Vp', 'Vbc', 'Vlymphnode'
        ],
        'value': [
            3126, 2558, 274
        ],
        'units': ['mL'] * 3,
        'description': [
            'Plasma volume',
            'Blood cell volume',
            'Lymph node volume'
        ],
        'source': ['Supplementary Table 5'] * 3
    }

    # Lymph flows
    lymph_flows = {
        'name': [
            'L_lung', 'L_heart', 'L_kidney', 'L_brain', 'L_muscle',
            'L_marrow', 'L_thymus', 'L_skin', 'L_fat', 'L_SI',
            'L_LI', 'L_spleen', 'L_pancreas', 'L_liver', 'L_other',
            'L_LN'
        ],
        'value': [
            183, 78, 364, 215, 335,
            26, 4, 116, 112, 124,
            129, 63, 31, 132, 55,
            2000  # Lymph node outflow
        ],
        'units': ['mL/h'] * 16,
        'description': [f'{organ} lymph flow' for organ in [
            'Lung', 'Heart', 'Kidney', 'Brain', 'Muscle',
            'Bone marrow', 'Thymus', 'Skin', 'Fat', 'Small intestine',
            'Large intestine', 'Spleen', 'Pancreas', 'Liver', 'Other',
            'Lymph node'
        ]],
        'source': ['Supplementary Table 5'] * 16
    }

    # Reflection coefficients
    reflection_coeffs = {
        'name': [
            'sigma_L_lung', 'sigma_L_heart', 'sigma_L_kidney',
            'sigma_L_brain_ISF', 'sigma_L_muscle', 'sigma_L_marrow',
            'sigma_L_thymus', 'sigma_L_skin', 'sigma_L_fat',
            'sigma_L_SI', 'sigma_L_LI', 'sigma_L_spleen',
            'sigma_L_pancreas', 'sigma_L_liver', 'sigma_L_other',
            'sigma_L_SAS'
        ],
        'value': [0.2] * 16,  # Default value for all coefficients
        'units': ['dimensionless'] * 16,
        'description': [f'{organ} lymphatic reflection coefficient' for organ in [
            'Lung', 'Heart', 'Kidney', 'Brain ISF', 'Muscle',
            'Bone marrow', 'Thymus', 'Skin', 'Fat', 'Small intestine',
            'Large intestine', 'Spleen', 'Pancreas', 'Liver', 'Other',
            'Subarachnoid space'
        ]],
        'source': ['Literature value'] * 16
    }

    # Plasma flows
    plasma_flows = {
        'name': [
            'Q_p_heart', 'Q_p_lung', 'Q_p_muscle', 'Q_p_skin', 'Q_p_fat',
            'Q_p_marrow', 'Q_p_kidney', 'Q_p_liver', 'Q_p_SI', 'Q_p_LI',
            'Q_p_pancreas', 'Q_p_thymus', 'Q_p_spleen', 'Q_p_other', 'Q_p_brain',
            'Q_CSF_brain', 'Q_ECF_brain'
        ],
        'value': [
            7752, 181913, 33469, 11626, 11233,
            2591, 36402, 13210, 12368, 12867,
            3056, 353, 6343, 5521, 21453.0,
            24.0, 10.5
        ],
        'units': ['mL/h'] * 17,
        'description': [f'{organ} plasma flow' for organ in [
            'Heart', 'Lung', 'Muscle', 'Skin', 'Fat',
            'Bone marrow', 'Kidney', 'Liver', 'Small intestine', 'Large intestine',
            'Pancreas', 'Thymus', 'Spleen', 'Other', 'Brain',
            'CSF production', 'Cerebral ISF production'
        ]],
        'source': ['Supplementary Table 5'] * 15 + ['Brain parameter'] * 2
    }

    # Blood cell flows
    blood_cell_flows = {
        'name': [
            'Q_bc_heart', 'Q_bc_lung', 'Q_bc_muscle', 'Q_bc_skin', 'Q_bc_fat',
            'Q_bc_marrow', 'Q_bc_kidney', 'Q_bc_liver', 'Q_bc_SI', 'Q_bc_LI',
            'Q_bc_pancreas', 'Q_bc_thymus', 'Q_bc_spleen', 'Q_bc_other', 'Q_bc_brain'
        ],
        'value': [
            6342, 148838, 27383, 9512, 9191,
            2120, 29784, 10808, 10120, 10527,
            2500, 289, 5189, 4517, 17553.0
        ],
        'units': ['mL/h'] * 15,
        'description': [f'{organ} blood cell flow' for organ in [
            'Heart', 'Lung', 'Muscle', 'Skin', 'Fat',
            'Bone marrow', 'Kidney', 'Liver', 'Small intestine', 'Large intestine',
            'Pancreas', 'Thymus', 'Spleen', 'Other', 'Brain'
        ]],
        'source': ['Supplementary Table 5'] * 15
    }

    # Initial concentrations for blood and plasma compartments
    plasma_concentrations = {
        'name': [
            'C_p_0', 'C_bc_0', 'C_ln_0',  # Primary blood species
            # Coupling parameters start at 0.0
            'C_p_heart_0', 'C_p_kidney_0',
            'C_p_brain_0', 'C_p_muscle_0', 'C_p_marrow_0',
            'C_p_thymus_0', 'C_p_skin_0', 'C_p_fat_0',
            'C_p_liver_0', 'C_p_other_0',
            'C_is_lung_0', 'C_is_heart_0', 'C_is_kidney_0',
            'C_is_brain_0', 'C_is_muscle_0', 'C_is_marrow_0',
            'C_is_thymus_0', 'C_is_skin_0', 'C_is_fat_0',
            'C_is_SI_0', 'C_is_LI_0', 'C_is_spleen_0',
            'C_is_pancreas_0', 'C_is_liver_0', 'C_is_other_0',
            'C_SAS_brain_0'
        ],
        'value': [14118/3126, 0.0, 0.0] + [0.0] * 26,  # 36 mg/kg for 60kg person, Igg1 153 Kda 153000 g/mol, devided by Vp
        'units': ['nM/mL'] * 29,
        'description': [
            'Initial plasma concentration',
            'Initial blood cell concentration',
            'Initial lymph node concentration',
            'Initial heart plasma concentration',
            'Initial kidney plasma concentration',
            'Initial brain plasma concentration',
            'Initial muscle plasma concentration',
            'Initial marrow plasma concentration',
            'Initial thymus plasma concentration',
            'Initial skin plasma concentration',
            'Initial fat plasma concentration',
            'Initial liver plasma concentration',
            'Initial other plasma concentration',
            'Initial lung ISF concentration',
            'Initial heart ISF concentration',
            'Initial kidney ISF concentration',
            'Initial brain ISF concentration',
            'Initial muscle ISF concentration',
            'Initial marrow ISF concentration',
            'Initial thymus ISF concentration',
            'Initial skin ISF concentration',
            'Initial fat ISF concentration',
            'Initial SI ISF concentration',
            'Initial LI ISF concentration',
            'Initial spleen ISF concentration',
            'Initial pancreas ISF concentration',
            'Initial liver ISF concentration',
            'Initial other ISF concentration',
            'Initial SAS concentration'
        ],
        'source': ['Initial condition'] * 29
    }

    # Blood cell concentrations
    blood_cell_concentrations = {
        'name': [
            'C_bc_heart_0', 'C_bc_kidney_0',
            'C_bc_brain_0', 'C_bc_muscle_0', 'C_bc_marrow_0',
            'C_bc_thymus_0', 'C_bc_skin_0', 'C_bc_fat_0',
            'C_bc_liver_0', 'C_bc_other_0'
        ],
        'value': [0.0] * 10,  # Already 0.0
        'units': ['nM/mL'] * 10,  # Changed units to nM/mL
        'description': [f'Initial {organ} blood cell concentration' for organ in [
            'heart', 'kidney', 'brain', 'muscle', 'marrow',
            'thymus', 'skin', 'fat', 'liver', 'other'
        ]],
        'source': ['Initial condition'] * 10
    }

    # ISF concentrations
    isf_concentrations = {
        'name': [
            'C_is_lung', 'C_is_heart', 'C_is_kidney',
            'C_is_muscle', 'C_is_marrow', 'C_is_thymus', 'C_is_skin', 'C_is_fat',
            'C_is_SI', 'C_is_LI', 'C_is_spleen', 'C_is_pancreas', 'C_is_liver',
            'C_is_other', 'C_is_brain_0', 'C_SAS_brain_0'
        ],
        'value': [0.0] * 14 + [0.0] * 2,
        'units': ['nM/mL'] * 16,  # Changed units to nM/mL
        'description': [
            'Lung ISF concentration', 'Heart ISF concentration',
            'Kidney ISF concentration', 'Muscle ISF concentration',
            'Bone marrow ISF concentration', 'Thymus ISF concentration',
            'Skin ISF concentration', 'Fat ISF concentration',
            'Small intestine ISF concentration', 'Large intestine ISF concentration',
            'Spleen ISF concentration', 'Pancreas ISF concentration',
            'Liver ISF concentration', 'Other ISF concentration',
            'Initial brain ISF concentration', 'Initial SAS concentration'
        ],
        'source': ['Initial condition'] * 16
    }

    # Write to Excel - all sheets must be included
    with pd.ExcelWriter('parameters/blood_params.xlsx') as writer:
        pd.DataFrame(volumes).to_excel(writer, sheet_name='Volumes', index=False)
        pd.DataFrame(plasma_flows).to_excel(writer, sheet_name='Plasma_Flows', index=False)
        pd.DataFrame(blood_cell_flows).to_excel(writer, sheet_name='Blood_Cell_Flows', index=False)
        pd.DataFrame(lymph_flows).to_excel(writer, sheet_name='Lymph_Flows', index=False)
        pd.DataFrame(reflection_coeffs).to_excel(writer, sheet_name='Reflection_Coefficients', index=False)
        pd.DataFrame(plasma_concentrations).to_excel(writer, sheet_name='Plasma_Concentrations', index=False)
        pd.DataFrame(blood_cell_concentrations).to_excel(writer, sheet_name='Blood_Cell_Concentrations', index=False)
        pd.DataFrame(isf_concentrations).to_excel(writer, sheet_name='ISF_Concentrations', index=False)

def create_brain_params_excel():
    # Brain volumes
    volumes = {
        'name': [
            'Vp_brain', 'VBBB_brain', 'VIS_brain', 
            'VBC_brain', 'V_ES_brain'
        ],
        'value': [
            214.5,   # Brain plasma volume
            7.25,    # BBB endosomal volume
            275.0,   # Brain ISF volume
            175.5,   # Brain blood cell volume
            7.25     # Brain endosomal volume
        ],
        'units': ['mL'] * 5,
        'description': [
            'Brain plasma volume',
            'BBB endosomal volume',
            'Brain ISF volume',
            'Brain blood cell volume',
            'Brain endosomal volume'
        ],
        'source': ['Table S2'] * 5
    }

    # Flow rates and clearances
    flows = {
        'name': [
            'Q_p_brain', 'Q_bc_brain', 'Q_ISF_brain',
            'Q_CSF_brain', 'CLup_brain', 'L_brain'
        ],
        'value': [
            21453.0,  # Brain plasma flow
            17553.0,  # Brain blood cell flow
            10.5,     # Brain ISF flow
            24.0,     # CSF production rate
            0.03,     # Brain uptake clearance
            215.0     # Brain lymph flow
        ],
        'units': ['mL/h'] * 6,
        'description': [
            'Brain plasma flow rate',
            'Brain blood cell flow rate',
            'Brain ISF flow rate',
            'CSF production rate',
            'Brain uptake clearance',
            'Brain lymph flow rate'
        ],
        'source': ['Table S2'] * 5 + ['Parameter estimate']
    }

    # Kinetic parameters
    kinetics = {
        'name': [
            'kon_FcRn', 'koff_FcRn', 'kdeg',
            'FR', 'FcRn_free_BBB', 'f_BBB',
            'sigma_V_BBB', 'sigma_V_BCSFB',
            'sigma_L_brain_ISF'
        ],
        'value': [
            0.1,    # FcRn binding rate
            0.01,   # FcRn unbinding rate
            26.6,     # Degradation rate (from Table 3)
            0.15,   # Recycling fraction
            1.0,    # Free FcRn receptors at BBB
            0.15,   # Blood-brain barrier fraction
            0.5,    # BBB vascular reflection coefficient
            0.9974,    # BCSFB vascular reflection coefficient (from Table 3)
            0.2     # Brain ISF lymphatic reflection coefficient
        ],
        'units': [
            '1/h', '1/h', '1/h',
            'dimensionless', 'dimensionless', 
            'dimensionless', 'dimensionless',
            'dimensionless', 'dimensionless'
        ],
        'description': [
            'FcRn binding rate',
            'FcRn unbinding rate',
            'Degradation rate',
            'Recycling fraction',
            'Free FcRn receptors at BBB',
            'Blood-brain barrier fraction',
            'BBB vascular reflection coefficient',
            'BCSFB vascular reflection coefficient',
            'Brain ISF lymphatic reflection coefficient'
        ],
        'source': ['Brain parameter'] * 9
    }

    # Initial concentrations (only for ODEs and coupling parameters)
    concentrations = {
        'name': [
            # Initial values for brain compartments (ODEs)
            'C_p_brain_0',
            'C_BBB_unbound_brain_0',
            'C_BBB_bound_brain_0',
            'C_is_brain_0',
            'C_bc_brain_0',
            
            # Initial values for coupling parameters from other models
            'C_p_lung_0',
            'C_bc_lung_0',
            'C_SAS_brain_0',
            'C_BCSFB_bound_brain_0'
        ],
        'value': [0.0] * 5 + [0.0] * 4,  
        'units': ['nM/mL'] * 9,  # Changed units to nM/mL
        'description': [
            'Initial brain plasma concentration',
            'Initial BBB unbound concentration',
            'Initial BBB bound concentration',
            'Initial interstitial space concentration in brain',
            'Initial brain blood cell concentration',
            
            'Initial lung plasma concentration',
            'Initial lung blood cell concentration',
            'Initial CSF SAS concentration',
            'Initial BCSFB bound concentration'
        ],
        'source': ['Initial condition'] * 9
    }

    # Write to Excel
    with pd.ExcelWriter('parameters/brain_params.xlsx') as writer:
        pd.DataFrame(volumes).to_excel(writer, sheet_name='Volumes', index=False)
        pd.DataFrame(flows).to_excel(writer, sheet_name='Flows', index=False)
        pd.DataFrame(kinetics).to_excel(writer, sheet_name='Kinetics', index=False)
        pd.DataFrame(concentrations).to_excel(writer, sheet_name='Concentrations', index=False)

def create_csf_params_excel():
    # CSF volumes (keep this at the top since we need these values to calculate f_LV)
    volumes = {
        'name': [
            'V_BCSFB_brain', 'V_LV_brain', 'V_TFV_brain', 
            'V_CM_brain', 'V_SAS_brain', 'V_ES_brain'
        ],
        'value': [
            7.25,    # Brain endosomal volume [27]
            22.5,    # Lateral ventricle volume [81]
            22.5,    # Third/Fourth ventricle volume [81]
            7.5,     # Cisterna magna volume [81]
            90.0,    # Subarachnoid space volume [81]
            7.25     # Brain endosomal volume [27]
        ],
        'units': ['mL'] * 6,
        'description': [
            'BCSFB volume',
            'Lateral ventricle volume',
            'Third/Fourth ventricle volume',
            'Cisterna magna volume',
            'Subarachnoid space volume',
            'Brain endosomal volume'
        ],
        'source': ['Table S2'] * 6
    }

    # Flow rates and clearances
    flows = {
        'name': [
            'Q_CSF_brain', 'Q_ISF_brain', 'CLup_brain'
        ],
        'value': [
            24.0,    # CSF production [81]
            10.5,    # ISF production [81]
            0.1      # Brain uptake clearance
        ],
        'units': ['mL/h'] * 3,
        'description': [
            'CSF production rate',
            'ISF flow rate',
            'Brain uptake clearance'
        ],
        'source': ['Table S2', 'Table S2', 'Parameter estimate']
    }

    # Calculate f_LV
    V_LV = volumes['value'][volumes['name'].index('V_LV_brain')]
    V_TFV = volumes['value'][volumes['name'].index('V_TFV_brain')]
    f_LV_value = V_LV / (V_LV + V_TFV)

    # Kinetic parameters with f_LV included
    kinetics = {
        'name': [
            'kon_FcRn', 'koff_FcRn', 'kdeg',
            'FR', 'FcRn_free_BCSFB', 'sigma_V_BCSFB',
            'sigma_L_SAS', 'f_BBB', 'f_BCSFB', 'f_LV'  # Added f_LV
        ],
        'value': [
            0.1,   # FcRn binding rate
            0.01,  # FcRn unbinding rate
            26.6,   # Degradation rate (from Table 3)   
            0.15,  # Recycling fraction
            1.0,   # Free FcRn receptors
            0.9974,   # BCSFB vascular reflection coefficient (from Table 3)
            0.2,   # SAS lymphatic reflection coefficient
            0.15,  # Blood-brain barrier fraction
            0.15,  # Blood-CSF barrier fraction
            f_LV_value  # Calculated lateral ventricle fraction
        ],
        'units': [
            '1/h', '1/h', '1/h',
            'dimensionless', 'dimensionless', 
            'dimensionless', 'dimensionless',
            'dimensionless', 'dimensionless',
            'dimensionless'  # Added f_L
        ],
        'description': [
            'FcRn binding rate',
            'FcRn unbinding rate',
            'Degradation rate',
            'Recycling fraction',
            'Free FcRn receptors at BCSFB',
            'BCSFB vascular reflection coefficient',
            'SAS lymphatic reflection coefficient',
            'Blood-brain barrier fraction',
            'Blood-CSF barrier fraction',
            'Lateral ventricle fraction'
        ],
        'source': ['CSF parameter'] * 10
    }

    # Initial concentrations and coupling parameters
    concentrations = {
        'name': [
            'C_BCSFB_unbound_brain_0', 'C_BCSFB_bound_brain_0',
            'C_LV_brain_0', 'C_TFV_brain_0', 'C_CM_brain_0', 'C_SAS_brain_0',
            'C_p_brain_0', 'C_is_brain_0'
        ],
        'value': [0.0] * 6 + [0.0] * 2,  # Only csf species start at 10.0
        'units': ['nM/mL'] * 8,  # Changed units to nM/mL
        'description': [
            'Initial BCSFB unbound concentration',
            'Initial BCSFB bound concentration',
            'Initial lateral ventricle concentration',
            'Initial third/fourth ventricle concentration',
            'Initial cisterna magna concentration',
            'Initial subarachnoid space concentration',
            'Initial brain plasma concentration',
            'Initial brain ISF concentration'
        ],
        'source': ['Initial condition'] * 8
    }

    # Create Excel writer object
    with pd.ExcelWriter('parameters/csf_params.xlsx') as writer:
        pd.DataFrame(volumes).to_excel(writer, sheet_name='Volumes', index=False)
        pd.DataFrame(flows).to_excel(writer, sheet_name='Flows', index=False)
        pd.DataFrame(kinetics).to_excel(writer, sheet_name='Kinetics', index=False)
        pd.DataFrame(concentrations).to_excel(writer, sheet_name='Concentrations', index=False)

def create_lung_params_excel():
    # Lung volumes
    volumes = {
        'name': [
            'Vp_lung', 'VBC_lung', 'VIS_lung', 'VES_lung'
        ],
        'value': [
            55.0,    # Plasma volume from table (55.0 mL)
            45.0,    # Blood cell volume from table (45.0 mL)
            300.0,   # Interstitial volume from table (300.0 mL)
            5.0      # Endosomal volume from table (5.00 mL)
        ],
        'units': ['mL'] * 4,
        'description': [
            'Lung plasma volume',
            'Lung blood cell volume',
            'Lung interstitial volume',
            'Lung endosomal volume'
        ],
        'source': ['Supplementary Table 5'] * 4
    }

    # Flow rates and clearances
    flows = {
        'name': [
            'Q_p_lung', 'Q_bc_lung', 'L_lung', 'CLup_lung'
        ],
        'value': [
            181913.0,  # Plasma flow from table (181913 mL/h)
            148838.0,  # Blood cell flow from table (148838 mL/h)
            183.0,     # Lymph flow from table (183 mL/h)
            0.1        # Uptake clearance (kept as estimate since not in table)
        ],
        'units': ['mL/h'] * 4,
        'description': [
            'Lung plasma flow rate',
            'Lung blood cell flow rate',
            'Lung lymph flow rate',
            'Lung uptake clearance'
        ],
        'source': ['Supplementary Table 5'] * 3 + ['Parameter estimate']
    }

    # Kinetic parameters
    kinetics = {
        'name': [
            'kon_FcRn', 'koff_FcRn', 'kdeg',
            'FR',
            'sigma_V_lung', 'sigma_L_lung'
        ],
        'value': [
            0.1,    # FcRn binding rate
            0.01,   # FcRn unbinding rate
            26.6,   # Degradation rate
            0.15,   # Recycling fraction
            0.2,    # Vascular reflection coefficient
            0.2     # Lymphatic reflection coefficient
        ],
        'units': [
            '1/h', '1/h', '1/h',
            'dimensionless',
            'dimensionless', 'dimensionless'
        ],
        'description': [
            'FcRn binding rate',
            'FcRn unbinding rate',
            'Degradation rate constant of unbound mAb in endosomal space',
            'Recycling fraction',
            'Lung vascular reflection coefficient',
            'Lung lymphatic reflection coefficient'
        ],
        'source': ['Parameter estimate'] * 6
    }

    # Initial concentrations
    concentrations = {
        'name': [
            'C_p_lung_0', 'C_bc_lung_0', 'C_is_lung_0',
            'C_e_unbound_lung_0', 'C_e_bound_lung_0',
            'C_p_0', 'C_bc_0',
            'FcRn_free_lung_0'
        ],
        'value': [0.0] * 5 + [0.0] * 2 + [1.0],  # Only lung species start at 10.0, FcRn at 1.0
        'units': ['nM/mL'] * 8,
        'description': [
            'Initial plasma concentration in lung',
            'Initial blood cells concentration in lung',
            'Initial interstitial space concentration in lung',
            'Initial unbound endosomal concentration in lung',
            'Initial bound endosomal concentration in lung',
            'Initial plasma concentration in blood',
            'Initial blood cells concentration in blood',
            'Initial free FcRn concentration in lung'
        ],
        'source': ['Initial condition'] * 8
    }

    # Replace the return statement with Excel writer
    with pd.ExcelWriter('parameters/lung_params.xlsx') as writer:
        pd.DataFrame(volumes).to_excel(writer, sheet_name='Volumes', index=False)
        pd.DataFrame(flows).to_excel(writer, sheet_name='Flows', index=False)
        pd.DataFrame(kinetics).to_excel(writer, sheet_name='Kinetics', index=False)
        pd.DataFrame(concentrations).to_excel(writer, sheet_name='Concentrations', index=False)

def create_liver_params_excel():
    # Liver volumes
    volumes = {
        'name': [
            'Vp_liver', 'VBC_liver', 'VIS_liver', 'VES_liver'
        ],
        'value': [
            108.0,   # Liver plasma volume
            88.5,    # Liver blood cell volume
            435.0,   # Liver ISF volume
            10.8     # Liver endosomal volume
        ],
        'units': ['mL'] * 4,
        'description': [
            'Liver plasma volume',
            'Liver blood cell volume',
            'Liver ISF volume',
            'Liver endosomal volume'
        ],
        'source': ['Table S2'] * 4
    }

    # Flow rates and clearances
    flows = {
        'name': [
            'Q_p_liver', 'Q_bc_liver', 'L_liver',
            'Q_p_spleen', 'Q_bc_spleen', 'L_spleen',
            'Q_p_pancreas', 'Q_bc_pancreas', 'L_pancreas',
            'Q_p_SI', 'Q_bc_SI', 'L_SI',
            'Q_p_LI', 'Q_bc_LI', 'L_LI',
            'CLup_liver'
        ],
        'value': [
            43200.0,  # Liver plasma flow
            35400.0,  # Liver blood cell flow
            432.0,    # Liver lymph flow
            1080.0,   # Spleen plasma flow
            885.0,    # Spleen blood cell flow
            10.8,     # Spleen lymph flow
            1080.0,   # Pancreas plasma flow
            885.0,    # Pancreas blood cell flow
            10.8,     # Pancreas lymph flow
            4320.0,   # SI plasma flow
            3540.0,   # SI blood cell flow
            43.2,     # SI lymph flow
            4320.0,   # LI plasma flow
            3540.0,   # LI blood cell flow
            43.2,     # LI lymph flow
            0.05      # Liver uptake clearance
        ],
        'units': ['mL/h'] * 16,
        'description': [
            'Liver plasma flow rate',
            'Liver blood cell flow rate',
            'Liver lymph flow rate',
            'Spleen plasma flow rate',
            'Spleen blood cell flow rate',
            'Spleen lymph flow rate',
            'Pancreas plasma flow rate',
            'Pancreas blood cell flow rate',
            'Pancreas lymph flow rate',
            'Small intestine plasma flow rate',
            'Small intestine blood cell flow rate',
            'Small intestine lymph flow rate',
            'Large intestine plasma flow rate',
            'Large intestine blood cell flow rate',
            'Large intestine lymph flow rate',
            'Liver uptake clearance'
        ],
        'source': ['Table S2'] * 16
    }

    # Kinetic parameters
    kinetics = {
        'name': [
            'kon_FcRn', 'koff_FcRn', 'kdeg', 'FR',
            'FcRn_free_liver', 'sigma_V_liver', 'sigma_L_liver'
        ],
        'value': [
            0.1,    # FcRn binding rate
            0.01,   # FcRn unbinding rate
            26.6,   # Degradation rate
            0.15,   # Recycling fraction
            1.0,    # Free FcRn receptors
            0.95,   # Vascular reflection coefficient
            0.2     # Lymphatic reflection coefficient
        ],
        'units': [
            '1/h', '1/h', '1/h', 'dimensionless',
            'dimensionless', 'dimensionless', 'dimensionless'
        ],
        'description': [
            'FcRn binding rate',
            'FcRn unbinding rate',
            'Degradation rate',
            'Recycling fraction',
            'Free FcRn receptors in liver',
            'Liver vascular reflection coefficient',
            'Liver lymphatic reflection coefficient'
        ],
        'source': ['Liver parameter'] * 7
    }

    # Initial concentrations
    concentrations = {
        'name': [
            # Primary liver species
            'C_p_liver_0', 'C_bc_liver_0', 'C_is_liver_0',
            'C_e_unbound_liver_0', 'C_e_bound_liver_0',
            'FcRn_free_liver_0',
            
            # Coupling parameters
            'C_p_lung_0', 'C_bc_lung_0',
            'C_p_spleen_0', 'C_bc_spleen_0',
            'C_p_pancreas_0', 'C_bc_pancreas_0',
            'C_p_SI_0', 'C_bc_SI_0',
            'C_p_LI_0', 'C_bc_LI_0'
        ],
        'value': [0.0] * 5 + [1.0] + [0.0] * 10,  # Only liver species start at 10.0, FcRn at 1.0
        'units': ['nM/mL'] * 16,
        'description': [
            'Initial liver plasma concentration',
            'Initial liver blood cell concentration',
            'Initial liver ISF concentration',
            'Initial liver endosomal unbound concentration',
            'Initial liver endosomal bound concentration',
            'Initial liver free FcRn concentration',
            'Initial lung plasma concentration',
            'Initial lung blood cell concentration',
            'Initial spleen plasma concentration',
            'Initial spleen blood cell concentration',
            'Initial pancreas plasma concentration',
            'Initial pancreas blood cell concentration',
            'Initial SI plasma concentration',
            'Initial SI blood cell concentration',
            'Initial LI plasma concentration',
            'Initial LI blood cell concentration'
        ],
        'source': ['Initial condition'] * 16
    }

    # Write to Excel
    with pd.ExcelWriter('parameters/liver_params.xlsx') as writer:
        pd.DataFrame(volumes).to_excel(writer, sheet_name='Volumes', index=False)
        pd.DataFrame(flows).to_excel(writer, sheet_name='Flows', index=False)
        pd.DataFrame(kinetics).to_excel(writer, sheet_name='Kinetics', index=False)
        pd.DataFrame(concentrations).to_excel(writer, sheet_name='Concentrations', index=False)

def main():
    # Create parameters directory if it doesn't exist
    Path('parameters').mkdir(exist_ok=True)
    
    print("Creating blood parameters Excel file...")
    create_blood_params_excel()
    
    print("Creating brain parameters Excel file...")
    create_brain_params_excel()
    
    print("Creating CSF parameters Excel file...")
    create_csf_params_excel()
    
    print("Creating lung parameters Excel file...")
    create_lung_params_excel()
    
    print("Creating liver parameters Excel file...")
    create_liver_params_excel()
    
    print("Excel files created successfully!")

if __name__ == "__main__":
    main()