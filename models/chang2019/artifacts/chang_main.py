import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jax import jit
import blood_sbml
import brain_sbml
import pandas as pd
import csf_sbml

from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse

def load_blood_params():
    # Read all sheets from the Excel file
    excel_file = 'parameters/blood_params.xlsx'
    sheets = [
        'Volumes',
        'Plasma_Flows',
        'Blood_Cell_Flows',
        'Plasma_Concentrations',
        'Blood_Cell_Concentrations',
        'ISF_Concentrations',
        'Lymph_Flows',
        'Reflection_Coefficients'
    ]
    
    # Combine all parameters into a single dictionary
    params = {}
    for sheet in sheets:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    return params

def load_brain_params():
    # Read all sheets from the Excel file
    excel_file = 'parameters/brain_params.xlsx'
    volumes_df = pd.read_excel(excel_file, sheet_name='Volumes')
    flows_df = pd.read_excel(excel_file, sheet_name='Flows')
    kinetics_df = pd.read_excel(excel_file, sheet_name='Kinetics')
    concentrations_df = pd.read_excel(excel_file, sheet_name='Concentrations')
    
    # Combine all parameters into a single dictionary
    params = {}
    for df in [volumes_df, flows_df, kinetics_df, concentrations_df]:
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    return params

def load_csf_params():
    # Read all sheets from the Excel file
    excel_file = 'parameters/csf_params.xlsx'
    volumes_df = pd.read_excel(excel_file, sheet_name='Volumes')
    flows_df = pd.read_excel(excel_file, sheet_name='Flows')
    kinetics_df = pd.read_excel(excel_file, sheet_name='Kinetics')
    concentrations_df = pd.read_excel(excel_file, sheet_name='Concentrations')
    
    # Combine all parameters into a single dictionary
    params = {}
    for df in [volumes_df, flows_df, kinetics_df, concentrations_df]:
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    return params

# Load all parameters
blood_params = load_blood_params()
brain_params = load_brain_params()
csf_params = load_csf_params()
params = {**blood_params, **brain_params, **csf_params}

# Generate all SBML files
print("Generating Blood SBML file...")
blood_sbml.main(params)

print("Generating Brain SBML file...")
brain_sbml.main(params)

print("Generating CSF SBML file...")
csf_sbml.main(params)

# Parse all SBML files
print("Parsing Blood SBML...")
blood_model_data = parse.ParseSBMLFile("blood_sbml.xml")
GenerateModel(blood_model_data, "blood_jax.py")

print("Parsing Brain SBML...")
brain_model_data = parse.ParseSBMLFile("brain_sbml.xml")
GenerateModel(brain_model_data, "brain_jax.py")

print("Parsing CSF SBML...")
csf_model_data = parse.ParseSBMLFile("csf_sbml.xml")
GenerateModel(csf_model_data, "csf_jax.py")

# Import all generated models
import importlib
import blood_jax
import brain_jax
import csf_jax
importlib.reload(blood_jax)
importlib.reload(brain_jax)
importlib.reload(csf_jax)

from blood_jax import RateofSpeciesChange as BloodRateofSpeciesChange
from brain_jax import RateofSpeciesChange as BrainRateofSpeciesChange
from csf_jax import RateofSpeciesChange as CSFRateofSpeciesChange
from blood_jax import y0 as blood_y0, c as blood_c
from brain_jax import y0 as brain_y0, c as brain_c
from csf_jax import y0 as csf_y0, c as csf_c

# Combine initial conditions
y0_combined = jnp.concatenate([blood_y0, brain_y0, csf_y0])

# Modified ODE function to include CSF
@jit
def ode_func(t, y, args):
    """
    Combined ODE function for all compartments
    
    States y are ordered as:
    blood_states = y[:3]  # plasma, blood cells, lymph node
    brain_states = y[3:8]  # plasma, BBB unbound, BBB bound, ISF, blood cells
    csf_states = y[8:]    # BCSFB unbound, BCSFB bound, LV, TFV, CM, SAS
    """
    # Split states into individual models
    blood_states = y[:3]
    brain_states = y[3:8]
    csf_states = y[8:]

    # Update blood parameters with brain and CSF coupling values
    blood_params = blood_c
    blood_params = blood_params.at[0].set(brain_states[0])  # C_p_brain
    blood_params = blood_params.at[1].set(brain_states[4])  # C_BC_brain
    blood_params = blood_params.at[2].set(brain_states[3])  # C_IS_brain
    blood_params = blood_params.at[3].set(csf_states[5])    # C_SAS_brain

    # Update brain parameters with blood and CSF coupling values
    brain_params = brain_c
    brain_params = brain_params.at[0].set(blood_states[0])  # C_p_lung (from blood plasma)
    brain_params = brain_params.at[1].set(blood_states[1])  # C_BC_lung (from blood cells)
    brain_params = brain_params.at[2].set(csf_states[5])    # C_SAS_brain
    brain_params = brain_params.at[3].set(csf_states[1])    # C_BCSFB_bound_brain

    # Update CSF parameters with brain coupling values
    csf_params = csf_c
    csf_params = csf_params.at[0].set(brain_states[0])  # C_p_brain
    csf_params = csf_params.at[1].set(brain_states[3])  # C_IS_brain

    # Calculate rates using updated parameters
    blood_rate = BloodRateofSpeciesChange()
    brain_rate = BrainRateofSpeciesChange()
    csf_rate = CSFRateofSpeciesChange()

    dy_dt_blood = blood_rate(blood_states, t, {}, blood_params)
    dy_dt_brain = brain_rate(brain_states, t, {}, brain_params)
    dy_dt_csf = csf_rate(csf_states, t, {}, csf_params)

    return jnp.concatenate([dy_dt_blood, dy_dt_brain, dy_dt_csf])

# Set up solver
t0 = 0.0
t1 = 24.0  # 24 hours
dt0 = 0.0002
n_points = 1000

term = diffrax.ODETerm(ode_func)
solver = diffrax.Kvaerno5()
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points))

# Solve system
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0_combined,
    args=(None,),  # No additional args needed since we handle everything in ode_func
    saveat=saveat,
    max_steps=10000000,
    stepsize_controller=diffrax.PIDController(
        rtol=1e-8,
        atol=1e-8
    )
)

# Create figure with tighter layout
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.4)  # Add more space between subplots

# Blood compartments plot
ax1.plot(sol.ts, sol.ys[:, 0], label='Plasma', linewidth=2)
ax1.plot(sol.ts, sol.ys[:, 1], label='Blood Cells', linewidth=2)
ax1.plot(sol.ts, sol.ys[:, 2], label='Lymph Node', linewidth=2)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Concentration')
ax1.set_title('Blood Compartments')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Brain compartments plot
ax2.plot(sol.ts, sol.ys[:, 3], label='Brain Plasma', linewidth=2)
ax2.plot(sol.ts, sol.ys[:, 4], label='BBB Unbound', linewidth=2)
ax2.plot(sol.ts, sol.ys[:, 5], label='BBB Bound', linewidth=2)
ax2.plot(sol.ts, sol.ys[:, 6], label='Brain ISF', linewidth=2)
ax2.plot(sol.ts, sol.ys[:, 7], label='Brain Blood Cells', linewidth=2)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Concentration')
ax2.set_title('Brain Compartments')
ax2.legend()
ax2.grid(True, alpha=0.3)

# CSF compartments plot
ax3.plot(sol.ts, sol.ys[:, 8], label='BCSFB Unbound', linewidth=2)
ax3.plot(sol.ts, sol.ys[:, 9], label='BCSFB Bound', linewidth=2)
ax3.plot(sol.ts, sol.ys[:, 10], label='Lateral Ventricle', linewidth=2)
ax3.plot(sol.ts, sol.ys[:, 11], label='Third/Fourth Ventricle', linewidth=2)
ax3.plot(sol.ts, sol.ys[:, 12], label='Cisterna Magna', linewidth=2)
ax3.plot(sol.ts, sol.ys[:, 13], label='Subarachnoid Space', linewidth=2)
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Concentration')
ax3.set_title('CSF Compartments')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('concentration_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final concentrations in a more compact format
print("\nFinal Concentrations:")
print("-" * 30)
compartments = [
    'Blood:', 
    '  Plasma', '  Blood Cells', '  Lymph Node',
    'Brain:', 
    '  Plasma', '  BBB Unbound', '  BBB Bound', '  ISF', '  Blood Cells',
    'CSF:',
    '  BCSFB Unbound', '  BCSFB Bound', '  Lateral Ventricle', 
    '  Third/Fourth Ventricle', '  Cisterna Magna', '  Subarachnoid Space'
]

for i, comp in enumerate(compartments):
    if comp.endswith(':'):
        print(f"\n{comp}")
    else:
        print(f"{comp:<25}: {sol.ys[-1, i-1]:.6f}") 