import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jax import jit
import csf_sbml
import pandas as pd

from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse

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

# Load parameters from Excel
csf_params = load_csf_params()

print("Generating SBML file...")
csf_sbml.main(csf_params)

print("Parsing CSF SBML...")
modelData = parse.ParseSBMLFile("csf_sbml.xml")
GenerateModel(modelData, "csf_jax.py")

# Import the generated model
import importlib
import csf_jax
importlib.reload(csf_jax)

from csf_jax import RateofSpeciesChange
from csf_jax import y0, c as csf_c

# Modified ODE function for isolated CSF model
@jit
def ode_func(t, y, args):
    # For isolated simulation, brain coupling parameters remain constant
    params = csf_c
    
    # Set initial values for coupling parameters from brain
    params = params.at[0].set(csf_params["C_p_brain_0"])  # Brain plasma
    params = params.at[1].set(csf_params["C_IS_brain_0"]) # Brain ISF
    
    # Calculate rates
    rate = RateofSpeciesChange()
    dy_dt = rate(y, t, {}, params)
    
    return dy_dt

# Set up solver
t0 = 0.0
t1 = 24.0  # 24 hours
dt0 = 0.0002
n_points = 1000

term = diffrax.ODETerm(ode_func)
solver = diffrax.Kvaerno5()
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points))

# Initial conditions from parameters
y0_values = jnp.array([
    csf_params["C_BCSFB_unbound_brain_0"],
    csf_params["C_BCSFB_bound_brain_0"],
    csf_params["C_LV_brain_0"],
    csf_params["C_TFV_brain_0"],
    csf_params["C_CM_brain_0"],
    csf_params["C_SAS_brain_0"]
])

# Solve system with adjusted step control parameters
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0_values,
    args=(None,),
    saveat=saveat,
    max_steps=10000000,
    stepsize_controller=diffrax.PIDController(
        rtol=1e-8,
        atol=1e-8,
        dtmax=0.1
    )
)

# Plotting
plt.figure(figsize=(12, 8))

# Plot all CSF compartment concentrations
plt.plot(sol.ts, sol.ys[:, 0], label='BCSFB Unbound', color='#96CEB4', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 1], label='BCSFB Bound', color='#FF9999', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 2], label='Lateral Ventricle', color='#D4A5A5', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 3], label='Third/Fourth Ventricle', color='#9FA8DA', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 4], label='Cisterna Magna', color='#45B7D1', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 5], label='Subarachnoid Space', color='#FFB347', linewidth=2)

plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('CSF Compartment Concentrations')
plt.legend()
plt.grid(True, alpha=0.5)

plt.show() 