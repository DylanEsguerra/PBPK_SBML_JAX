import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jax import jit
import blood_sbml
import pandas as pd

from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse

def load_blood_params():
    # Read all sheets from the Excel file
    excel_file = 'parameters/blood_params.xlsx'
    volumes_df = pd.read_excel(excel_file, sheet_name='Volumes')
    plasma_flows_df = pd.read_excel(excel_file, sheet_name='Plasma_Flows')
    blood_cell_flows_df = pd.read_excel(excel_file, sheet_name='Blood_Cell_Flows')
    lymph_flows_df = pd.read_excel(excel_file, sheet_name='Lymph_Flows')
    reflection_df = pd.read_excel(excel_file, sheet_name='Reflection_Coefficients')
    plasma_conc_df = pd.read_excel(excel_file, sheet_name='Plasma_Concentrations')
    blood_cell_conc_df = pd.read_excel(excel_file, sheet_name='Blood_Cell_Concentrations')
    
    # Combine all parameters into a single dictionary
    params = {}
    for df in [volumes_df, plasma_flows_df, blood_cell_flows_df, lymph_flows_df, 
               reflection_df, plasma_conc_df, blood_cell_conc_df]:
        for _, row in df.iterrows():
            params[row['name']] = row['value']
    
    return params

# Load parameters from Excel
blood_params = load_blood_params()

print("Generating SBML file...")
blood_sbml.main(blood_params)

print("Parsing Blood SBML...")
modelData = parse.ParseSBMLFile("blood_sbml.xml")
GenerateModel(modelData, "blood_jax.py")

# Import the generated model
import importlib
import blood_jax
importlib.reload(blood_jax)

from blood_jax import RateofSpeciesChange
from blood_jax import y0, c as blood_c

# Modified ODE function for isolated blood model
@jit
def ode_func(t, y, args):
    # For isolated simulation, coupling parameters remain constant
    params = blood_c
    
    # Set initial values for coupling parameters from brain
    params = params.at[0].set(blood_params["C_p_brain_0"])    # Brain plasma
    params = params.at[1].set(blood_params["C_bc_brain_0"])   # Brain blood cells
    params = params.at[2].set(blood_params["C_is_brain_0"])   # Brain ISF
    params = params.at[3].set(blood_params["C_SAS_brain_0"])  # CSF SAS
    
    # Calculate rates
    rate = RateofSpeciesChange()
    dy_dt = rate(y, t, {}, params)
    
    return dy_dt

# Set up solver
t0 = 0.0
t1 = 24.0
dt0 = 0.0002
n_points = 1000

term = diffrax.ODETerm(ode_func)
solver = diffrax.Kvaerno5()
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points))

# Initial conditions from parameters (only for species with ODEs)
y0_values = jnp.array([
    blood_params["C_p_0"],      # Plasma
    blood_params["C_bc_0"],     # Blood cells
    blood_params["C_ln_0"]      # Lymph node
])

# Solve system
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
plt.figure(figsize=(10, 6))

# Plot only the main compartments (species with ODEs)
plt.plot(sol.ts, sol.ys[:, 0], label='Plasma', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 1], label='Blood Cells', linewidth=2)
plt.plot(sol.ts, sol.ys[:, 2], label='Lymph Node', linewidth=2)

plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Blood Compartment Concentrations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show() 