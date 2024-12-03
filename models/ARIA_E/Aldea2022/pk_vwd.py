import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jax import jit
import pk_sbml
import vwd_sbml

from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse

# Your existing parameters and dosing schedules
pk_params = {
    "F": 0.494,      # Bioavailability
    "D1": 0.0821,    # Duration of zero-order absorption
    "KA": 0.220,     # First-order absorption rate constant
    "CL": 0.336,     # Clearance
    "Vc": 3.52,      # Volume of central compartment
    "Q": 0.869,      # Intercompartmental clearance
    "Vp": 6.38,      # Volume of peripheral compartment
}

# Rename pd_params to vwd_params for consistency
vwd_params = {
    "alpha_removal": 0.126e-3,
    "k_repair": 12.4e-3,
    "A_beta0": 3.2,
    "VWD0": 0.0,
    "BGTS_max": 60.0,
    "EG50": 1.0,
    "pow": 3.72
}


# Define SC dosing schedule
sc_doses = [
    (0, 450.0),    # Day 1
    (28, 450.0),   # Week 4
    (56, 900.0),   # Week 8
    (84, 900.0),   # Week 12
    (112, 1200.0), # Week 16
    (140, 1200.0), # Week 20
]
    
# Add maintenance doses (1200 mg) for 120 weeks
for week in range(24, 121, 4):  # Changed from 101 to 121 to include full 120 weeks
    day = week * 7
    sc_doses.append((day, 1200.0))
    
# Define IV dosing schedule (all zeros)
iv_doses = [
    (0, 0.0),
    (28, 0.0),
    (56, 0.0),
]
    
# Add maintenance IV doses (zeros) for 120 weeks
for week in range(12, 121, 4):  # Changed from 52 to 121
    day = week * 7
    iv_doses.append((day, 0.0))
    
# Force regeneration of SBML files
print("Generating SBML files...")
pk_sbml.main(pk_params, sc_doses, iv_doses)
vwd_sbml.main(vwd_params)

# Force regeneration of JAX files
print("Generating JAX models...")
modelData = parse.ParseSBMLFile("pk_sbml.xml")
GenerateModel(modelData, "pk_jax.py")

modelData = parse.ParseSBMLFile("vwd_sbml.xml")
GenerateModel(modelData, "vwd_jax.py")

# Force reload of the generated modules
import importlib
import pk_jax
import vwd_jax
importlib.reload(pk_jax)
importlib.reload(vwd_jax)

# Now import the reloaded modules
from pk_jax import RateofSpeciesChange as PKRateofSpeciesChange
from pk_jax import AssignmentRule as PKAssignmentRule
from pk_jax import y0 as pk_y0, w0, t0, c

from vwd_jax import RateofSpeciesChange as VWDRateofSpeciesChange
from vwd_jax import AssignmentRule as VWDAssignmentRule
from vwd_jax import y0 as vwd_y0

# Combine initial conditions
y0_combined = jnp.concatenate([pk_y0, vwd_y0])

# Helper function to create c_vwd array
@jit
def build_c_vwd(central_conc):
    return jnp.array([
        vwd_params["alpha_removal"],
        vwd_params["k_repair"],
        vwd_params["A_beta0"],
        vwd_params["VWD0"],
        vwd_params["BGTS_max"],
        vwd_params["EG50"],
        vwd_params["pow"],
        central_conc
    ])

# Modify the ODE function to use the helper
@jit
def ode_func(t, y, args):
    w, c = args
    pk_rate = PKRateofSpeciesChange()
    vwd_rate = VWDRateofSpeciesChange()
    pk_assignment = PKAssignmentRule()
    vwd_assignment = VWDAssignmentRule()
    
    # Split y into PK and VWD components
    y_pk = y[:len(pk_y0)]
    y_vwd = y[len(pk_y0):]
    
    # Calculate central concentration and build c_vwd
    central_conc = y_pk[1] / c[4]  # c[4] is Vc
    c_vwd = build_c_vwd(central_conc)
    
    # Rest of the function remains the same
    w_vwd = jnp.array([0.0])
    BGTS = vwd_assignment(y_vwd, w_vwd, c_vwd, t)[0]
    
    w = jnp.where(
        BGTS < 4.0,
        pk_assignment(y_pk, w, c, t),
        w * 0.0
    )
    
    dy_pk_dt = pk_rate(y_pk, t, w, c)
    dy_vwd_dt = vwd_rate(y_vwd, t, w_vwd, c_vwd)
    
    return jnp.concatenate([dy_pk_dt, dy_vwd_dt])

# Set up simulation parameters
t1 = 840.0  # End time (120 weeks)
dt0 = 0.01  # Time step size
n_points = 84000  # Number of points to save

# Create diffrax solver
term = diffrax.ODETerm(ode_func)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points))

# Solve the ODE system
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0_combined,
    args=(w0, c),
    saveat=saveat,
    max_steps=500000
)

# Update plotting code to use new indices
y_indexes = {
    'A': 0, 
    'C': 1, 
    'Cp': 2, 
    'A_beta': 3, 
    'VWD': 4,
    'BGTS': 5
}

# Create vertically stacked plots with improved readability
plt.figure(figsize=(10, 12))
plt.subplots_adjust(hspace=0.5)

# Plot 1: Central Concentration with doses
ax1 = plt.subplot(4, 1, 1)

# Get updated w values for all timepoints
w_values = jax.vmap(lambda t, y: PKAssignmentRule()(y, w0, c, t))(sol.ts, sol.ys)

line2 = plt.plot(sol.ts/7, sol.ys[:, y_indexes['C']]/pk_params['Vc'], 
                 label='Central Concentration (C)', 
                 color='m', 
                 linewidth=2)

# Add dose markers where Dsc is non-zero
dose_mask = w_values[:, 0] > 0  # Find where doses are applied
line1 = plt.plot(sol.ts[dose_mask]/7, w_values[dose_mask, 0]/10, '+', 
                 color='orange', 
                 markersize=10, 
                 markeredgewidth=2, 
                 label='Dose administrations')

# Add secondary y-axis for true dose values
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0]*10, ax1.get_ylim()[1]*10)
ax2.set_ylabel('Dose (mg)', color='orange', fontsize=10)
ax2.tick_params(axis='y', labelcolor='orange')

ax1.set_xlabel('Time (weeks)', fontsize=10)
ax1.set_ylabel('Concentration (mcg/mL)', color='m', fontsize=10)
ax1.tick_params(axis='y', labelcolor='m')
ax1.grid(True, alpha=0.3)

# Get lines from both axes and combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=10)

ax1.set_title('PK: Central Concentration', fontsize=12, pad=20)

# Plot 2: Amyloid Beta
plt.subplot(4, 1, 2)
plt.plot(sol.ts/7, sol.ys[:, y_indexes['A_beta']], 
         label='Local Amyloid (Aβ)', 
         color='g',
         linewidth=2)
plt.xlabel('Time (weeks)', fontsize=10)
plt.ylabel('Level', fontsize=10)
plt.title('Local Amyloid', fontsize=12, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 5)

# Plot 3: VWD
plt.subplot(4, 1, 3)
plt.plot(sol.ts/7, sol.ys[:, y_indexes['VWD']], 
         label='VWD', 
         color='r',
         linewidth=2)
plt.xlabel('Time (weeks)', fontsize=10)
plt.ylabel('Level', fontsize=10)
plt.title('Vascular Wall Disturbance', fontsize=12, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Plot 4: BGTS Score
plt.subplot(4, 1, 4)
w_vwd = jnp.array([0.0])
central_conc = sol.ys[:, y_indexes['C']] / pk_params['Vc']
bgts_values = jax.vmap(
    lambda y_vwd, cc: VWDAssignmentRule()(y_vwd, w_vwd, build_c_vwd(cc), 0.0)[0]
)(sol.ys[:, len(pk_y0):], central_conc)
plt.plot(sol.ts/7, bgts_values, 
         label='BGTS', 
         color='c',
         linewidth=2)
plt.axhline(y=4, color='r', linestyle='--', label='BGTS threshold')
plt.xlabel('Time (weeks)', fontsize=10)
plt.ylabel('BGTS Score', fontsize=10)
plt.title('ARIA-E BGTS Score', fontsize=12, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 30)

# Plot 5: Peripheral Concentration
#plt.subplot(5, 1, 5)
#plt.plot(sol.ts/7, sol.ys[:, y_indexes['Cp']]/pk_params['Vp'], 
#         label='Peripheral Concentration (Cp)', 
#         color='g',
#         linewidth=2)
#plt.xlabel('Time (weeks)', fontsize=10)
#plt.ylabel('Concentration (mcg/mL)', fontsize=10)
#plt.title('Peripheral Compartment Concentration', fontsize=12, pad=20)
#plt.legend(fontsize=10)
#plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional BGTS vs VWD plot
plt.figure(figsize=(8, 6))
plt.plot(sol.ys[:, y_indexes['VWD']], bgts_values, 
         color='k',
         linewidth=2)
plt.axhline(y=4, color='r', linestyle='--', label='BGTS threshold')
plt.xlabel('VWD Score', fontsize=10)
plt.ylabel('BGTS Score', fontsize=10)
plt.title('ARIA-E Score vs Vascular Wall Disturbance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show() 