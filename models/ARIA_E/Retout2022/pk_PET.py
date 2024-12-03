import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jax import jit
import pk_sbml
import PET_sbml

from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse

# PK parameters mg/L
pk_params = { 
    "F": 0.494,      # Bioavailability
    "D1": 0.0821,    # Duration of zero-order absorption
    "KA": 0.220,     # First-order absorption rate constant
    "CL": 0.336,     # Clearance
    "Vc": 3.52,      # Volume of central compartment
    "Q": 0.869,      # Intercompartmental clearance
    "Vp": 6.38,      # Volume of peripheral compartment L 
}

# PET parameters (updated to include ARIA parameters)
pet_params = {
    "Ke0": 1740.0,
    "SLOP": 0.019,
    "power": 0.716,
    "SUVR_0": 1.8,
    # ARIA parameters
    "BSAPOE4_non": 8.7E-06,    
    "BSAPOE4_carrier": 3.56E-05,  
    "T50": 323,                 
    "gamma": 2.15,              
    "Emax": 6.05,              
    "EC50": 8.60        #ug per mL       
}

# Define SC dosing schedule
sc_doses = [
    # 105 mg x 3 doses
    (0, 1200.0),
    (28, 1200.0),
    (56, 1200.0),
    
    # 225 mg x 3 doses
    (84, 1200.0),
    (112, 1200.0),
    (140, 1200.0),
    
    # 450 mg x 2 doses
    (168, 1200.0),
    (196, 1200.0),
    
    # 900 mg x 2 doses
    (224, 1200.0),
    (252, 1200.0),
]

# Add maintenance doses (1200 mg)
for week in range(40, 105, 4):
    day = week * 7
    sc_doses.append((day, 1200.0))

iv_doses = [
    (0, 1.0),
    (28, 3.0),
    (56, 6.0),
]

for week in range(12, 52, 4):
    day = week * 7
    iv_doses.append((day, 10.0))

# Add debug prints
print("Generating SBML files...")
pk_sbml.main(pk_params, sc_doses, iv_doses)
PET_sbml.main(pet_params)

print("Parsing PK SBML...")
modelData = parse.ParseSBMLFile("pk_sbml.xml")
GenerateModel(modelData, "pk_jax.py")

print("Parsing PET SBML...")
modelData = parse.ParseSBMLFile("PET_sbml.xml")
print("PET model data:", modelData)
GenerateModel(modelData, "PET_jax.py")

# Import the generated models
import importlib
import pk_jax
import PET_jax
importlib.reload(pk_jax)  # Reload in case the module was previously imported
importlib.reload(PET_jax)

from pk_jax import RateofSpeciesChange as PKRateofSpeciesChange
from pk_jax import AssignmentRule as PKAssignmentRule
from pk_jax import y0 as pk_y0, w0, t0, c

from PET_jax import RateofSpeciesChange as PETRateofSpeciesChange
from PET_jax import AssignmentRule as PETAssignmentRule
from PET_jax import y0 as pet_y0, w0 as pet_w0

# Combine initial conditions
y0_combined = jnp.concatenate([pk_y0, pet_y0])

# Helper function to create c_suvr array
@jit
def build_c_suvr(central_conc):
    return jnp.array([
        pet_params["Ke0"],
        pet_params["SLOP"],
        pet_params["power"],
        pet_params["SUVR_0"],
        pet_params["BSAPOE4_non"],     # index 4
        pet_params["BSAPOE4_carrier"], # index 5
        pet_params["T50"],             # index 6
        pet_params["gamma"],           # index 7
        pet_params["Emax"],           # index 8
        pet_params["EC50"],           # index 9
        central_conc, 
        1.0  # effect_compartment
    ])


# Combined ODE function
@jit
def ode_func(t, y, args):
    w, c = args
    pk_rate = PKRateofSpeciesChange()
    pet_rate = PETRateofSpeciesChange()
    pk_assignment = PKAssignmentRule()
    pet_assignment = PETAssignmentRule()
    
    # Split y into PK and PET components
    y_pk = y[:len(pk_y0)]
    y_pet = y[len(pk_y0):]
    
    # Calculate central concentration and build c_suvr
    central_conc = y_pk[1] / c[4]  # c[4] is Vc
    c_suvr = build_c_suvr(central_conc)
    
    # Get SUVR value from assignment rule
    w_pet = jnp.array([0.0])
    SUVR = pet_assignment(y_pet, w_pet, c_suvr, t)[0]
    
    w = pk_assignment(y_pk, w, c, t)
    # Calculate derivativesx
    dy_pk_dt = pk_rate(y_pk, t, w, c)
    dy_pet_dt = pet_rate(y_pet, t, w_pet, c_suvr)
    
    return jnp.concatenate([dy_pk_dt, dy_pet_dt])

# Set up simulation parameters
t1 = 728.0  # End time (104 weeks)
dt0 = 0.002  # Time step size
n_points = 728000  # Number of points to save

# Create step_ts for accurate dose timing
sc_dose_times = jnp.array([d[0] for d in sc_doses])
iv_dose_times = jnp.array([d[0] for d in iv_doses])
absorption_times = jnp.linspace(sc_dose_times, sc_dose_times + pk_params["D1"], 10)
step_ts = jnp.sort(jnp.concatenate([
    absorption_times.flatten(),
    sc_dose_times + pk_params["D1"],
    iv_dose_times
]))

# Create diffrax solver
term = diffrax.ODETerm(ode_func)
solver = diffrax.Kvaerno5()
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points))

# Solve the ODE system with improved controller
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0_combined,
    args=(w0, c),
    saveat=saveat,
    max_steps=5000000,
    stepsize_controller=diffrax.PIDController(
        rtol=1e-6,
        atol=1e-8,
        step_ts=step_ts,
        dtmax=pk_params["D1"]/5
    )
)

# Update plotting code
y_indexes = {
    'A': 0, 
    'C': 1, 
    'Cp': 2,
    'Ce': 3, 
    'SUVR': 4,
    'ARIA_hazard': 5
}

# Create 2x2 grid of plots
plt.figure(figsize=(15, 12))  # Adjusted figure size for better proportions
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjusted spacing

# Plot 1: Central Concentration with doses (top left)
ax1 = plt.subplot(2, 2, 1)

# Get updated w values for all timepoints
w_values = jax.vmap(lambda t, y: PKAssignmentRule()(y, w0, c, t))(sol.ts, sol.ys)

line2 = plt.plot(sol.ts/7, sol.ys[:, y_indexes['C']]/pk_params['Vc'], 
                 label='Central Concentration (C)', 
                 color='m', 
                 linewidth=2)

# Add dose markers
dose_mask = w_values[:, 0] > 0
line1 = plt.plot(sol.ts[dose_mask]/7, w_values[dose_mask, 0]/10, '+', 
                 color='orange', 
                 markersize=10, 
                 markeredgewidth=2, 
                 label='Dose administrations')

# Add secondary y-axis for doses
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0]*10, ax1.get_ylim()[1]*10)
ax2.set_ylabel('Dose (mg)', color='orange', fontsize=10)
ax2.tick_params(axis='y', labelcolor='orange')

ax1.set_xlabel('Time (weeks)', fontsize=10)
ax1.set_ylabel('Concentration (mcg/mL)', color='m', fontsize=10)
ax1.tick_params(axis='y', labelcolor='m')
ax1.grid(True, alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=10)
ax1.set_title('PK: Central Concentration', fontsize=12, pad=20)

# Plot 2: Effect Compartment Concentration (top right)
plt.subplot(2, 2, 2)
plt.plot(sol.ts/7, sol.ys[:, y_indexes['Ce']], 
         label='Effect Compartment (Ce)', 
         color='g',
         linewidth=2)
plt.xlabel('Time (weeks)', fontsize=10)
plt.ylabel('Concentration (mcg/mL)', fontsize=10)
plt.title('Effect Compartment Concentration', fontsize=12, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: SUVR %CFB (bottom left)
plt.subplot(2, 2, 3)
w_suvr = pet_w0
central_conc = sol.ys[:, y_indexes['C']] / pk_params['Vc']

# Fix: Ensure both arrays have the same size
pet_states = sol.ys[:, len(pk_y0):]  # Get all PET states
suvr_values = jax.vmap(
    lambda y_pet, cc: PETAssignmentRule()(y_pet, w_suvr, build_c_suvr(cc), 0.0)[0]  
)(pet_states, central_conc)

# Calculate percent change from baseline
suvr_pcfb = ((suvr_values - pet_params['SUVR_0']) / pet_params['SUVR_0']) * 100

plt.plot(sol.ts/7, suvr_pcfb, 
         label='SUVR %CFB', 
         color='b',
         linewidth=2)
plt.xlabel('Time (weeks)', fontsize=10)
plt.ylabel('SUVR Change from Baseline (%)', fontsize=10)
plt.title('SUVR Percent Change from Baseline', fontsize=12, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 4: ARIA Events (bottom right)
plt.subplot(2, 2, 4)
w_aria = pet_w0
central_conc = sol.ys[:, y_indexes['C']] / pk_params['Vc']

# Calculate log hazard using the correct parameters
log_hazard = jax.vmap(
    lambda y_pet, cc, t: PETAssignmentRule()(
        y_pet,
        w_aria,
        build_c_suvr(cc),
        t
    )[1]
)(pet_states, central_conc, sol.ts)

# Create figure with two y-axes
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot log hazard on left y-axis (red)
line1 = ax1.plot(sol.ts/7, log_hazard, 
                 label='Log Hazard Rate', 
                 color='r',
                 linewidth=2)
ax1.set_xlabel('Time (weeks)', fontsize=10)
ax1.set_ylabel('Log Hazard Rate', color='r', fontsize=10)
ax1.tick_params(axis='y', labelcolor='r')

# Calculate hazard and cumulative incidence
h = jnp.exp(log_hazard)  # Convert from log hazard to hazard
delta_t = jnp.diff(sol.ts)
h_avg = (h[:-1] + h[1:]) / 2
integrand = h_avg * delta_t
H_t = jnp.concatenate([jnp.array([0.0]), jnp.nancumsum(integrand)])
S_t = jnp.exp(-H_t)  # Survival function
F_t = 1 - S_t  # Cumulative incidence function

# Plot cumulative probability on right y-axis (blue)
line2 = ax2.plot(sol.ts/7, F_t * 100, 
                 label='Cumulative ARIA Risk', 
                 color='b',
                 linewidth=2,
                 linestyle='--')
ax2.set_ylabel('Cumulative Risk (%)', color='b', fontsize=10)
ax2.tick_params(axis='y', labelcolor='b')

# Add grid and adjust limits
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=min(log_hazard), top=max(log_hazard))

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=10)

plt.title('ARIA Hazard and Cumulative Events', fontsize=12, pad=20)

plt.grid(True, alpha=0.3)
plt.show()
