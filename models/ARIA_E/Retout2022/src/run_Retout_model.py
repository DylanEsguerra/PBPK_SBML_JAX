import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jax import jit
from pathlib import Path
import sys
import importlib
from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from pk_module.pk_sbml import create_pk_model
from Retout2022.src.models.hazard.hazard_sbml import create_hazard_model
from Retout2022.src.models.Retout_modular_SBML import create_master_model, save_model

def load_parameters():
    """Load parameters and create dosing schedules"""
    pk_params = {
        "F": 0.494,
        "D1": 0.0821,
        "KA": 0.220,
        "CL": 0.336,
        "Vc": 3.52,
        "Q": 0.869,
        "Vp": 6.38,
    }

    pet_params = {
        "Ke0": 1740.0,
        "SLOP": 0.019,
        "power": 0.716,
        "SUVR_0": 1.8,
        "BSAPOE4_non": 8.7E-06,    
        "BSAPOE4_carrier": 3.56E-05,  
        "T50": 323,                 
        "gamma": 2.15,              
        "Emax": 6.05,              
        "EC50": 8.60
    }

    sc_doses = [
        (0.0, 1200.0),
        (28.0, 1200.0),
        (56.0, 1200.0),
        (84.0, 1200.0),
        (112.0, 1200.0),
        (140.0, 1200.0),
        (168.0, 1200.0),
        (196.0, 1200.0),
        (224.0, 1200.0),
        (252.0, 1200.0),
    ]
    
    # Add maintenance doses
    for week in range(40, 105, 4):
        day = week * 7.0
        sc_doses.append((day, 1200.0))

    iv_doses = [
        (0.0, 1.0),
        (28.0, 3.0),
        (56.0, 6.0),
    ]

    for week in range(12, 52, 4):
        day = week * 7.0
        iv_doses.append((day, 10.0))

    return {
        'pk': pk_params,
        'pet': pet_params,
        'sc_doses': sc_doses,
        'iv_doses': iv_doses
    }

def run_simulation():
    params = load_parameters()
    
    base_dir = Path(__file__).parent.parent
    sbml_dir = base_dir / "generated" / "sbml"
    jax_dir = base_dir / "generated" / "jax"
    figures_dir = base_dir / "generated" / "figures"
    
    # Create all necessary directories
    for dir_path in [jax_dir, figures_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating master SBML model...")
    master_document = create_master_model(params)
    save_model(master_document, sbml_dir)
    
    print("Generating JAX model...")
    model_data = parse.ParseSBMLFile(str(sbml_dir / "retout_sbml.xml"))
    GenerateModel(model_data, str(jax_dir / "retout_jax.py"))
    
    sys.path.append(str(jax_dir))
    import retout_jax
    importlib.reload(retout_jax)
    
    from retout_jax import ModelRollout
    
    t0 = 0.0
    t1 = 728.0  # 104 weeks
    dt0 = 0.01
    n_steps = int(t1/dt0)
    
    model = ModelRollout(deltaT=dt0)
    
    ys, ws, ts = model(n_steps=n_steps)
    
    times = jnp.linspace(t0, t1, n_steps)
    
    # Plot results
    plot_results(times, ys, ws, params, figures_dir)

def plot_results(times, ys, ws, params, figures_dir):
    y_indexes = {
        'A': 0, 
        'C': 1, 
        'Cp': 2,
        'Ce': 3, 
        'SUVR': 4,
        'ARIA_hazard': 5
    }

    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Plot 1: Central Concentration with doses (top left)
    ax1 = plt.subplot(2, 2, 1)
    line2 = plt.plot(times/7, ys[y_indexes['C']]/params['pk']['Vc'], 
                     label='Central Concentration (C)', 
                     color='m', 
                     linewidth=2)

    dose_mask = ws[0] > 0
    line1 = plt.plot(times[dose_mask]/7, ws[0][dose_mask]/10, '+', 
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
    plt.plot(times/7, ys[y_indexes['Ce']], 
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
    suvr_values = ws[2]
    suvr_pcfb = ((suvr_values - params['pet']['SUVR_0']) / params['pet']['SUVR_0']) * 100

    plt.plot(times/7, suvr_pcfb, 
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
    log_hazard = ws[3]

    # Create figure with two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot log hazard on left y-axis (red)
    line1 = ax1.plot(times/7, log_hazard, 
                     label='Log Hazard Rate', 
                     color='r',
                     linewidth=2)
    ax1.set_xlabel('Time (weeks)', fontsize=10)
    ax1.set_ylabel('Log Hazard Rate', color='r', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='r')

    # Calculate hazard and cumulative incidence
    h = jnp.exp(log_hazard)  # Convert from log hazard to hazard
    delta_t = jnp.diff(times)
    h_avg = (h[:-1] + h[1:]) / 2
    integrand = h_avg * delta_t
    H_t = jnp.concatenate([jnp.array([0.0]), jnp.nancumsum(integrand)])
    S_t = jnp.exp(-H_t)  # Survival function
    F_t = 1 - S_t  # Cumulative incidence function

    # Plot cumulative probability on right y-axis (blue)
    line2 = ax2.plot(times/7, F_t * 100, 
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

    plt.savefig(figures_dir / 'retout2022_simulation.png', 
                bbox_inches='tight', 
                dpi=300)
    
    plt.show()

if __name__ == "__main__":
    run_simulation() 