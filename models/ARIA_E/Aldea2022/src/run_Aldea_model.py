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
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# Now import using absolute paths from project root
from src.models.pk_sbml import create_pk_model
from src.models.vwd.vwd_sbml import create_vwd_model
from src.models.Aldea_modular_SBML import create_master_model, save_model

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

    vwd_params = {
        "alpha_removal": 0.126e-3,
        "k_repair": 12.4e-3,
        "A_beta0": 3.2,
        "VWD0": 0.0,
        "BGTS_max": 60.0,
        "EG50": 1.0,
        "pow": 3.72
    }

    sc_doses = [
        (0.0, 450.0),    # Day 1
        (28.0, 450.0),   # Week 4
        (56.0, 900.0),   # Week 8
        (84.0, 900.0),   # Week 12
        (112.0, 1200.0), # Week 16
        (140.0, 1200.0), # Week 20
    ]
    
    # Add maintenance doses (1200 mg) for 120 weeks
    for week in range(24, 121, 4):  # Changed from 101 to 121 to include full 120 weeks
        day = week * 7.0
        sc_doses.append((day, 1200.0))
    
    iv_doses = [
        (0.0, 0.0),
        (28.0, 0.0),
        (56.0, 0.0),
    ]
    
    for week in range(12, 121, 4):
        day = week * 7.0
        iv_doses.append((day, 0.0))

    return {
        'pk': pk_params,
        'vwd': vwd_params,
        'sc_doses': sc_doses,
        'iv_doses': iv_doses
    }

def run_simulation():
    params = load_parameters()
    
    base_dir = Path(__file__).parent
    sbml_dir = base_dir / "generated" / "sbml"
    jax_dir = base_dir / "generated" / "jax"
    jax_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating master SBML model...")
    master_document = create_master_model(params)
    save_model(master_document, sbml_dir)
    
    print("Generating JAX model...")
    model_data = parse.ParseSBMLFile(str(sbml_dir / "aldea_sbml.xml"))
    GenerateModel(model_data, str(jax_dir / "aldea_jax.py"))
    
    sys.path.append(str(jax_dir))
    import aldea_jax
    importlib.reload(aldea_jax)
    
    # Use diffrax with better solver parameters
    t0 = 0.0
    t1 = 840  # 120 weeks
    dt0 = 0.01
    
    solver = diffrax.Tsit5()  # Use Tsit5 solver for better stability
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 8400))  # Save more points
    
    ratefunc = aldea_jax.RateofSpeciesChange()
    assignment_rule = aldea_jax.AssignmentRule()

    def vector_field(t, y, args):
        w0, c = args
        w = assignment_rule(y, w0, c, t)
        dy = ratefunc(y, t, w, c)
        return dy
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=aldea_jax.y0,
        args=(aldea_jax.w0, aldea_jax.c),
        saveat=saveat,
        max_steps=1000000
    )
    
    times = sol.ts
    ys = sol.ys
    
    
    y_indexes = {
        'A': 0, 
        'C': 1, 
        'Cp': 2, 
        'A_beta': 3, 
        'VWD': 4
    }

    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(hspace=0.5)

    ax1 = plt.subplot(4, 1, 1)

    w_values = jax.vmap(lambda t, y: aldea_jax.AssignmentRule()(y, aldea_jax.w0, aldea_jax.c, t))(sol.ts, sol.ys)

    
    line2 = plt.plot(times/7, ys[:,y_indexes['C']]/params['pk']['Vc'], 
                     label='Central Concentration (C)', 
                     color='m', 
                     linewidth=2)

    dose_mask = w_values[:, 0] > 0  # Find where doses are applied
    line1 = plt.plot(times[dose_mask]/7, w_values[dose_mask, 0]/10, '+', 
                     color='orange', 
                     markersize=10, 
                     markeredgewidth=2, 
                     label='Dose administrations')

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

    plt.subplot(4, 1, 2)
    plt.plot(times/7, ys[:,y_indexes['A_beta']], 
             label='Local Amyloid (Aβ)', 
             color='g',
             linewidth=2)
    plt.xlabel('Time (weeks)', fontsize=10)
    plt.ylabel('Level', fontsize=10)
    plt.title('Local Amyloid', fontsize=12, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(times/7, ys[:,y_indexes['VWD']], 
             label='VWD', 
             color='b',
             linewidth=2)
    plt.xlabel('Time (weeks)', fontsize=10)
    plt.ylabel('Level', fontsize=10)
    plt.title('VWD', fontsize=12, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    bgts_values = w_values[:, 2]


    plt.subplot(4, 1, 4)
    plt.plot(times/7, bgts_values, 
             label='BGTS', 
             color='r',
             linewidth=2)
    plt.xlabel('Time (weeks)', fontsize=10)
    plt.ylabel('Level', fontsize=10)
    plt.title('BGTS', fontsize=12, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.ylim(0, 30)
    plt.show()

if __name__ == "__main__":
    run_simulation() 