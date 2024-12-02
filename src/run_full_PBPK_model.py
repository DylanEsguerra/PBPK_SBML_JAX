"""
Legacy simulation runner for the PBPK model using a unified approach.
This file handles the setup of coupled systems, running simulations, and plotting results
for the entire PBPK model.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import diffrax
from diffrax import Tsit5, ODETerm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
from sbmltoodejax import parse
from sbmltoodejax.modulegeneration import GenerateModel
from models.PBPK_full_SBML import create_pbpk_model, save_model, load_parameters

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def generate_jax_model():
    """Generate JAX model from combined PBPK SBML"""
    xml_path = Path("generated/sbml/pbpk_model.xml")
    jax_path = Path("generated/jax/pbpk_jax.py")
    
    # Always create a new SBML model
    print("\nCreating new SBML model...")
    params_path = Path("parameters/pbpk_parameters.csv")
    params, params_with_units = load_parameters(params_path)
    document = create_pbpk_model(params, params_with_units)
    save_model(document, str(xml_path))
    
    print(f"\nChecking SBML file at {xml_path}")
    if not xml_path.exists():
        print("ERROR: SBML file not found!")
        return
    
    # Parse SBML and generate JAX model
    print("Parsing PBPK SBML...")
    model_data = parse.ParseSBMLFile(str(xml_path))
    
    
    print("\nGenerating JAX model...")
    GenerateModel(model_data, str(jax_path))

def run_simulation():
    # Generate JAX model if needed
    generate_jax_model()
    
    # Import the generated model
    from generated.jax.pbpk_jax import (
        RateofSpeciesChange, AssignmentRule, y0, w0, t0, c
    )

    rate_of_change = RateofSpeciesChange()
    assignment_rule = AssignmentRule()
    
    @jit
    def ode_func(t, y, args):
        w, c = args
        
        # Update w using the assignment rule
        w = assignment_rule(y, w, c, t)
        
        # Calculate the rate of change
        dy_dt = rate_of_change(y, t, w, c)
        
        return dy_dt

    # Simulation parameters
    t1 = 2000  # 2000 hours
    dt = 0.001
    n_steps = 2000
    # Create diffrax solver
    term = ODETerm(ode_func)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))

    # Solve the ODE system
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=(w0, c),
        saveat=saveat,
        max_steps=1000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8)
    )
    
    return sol

    

def plot_results(sol):
    """Plot results from the combined PBPK model simulation"""
    # Create figure for non-typical compartments
    fig1 = plt.figure(figsize=(15, 12))
    gs1 = fig1.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Blood compartments (indices 0-2)
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.semilogy(sol.ts, sol.ys[:, 0], label='Plasma', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 1], label='Blood Cells', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 2], label='Lymph Node', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Blood Compartments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lung compartments (indices 3-8)
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.semilogy(sol.ts, sol.ys[:, 3], label='Plasma', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 4], label='Blood Cells', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 5], label='ISF', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 6], label='E Unbound', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 7], label='E Bound', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 8], label='FcRn Free', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('Lung Compartments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Brain compartments (indices 9-13)
    ax3 = fig1.add_subplot(gs1[1, 0])
    ax3.semilogy(sol.ts, sol.ys[:, 9], label='Plasma', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 10], label='BBB Unbound', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 11], label='BBB Bound', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 12], label='Brain ISF', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Brain Compartments')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # CSF compartments (indices 14-19)
    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.semilogy(sol.ts, sol.ys[:, 13], label='BCSFB Unbound', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 14], label='BCSFB Bound', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 15], label='Lateral Ventricle', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 16], label='Third/Fourth Ventricle', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 17], label='Cisterna Magna', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 18], label='Subarachnoid Space', linewidth=2)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Concentration')
    ax4.set_title('CSF Compartments')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Liver compartments (indices 20-25)
    ax5 = fig1.add_subplot(gs1[2, 0])
    ax5.semilogy(sol.ts, sol.ys[:, 19], label='Plasma', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 20], label='Blood Cells', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 21], label='ISF', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 22], label='E Unbound', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 23], label='E Bound', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 24], label='FcRn Free', linewidth=2)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Concentration')
    ax5.set_title('Liver Compartments')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Save non-typical tissues figure
    fig1.savefig('full_model_concentration_plots_nontypical.png', dpi=300, bbox_inches='tight')

    # Create figure for typical compartments
    fig2 = plt.figure(figsize=(15, 12))
    gs2 = fig2.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    typical_organs = [
        ('Heart', 25),     # C_p_heart
        ('Muscle', 31),    # C_p_muscle
        ('Kidney', 37),    # C_p_kidney
        ('Skin', 43),      # C_p_skin
        ('Fat', 49),       # C_p_fat
        ('Marrow', 55),    # C_p_marrow
        ('Thymus', 61),    # C_p_thymus
        ('SI', 67),        # C_p_SI
        ('LI', 73),        # C_p_LI
        ('Spleen', 79),    # C_p_spleen
        ('Pancreas', 85),  # C_p_pancreas
        ('Other', 91)      # C_p_other
    ]

    # Create subplots for each typical organ
    lines = []  # Store lines for the legend
    labels = ['Plasma', 'Blood Cells', 'ISF', 'E Unbound', 'E Bound', 'FcRn Free']
    
    for i, (organ, start_idx) in enumerate(typical_organs):
        row = i // 3
        col = i % 3
        
        ax = fig2.add_subplot(gs2[row, col])
        # Plot lines and store them for the legend
        plot_lines = [
            ax.semilogy(sol.ts, sol.ys[:, start_idx+j], linewidth=2)[0]
            for j in range(6)
        ]
        if i == 0:  # Only store lines from the first plot for legend
            lines = plot_lines
            
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{organ} Compartments')
        ax.grid(True, alpha=0.3)

    # Create a single legend outside the subplots
    fig2.legend(lines, labels, 
               loc='center right', 
               bbox_to_anchor=(0.98, 0.5),
               title='Compartments')
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    fig2.subplots_adjust(right=0.85)

    # Save typical tissues figure
    fig2.savefig('full_model_concentration_plots_typical.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print final concentrations with proper labels
    print("\nFinal Concentrations:")
    print("-" * 30)
    compartments = [
        'Blood:', 
        '  Plasma (C_p)', '  Blood Cells (C_bc)', '  Lymph Node (C_ln)',
        'Lung:', 
        '  Plasma (C_p_lung)', '  Blood Cells (C_bc_lung)', '  ISF (C_is_lung)',
        '  E Unbound (C_e_unbound_lung)', '  E Bound (C_e_bound_lung)', '  FcRn Free (FcRn_free_lung)',
        'Brain:', 
        '  Plasma (C_p_brain)', '  BBB Unbound (C_BBB_unbound_brain)', '  BBB Bound (C_BBB_bound_brain)',
        '  ISF (C_is_brain)', '  BCSFB Unbound (C_BCSFB_unbound_brain)', '  BCSFB Bound (C_BCSFB_bound_brain)',
        'CSF:',
        '  Lateral Ventricle (C_LV_brain)', '  Third/Fourth Ventricle (C_TFV_brain)',
        '  Cisterna Magna (C_CM_brain)', '  Subarachnoid Space (C_SAS_brain)',
        'Liver:',
        '  Plasma (C_p_liver)', '  Blood Cells (C_bc_liver)', '  ISF (C_is_liver)',
        '  E Unbound (C_e_unbound_liver)', '  E Bound (C_e_bound_liver)', '  FcRn Free (FcRn_free_liver)',
        'Heart:',
        '  Plasma (C_p_heart)', '  Blood Cells (C_bc_heart)', '  ISF (C_is_heart)',
        '  E Unbound (C_e_unbound_heart)', '  E Bound (C_e_bound_heart)', '  FcRn Free (FcRn_free_heart)',
        'Muscle:',
        '  Plasma (C_p_muscle)', '  Blood Cells (C_bc_muscle)', '  ISF (C_is_muscle)',
        '  E Unbound (C_e_unbound_muscle)', '  E Bound (C_e_bound_muscle)', 
        '  FcRn Free (FcRn_free_muscle)'
    ]

    current_index = -1
    for comp in compartments:
        if comp.endswith(':'):
            print(f"\n{comp}")
        else:
            current_index += 1
            print(f"{comp:<45}: {sol.ys[-1, current_index]:.6f}")

if __name__ == "__main__":
    sol = run_simulation()
    plot_results(sol) 