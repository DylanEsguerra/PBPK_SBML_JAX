"""
Simulation runner for the combined Brain-CSF system using the master model approach.
This version uses SBML's hierarchical model composition to create a single unified model,
then converts it to JAX using sbmltoodejax.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import diffrax
from diffrax import Tsit5, ODETerm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
import importlib
from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse
import libsbml

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from src.models.PBPK_modular_SBML import create_master_model


def run_simulation():
    # Load parameters from CSV
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')

    # Convert DataFrame to dictionary for easier parameter access
    params_dict = dict(zip(params_df['name'], params_df['value']))
    
    # Create master model
    master_document = create_master_model(params_dict)
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "master_sbml.xml"
    
    libsbml.writeSBMLToFile(master_document, str(master_path))
    print(f"Master model saved to {master_path}")
    
    # Generate JAX model from master SBML
    jax_dir = Path("generated/jax")
    jax_dir.mkdir(parents=True, exist_ok=True)
    jax_path = jax_dir / "master_jax.py"
    
    model_data = parse.ParseSBMLFile(str(master_path))
    GenerateModel(model_data, str(jax_path))
    
    # Import generated JAX model
    master_module = importlib.import_module("generated.jax.master_jax")
    importlib.reload(master_module)
    
    # Set up simulation parameters
    t0 = 0.0
    t1 = 2000
    dt = 0.001
    n_steps = 2000
    
    # Create solver
    term = ODETerm(lambda t, y, args: master_module.RateofSpeciesChange()(y, t, args[0], args[1]))
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))
    
    # Solve system using initial conditions from the master model
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=master_module.y0,
        args=(None, master_module.c),
        saveat=saveat,
        max_steps=1000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8)
    )
    
    return sol, master_module

def plot_results(sol, master_module):
    """Plot results from the master model simulation in the style of run_PBPK.py"""
    # Create figures directory if it doesn't exist
    figures_dir = Path("generated/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure for non-typical compartments
    fig1 = plt.figure(figsize=(15, 12))
    gs1 = fig1.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Blood compartments
    ax1 = fig1.add_subplot(gs1[0, 0])
    for species, label in zip(['C_p', 'C_bc', 'C_ln'], ['Plasma', 'Blood Cells', 'Lymph Node']):
        ax1.semilogy(sol.ts, sol.ys[:, master_module.y_indexes[species]], label=label, linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Blood Compartments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lung compartments
    ax2 = fig1.add_subplot(gs1[0, 1])
    lung_species = ['C_p_lung', 'C_bc_lung', 'C_is_lung', 'C_e_unbound_lung', 'C_e_bound_lung', 'FcRn_free_lung']
    lung_labels = ['Plasma', 'Blood Cells', 'ISF', 'E Unbound', 'E Bound', 'FcRn Free']
    for species, label in zip(lung_species, lung_labels):
        ax2.semilogy(sol.ts, sol.ys[:, master_module.y_indexes[species]], label=label, linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('Lung Compartments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Brain compartments
    ax3 = fig1.add_subplot(gs1[1, 0])
    brain_species = ['C_p_brain', 'C_BBB_unbound_brain', 'C_BBB_bound_brain', 'C_is_brain']
    brain_labels = ['Plasma', 'BBB Unbound', 'BBB Bound', 'Brain ISF']
    for species, label in zip(brain_species, brain_labels):
        ax3.semilogy(sol.ts, sol.ys[:, master_module.y_indexes[species]], label=label, linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Brain Compartments')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # CSF compartments
    ax4 = fig1.add_subplot(gs1[1, 1])
    csf_species = ['C_BCSFB_unbound_brain', 'C_BCSFB_bound_brain', 'C_LV_brain', 
                  'C_TFV_brain', 'C_CM_brain', 'C_SAS_brain']
    csf_labels = ['BCSFB Unbound', 'BCSFB Bound', 'Lateral Ventricle',
                 'Third/Fourth Ventricle', 'Cisterna Magna', 'Subarachnoid Space']
    for species, label in zip(csf_species, csf_labels):
        ax4.semilogy(sol.ts, sol.ys[:, master_module.y_indexes[species]], label=label, linewidth=2)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Concentration')
    ax4.set_title('CSF Compartments')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Liver compartments
    ax5 = fig1.add_subplot(gs1[2, 0])
    liver_species = ['C_p_liver', 'C_bc_liver', 'C_is_liver', 
                    'C_e_unbound_liver', 'C_e_bound_liver', 'FcRn_free_liver']
    liver_labels = ['Plasma', 'Blood Cells', 'ISF', 'E Unbound', 'E Bound', 'FcRn Free']
    for species, label in zip(liver_species, liver_labels):
        ax5.semilogy(sol.ts, sol.ys[:, master_module.y_indexes[species]], label=label, linewidth=2)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Concentration')
    ax5.set_title('Liver Compartments')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Save non-typical tissues figure
    fig1.savefig(figures_dir / 'modular_model_concentration_plots_nontypical.png', 
                 dpi=300, bbox_inches='tight')

    # Create figure for typical compartments
    fig2 = plt.figure(figsize=(15, 12))
    gs2 = fig2.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    typical_organs = [
        ('Heart', 'heart'), ('Muscle', 'muscle'), ('Kidney', 'kidney'),
        ('Skin', 'skin'), ('Fat', 'fat'), ('Marrow', 'marrow'),
        ('Thymus', 'thymus'), ('SI', 'SI'), ('LI', 'LI'),
        ('Spleen', 'spleen'), ('Pancreas', 'pancreas'), ('Other', 'other')
    ]

    # Create subplots for each typical organ
    lines = []  # Store lines for the legend
    labels = ['Plasma', 'Blood Cells', 'ISF', 'E Unbound', 'E Bound', 'FcRn Free']
    
    for i, (organ_name, organ_id) in enumerate(typical_organs):
        row = i // 3
        col = i % 3
        
        ax = fig2.add_subplot(gs2[row, col])
        species_list = [
            f'C_p_{organ_id}', f'C_bc_{organ_id}', f'C_is_{organ_id}',
            f'C_e_unbound_{organ_id}', f'C_e_bound_{organ_id}', f'FcRn_free_{organ_id}'
        ]
        
        # Plot lines and store them for the legend
        plot_lines = []
        for species, label in zip(species_list, labels):
            line = ax.semilogy(sol.ts, sol.ys[:, master_module.y_indexes[species]], linewidth=2)[0]
            plot_lines.append(line)
            
        if i == 0:  # Only store lines from the first plot for legend
            lines = plot_lines
            
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{organ_name} Compartments')
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
    fig2.savefig(figures_dir / 'modular_model_concentration_plots_typical.png', 
                 dpi=300, bbox_inches='tight')
    
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
        '  E Unbound (C_e_unbound_liver)', '  E Bound (C_e_bound_liver)', '  FcRn Free (FcRn_free_liver)'
    ]

    # Add typical tissues to compartments list
    for organ_name, organ_id in typical_organs:
        compartments.extend([
            f'{organ_name}:',
            f'  Plasma (C_p_{organ_id})', f'  Blood Cells (C_bc_{organ_id})', f'  ISF (C_is_{organ_id})',
            f'  E Unbound (C_e_unbound_{organ_id})', f'  E Bound (C_e_bound_{organ_id})', 
            f'  FcRn Free (FcRn_free_{organ_id})'
        ])

    for comp in compartments:
        if comp.endswith(':'):
            print(f"\n{comp}")
        else:
            species = comp.split('(')[1].rstrip(')')
            idx = master_module.y_indexes[species]
            print(f"{comp:<45}: {sol.ys[-1, idx]:.6f}")

if __name__ == "__main__":
    sol, master_module = run_simulation()
    plot_results(sol, master_module) 