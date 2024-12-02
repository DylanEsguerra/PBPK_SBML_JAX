import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([14117.999999999989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000533074, 0.0, 0.0, 0.0, 0.0, 0.0, 8.51922e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0074730000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 8.27012e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00084694, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003352886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0025308559999999997, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5992219999999998e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 9.61526e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00013650680000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 5.53002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5806760000000002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001210626])
y_indexes = {'C_p': 0, 'C_bc': 1, 'C_ln': 2, 'C_p_lung': 3, 'C_bc_lung': 4, 'C_e_unbound_lung': 5, 'C_e_bound_lung': 6, 'C_is_lung': 7, 'FcRn_free_lung': 8, 'C_p_brain': 9, 'C_BBB_unbound_brain': 10, 'C_BBB_bound_brain': 11, 'C_is_brain': 12, 'C_bc_brain': 13, 'C_BCSFB_unbound_brain': 14, 'C_BCSFB_bound_brain': 15, 'C_LV_brain': 16, 'C_TFV_brain': 17, 'C_CM_brain': 18, 'C_SAS_brain': 19, 'C_p_liver': 20, 'C_bc_liver': 21, 'C_is_liver': 22, 'C_e_unbound_liver': 23, 'C_e_bound_liver': 24, 'FcRn_free_liver': 25, 'C_p_heart': 26, 'C_bc_heart': 27, 'C_is_heart': 28, 'C_e_unbound_heart': 29, 'C_e_bound_heart': 30, 'FcRn_free_heart': 31, 'C_p_muscle': 32, 'C_bc_muscle': 33, 'C_is_muscle': 34, 'C_e_unbound_muscle': 35, 'C_e_bound_muscle': 36, 'FcRn_free_muscle': 37, 'C_p_kidney': 38, 'C_bc_kidney': 39, 'C_is_kidney': 40, 'C_e_unbound_kidney': 41, 'C_e_bound_kidney': 42, 'FcRn_free_kidney': 43, 'C_p_skin': 44, 'C_bc_skin': 45, 'C_is_skin': 46, 'C_e_unbound_skin': 47, 'C_e_bound_skin': 48, 'FcRn_free_skin': 49, 'C_p_fat': 50, 'C_bc_fat': 51, 'C_is_fat': 52, 'C_e_unbound_fat': 53, 'C_e_bound_fat': 54, 'FcRn_free_fat': 55, 'C_p_marrow': 56, 'C_bc_marrow': 57, 'C_is_marrow': 58, 'C_e_unbound_marrow': 59, 'C_e_bound_marrow': 60, 'FcRn_free_marrow': 61, 'C_p_thymus': 62, 'C_bc_thymus': 63, 'C_is_thymus': 64, 'C_e_unbound_thymus': 65, 'C_e_bound_thymus': 66, 'FcRn_free_thymus': 67, 'C_p_SI': 68, 'C_bc_SI': 69, 'C_is_SI': 70, 'C_e_unbound_SI': 71, 'C_e_bound_SI': 72, 'FcRn_free_SI': 73, 'C_p_LI': 74, 'C_bc_LI': 75, 'C_is_LI': 76, 'C_e_unbound_LI': 77, 'C_e_bound_LI': 78, 'FcRn_free_LI': 79, 'C_p_spleen': 80, 'C_bc_spleen': 81, 'C_is_spleen': 82, 'C_e_unbound_spleen': 83, 'C_e_bound_spleen': 84, 'FcRn_free_spleen': 85, 'C_p_pancreas': 86, 'C_bc_pancreas': 87, 'C_is_pancreas': 88, 'C_e_unbound_pancreas': 89, 'C_e_bound_pancreas': 90, 'FcRn_free_pancreas': 91, 'C_p_other': 92, 'C_bc_other': 93, 'C_is_other': 94, 'C_e_unbound_other': 95, 'C_e_bound_other': 96, 'FcRn_free_other': 97}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 8343.0, 2591.0, 32402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 36402.0, 21.0, 10.5, 6350.0, 148920.0, 27410.0, 9520.0, 6830.0, 2120.0, 26530.0, 10820.0, 10130.0, 10530.0, 2500.0, 289.0, 5190.0, 4520.0, 29810.0, 364.0, 16.0, 65.0, 73.0, 67.0, 5.0, 1.0, 23.0, 17.0, 25.0, 26.0, 13.0, 6.0, 26.0, 11.0, 500.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.51631477927063, 0.0, 0.0, 1000.0, 55.0, 45.0, 5.0, 559000000.0, 23.9, 26.6, 0.55, 0.715, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 31.9, 0.1, 261.0, 26.1, 7.25, 10.5, 0.3, 0.95, 1.0, 0.95, 0.9974, 0.1, 22.5, 22.5, 7.5, 90.0, 0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2143.0, 183.0, 149.0, 10.7, 0.55, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.95, 0.55, 0.95, 0.55, 0.9, 0.55, 0.95, 0.55, 0.95, 0.55, 0.95, 0.55, 0.9, 0.55, 0.9, 0.55, 0.95, 0.55, 0.85, 0.55, 0.9, 0.55, 0.95, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 31.9, 0.1, 0.1, 261.0, 26.1, 0.1, 0.1, 22.5, 22.5, 7.5, 90.0, 2143.0, 183.0, 149.0, 10.7, 341.0, 13.1, 10.8, 1.71, 30078.0, 662.0, 541.0, 150.0, 332.0, 18.2, 14.9, 1.66, 3408.0, 127.0, 104.0, 17.0, 13465.0, 148.0, 121.0, 67.3, 10165.0, 224.0, 183.0, 50.8, 6.41, 0.353, 0.288, 0.0321, 385.0, 6.15, 5.03, 1.93, 548.0, 8.74, 7.15, 2.74, 221.0, 26.8, 21.9, 1.11, 104.0, 5.7, 4.66, 0.518, 4852.0, 204.0, 167.0, 24.3]) 
c_indexes = {'Vp': 0, 'Vbc': 1, 'Vlymphnode': 2, 'Q_p_heart': 3, 'Q_p_lung': 4, 'Q_p_muscle': 5, 'Q_p_skin': 6, 'Q_p_fat': 7, 'Q_p_marrow': 8, 'Q_p_kidney': 9, 'Q_p_liver': 10, 'Q_p_SI': 11, 'Q_p_LI': 12, 'Q_p_pancreas': 13, 'Q_p_thymus': 14, 'Q_p_spleen': 15, 'Q_p_other': 16, 'Q_p_brain': 17, 'Q_CSF_brain': 18, 'Q_ECF_brain': 19, 'Q_bc_heart': 20, 'Q_bc_lung': 21, 'Q_bc_muscle': 22, 'Q_bc_skin': 23, 'Q_bc_fat': 24, 'Q_bc_marrow': 25, 'Q_bc_kidney': 26, 'Q_bc_liver': 27, 'Q_bc_SI': 28, 'Q_bc_LI': 29, 'Q_bc_pancreas': 30, 'Q_bc_thymus': 31, 'Q_bc_spleen': 32, 'Q_bc_other': 33, 'Q_bc_brain': 34, 'L_lung': 35, 'L_heart': 36, 'L_kidney': 37, 'L_brain': 38, 'L_muscle': 39, 'L_marrow': 40, 'L_thymus': 41, 'L_skin': 42, 'L_fat': 43, 'L_SI': 44, 'L_LI': 45, 'L_spleen': 46, 'L_pancreas': 47, 'L_liver': 48, 'L_other': 49, 'L_LN': 50, 'sigma_L_lung': 51, 'sigma_L_heart': 52, 'sigma_L_kidney': 53, 'sigma_L_brain_ISF': 54, 'sigma_L_muscle': 55, 'sigma_L_marrow': 56, 'sigma_L_thymus': 57, 'sigma_L_skin': 58, 'sigma_L_fat': 59, 'sigma_L_SI': 60, 'sigma_L_LI': 61, 'sigma_L_spleen': 62, 'sigma_L_pancreas': 63, 'sigma_L_liver': 64, 'sigma_L_other': 65, 'sigma_L_SAS': 66, 'C_p_0': 67, 'C_bc_0': 68, 'C_ln_0': 69, 'Vp_lung': 160, 'VBC_lung': 161, 'VIS_lung': 162, 'VES_lung': 163, 'kon_FcRn': 74, 'koff_FcRn': 75, 'kdeg': 76, 'CLup_lung': 77, 'FR': 78, 'sigma_V_lung': 79, 'C_p_lung_0': 80, 'C_bc_lung_0': 81, 'C_is_lung_0': 82, 'C_e_unbound_lung_0': 83, 'C_e_bound_lung_0': 84, 'FcRn_free_lung_0': 85, 'Vp_brain': 86, 'VBBB_brain': 87, 'VIS_brain': 88, 'VBC_brain': 89, 'V_ES_brain': 90, 'Q_ISF_brain': 91, 'CLup_brain': 92, 'f_BBB': 93, 'FcRn_free_BBB': 94, 'sigma_V_BBB': 95, 'sigma_V_BCSFB': 96, 'V_BCSFB_brain': 97, 'V_LV_brain': 98, 'V_TFV_brain': 99, 'V_CM_brain': 100, 'V_SAS_brain': 101, 'f_BCSFB': 102, 'FcRn_free_BCSFB': 103, 'f_LV': 104, 'C_BCSFB_unbound_brain_0': 105, 'C_BCSFB_bound_brain_0': 106, 'C_LV_brain_0': 107, 'C_TFV_brain_0': 108, 'C_CM_brain_0': 109, 'C_SAS_brain_0': 110, 'C_p_brain_0': 111, 'C_is_brain_0': 112, 'Vp_liver': 175, 'VBC_liver': 176, 'VIS_liver': 177, 'VES_liver': 178, 'CLup_liver': 117, 'sigma_V_liver': 118, 'C_p_liver_0': 119, 'C_bc_liver_0': 120, 'C_is_liver_0': 121, 'C_e_unbound_liver_0': 122, 'C_e_bound_liver_0': 123, 'FcRn_free_liver_0': 124, 'C_p_spleen_0': 125, 'C_bc_spleen_0': 126, 'C_p_pancreas_0': 127, 'C_bc_pancreas_0': 128, 'C_p_SI_0': 129, 'C_bc_SI_0': 130, 'C_p_LI_0': 131, 'C_bc_LI_0': 132, 'CLup_heart': 133, 'sigma_V_heart': 134, 'CLup_muscle': 135, 'sigma_V_muscle': 136, 'CLup_kidney': 137, 'sigma_V_kidney': 138, 'CLup_skin': 139, 'sigma_V_skin': 140, 'CLup_fat': 141, 'sigma_V_fat': 142, 'CLup_marrow': 143, 'sigma_V_marrow': 144, 'CLup_thymus': 145, 'sigma_V_thymus': 146, 'CLup_SI': 147, 'sigma_V_SI': 148, 'CLup_LI': 149, 'sigma_V_LI': 150, 'CLup_spleen': 151, 'sigma_V_spleen': 152, 'CLup_pancreas': 153, 'sigma_V_pancreas': 154, 'CLup_other': 155, 'sigma_V_other': 156, 'plasma': 157, 'blood_cells': 158, 'lymph_node': 159, 'brain_plasma': 164, 'BBB_unbound': 165, 'BBB_bound': 166, 'brain_ISF': 167, 'brain_blood_cells': 168, 'BCSFB_unbound': 169, 'BCSFB_bound': 170, 'LV': 171, 'TFV': 172, 'CM': 173, 'SAS': 174, 'Vp_heart': 179, 'VBC_heart': 180, 'VIS_heart': 181, 'VES_heart': 182, 'Vp_muscle': 183, 'VBC_muscle': 184, 'VIS_muscle': 185, 'VES_muscle': 186, 'Vp_kidney': 187, 'VBC_kidney': 188, 'VIS_kidney': 189, 'VES_kidney': 190, 'Vp_skin': 191, 'VBC_skin': 192, 'VIS_skin': 193, 'VES_skin': 194, 'Vp_fat': 195, 'VBC_fat': 196, 'VIS_fat': 197, 'VES_fat': 198, 'Vp_marrow': 199, 'VBC_marrow': 200, 'VIS_marrow': 201, 'VES_marrow': 202, 'Vp_thymus': 203, 'VBC_thymus': 204, 'VIS_thymus': 205, 'VES_thymus': 206, 'Vp_SI': 207, 'VBC_SI': 208, 'VIS_SI': 209, 'VES_SI': 210, 'Vp_LI': 211, 'VBC_LI': 212, 'VIS_LI': 213, 'VES_LI': 214, 'Vp_spleen': 215, 'VBC_spleen': 216, 'VIS_spleen': 217, 'VES_spleen': 218, 'Vp_pancreas': 219, 'VBC_pancreas': 220, 'VIS_pancreas': 221, 'VES_pancreas': 222, 'Vp_other': 223, 'VBC_other': 224, 'VIS_other': 225, 'VES_other': 226}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p(y, w, c, t), self.RateC_bc(y, w, c, t), self.RateC_ln(y, w, c, t), self.RateC_p_lung(y, w, c, t), self.RateC_bc_lung(y, w, c, t), self.RateC_e_unbound_lung(y, w, c, t), self.RateC_e_bound_lung(y, w, c, t), self.RateC_is_lung(y, w, c, t), self.RateFcRn_free_lung(y, w, c, t), self.RateC_p_brain(y, w, c, t), self.RateC_BBB_unbound_brain(y, w, c, t), self.RateC_BBB_bound_brain(y, w, c, t), self.RateC_is_brain(y, w, c, t), self.RateC_bc_brain(y, w, c, t), self.RateC_BCSFB_unbound_brain(y, w, c, t), self.RateC_BCSFB_bound_brain(y, w, c, t), self.RateC_LV_brain(y, w, c, t), self.RateC_TFV_brain(y, w, c, t), self.RateC_CM_brain(y, w, c, t), self.RateC_SAS_brain(y, w, c, t), self.RateC_p_liver(y, w, c, t), self.RateC_bc_liver(y, w, c, t), self.RateC_is_liver(y, w, c, t), self.RateC_e_unbound_liver(y, w, c, t), self.RateC_e_bound_liver(y, w, c, t), self.RateFcRn_free_liver(y, w, c, t), self.RateC_p_heart(y, w, c, t), self.RateC_bc_heart(y, w, c, t), self.RateC_is_heart(y, w, c, t), self.RateC_e_unbound_heart(y, w, c, t), self.RateC_e_bound_heart(y, w, c, t), self.RateFcRn_free_heart(y, w, c, t), self.RateC_p_muscle(y, w, c, t), self.RateC_bc_muscle(y, w, c, t), self.RateC_is_muscle(y, w, c, t), self.RateC_e_unbound_muscle(y, w, c, t), self.RateC_e_bound_muscle(y, w, c, t), self.RateFcRn_free_muscle(y, w, c, t), self.RateC_p_kidney(y, w, c, t), self.RateC_bc_kidney(y, w, c, t), self.RateC_is_kidney(y, w, c, t), self.RateC_e_unbound_kidney(y, w, c, t), self.RateC_e_bound_kidney(y, w, c, t), self.RateFcRn_free_kidney(y, w, c, t), self.RateC_p_skin(y, w, c, t), self.RateC_bc_skin(y, w, c, t), self.RateC_is_skin(y, w, c, t), self.RateC_e_unbound_skin(y, w, c, t), self.RateC_e_bound_skin(y, w, c, t), self.RateFcRn_free_skin(y, w, c, t), self.RateC_p_fat(y, w, c, t), self.RateC_bc_fat(y, w, c, t), self.RateC_is_fat(y, w, c, t), self.RateC_e_unbound_fat(y, w, c, t), self.RateC_e_bound_fat(y, w, c, t), self.RateFcRn_free_fat(y, w, c, t), self.RateC_p_marrow(y, w, c, t), self.RateC_bc_marrow(y, w, c, t), self.RateC_is_marrow(y, w, c, t), self.RateC_e_unbound_marrow(y, w, c, t), self.RateC_e_bound_marrow(y, w, c, t), self.RateFcRn_free_marrow(y, w, c, t), self.RateC_p_thymus(y, w, c, t), self.RateC_bc_thymus(y, w, c, t), self.RateC_is_thymus(y, w, c, t), self.RateC_e_unbound_thymus(y, w, c, t), self.RateC_e_bound_thymus(y, w, c, t), self.RateFcRn_free_thymus(y, w, c, t), self.RateC_p_SI(y, w, c, t), self.RateC_bc_SI(y, w, c, t), self.RateC_is_SI(y, w, c, t), self.RateC_e_unbound_SI(y, w, c, t), self.RateC_e_bound_SI(y, w, c, t), self.RateFcRn_free_SI(y, w, c, t), self.RateC_p_LI(y, w, c, t), self.RateC_bc_LI(y, w, c, t), self.RateC_is_LI(y, w, c, t), self.RateC_e_unbound_LI(y, w, c, t), self.RateC_e_bound_LI(y, w, c, t), self.RateFcRn_free_LI(y, w, c, t), self.RateC_p_spleen(y, w, c, t), self.RateC_bc_spleen(y, w, c, t), self.RateC_is_spleen(y, w, c, t), self.RateC_e_unbound_spleen(y, w, c, t), self.RateC_e_bound_spleen(y, w, c, t), self.RateFcRn_free_spleen(y, w, c, t), self.RateC_p_pancreas(y, w, c, t), self.RateC_bc_pancreas(y, w, c, t), self.RateC_is_pancreas(y, w, c, t), self.RateC_e_unbound_pancreas(y, w, c, t), self.RateC_e_bound_pancreas(y, w, c, t), self.RateFcRn_free_pancreas(y, w, c, t), self.RateC_p_other(y, w, c, t), self.RateC_bc_other(y, w, c, t), self.RateC_is_other(y, w, c, t), self.RateC_e_unbound_other(y, w, c, t), self.RateC_e_bound_other(y, w, c, t), self.RateFcRn_free_other(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p(self, y, w, c, t):
		return (1 / c[0]) * (-(c[4] + c[35]) * (y[0]/3126.0) + (c[3] - c[36]) * (y[26]/341.0) + (c[9] - c[37]) * (y[38]/332.0) + (c[17] - c[38]) * (y[9]/31.9) + (c[5] - c[39]) * (y[32]/30078.0) + (c[8] - c[40]) * (y[56]/10165.0) + (c[14] - c[41]) * (y[62]/6.41) + (c[6] - c[42]) * (y[44]/3408.0) + (c[7] - c[43]) * (y[50]/13465.0) + ((c[11] - c[44]) + (c[12] - c[45]) + (c[15] - c[46]) + (c[13] - c[47]) + (c[10] - c[48])) * (y[20]/2143.0) + (c[16] - c[49]) * (y[92]/4852.0) + c[50] * (y[2]/274.0))

	def RateC_bc(self, y, w, c, t):
		return (1 / c[1]) * (-c[21] * (y[1]/2558.0) + c[20] * (y[27]/13.1) + c[26] * (y[39]/18.2) + c[34] * (y[13]/26.1) + c[22] * (y[33]/662.0) + c[25] * (y[57]/224.0) + c[31] * (y[63]/0.353) + c[23] * (y[45]/127.0) + c[24] * (y[51]/148.0) + (c[28] + c[29] + c[32] + c[30] + c[27]) * (y[21]/183.0) + c[33] * (y[93]/204.0))

	def RateC_ln(self, y, w, c, t):
		return (1 / c[2]) * ((1 - c[51]) * c[35] * (y[7]/45.0) + (1 - c[52]) * c[36] * (y[28]/10.8) + (1 - c[53]) * c[37] * (y[40]/14.9) + (1 - c[66]) * c[18] * (y[19]/90.0) + (1 - c[54]) * c[19] * (y[12]/261.0) + (1 - c[55]) * c[39] * (y[34]/541.0) + (1 - c[56]) * c[40] * (y[58]/183.0) + (1 - c[57]) * c[41] * (y[64]/0.288) + (1 - c[58]) * c[42] * (y[46]/104.0) + (1 - c[59]) * c[43] * (y[52]/121.0) + (1 - c[60]) * c[44] * (y[70]/5.03) + (1 - c[61]) * c[45] * (y[76]/7.15) + (1 - c[62]) * c[46] * (y[82]/21.9) + (1 - c[63]) * c[47] * (y[88]/4.66) + (1 - c[64]) * c[48] * (y[22]/149.0) + (1 - c[65]) * c[49] * (y[94]/167.0) - c[50] * (y[2]/274.0))

	def RateC_p_lung(self, y, w, c, t):
		return (1 / c[160]) * ((c[4] * (y[0]/3126.0) - (c[4] - c[35]) * (y[3]/1000.0) - (1 - c[79]) * c[35] * (y[3]/1000.0) - c[77] * (y[3]/1000.0)) + c[77] * c[78] * (y[6]/5.0))

	def RateC_bc_lung(self, y, w, c, t):
		return (1 / c[161]) * (c[21] * (y[1]/2558.0) - c[21] * (y[4]/55.0))

	def RateC_e_unbound_lung(self, y, w, c, t):
		return (1 / c[163]) * ((c[77] * ((y[3]/1000.0) + (y[7]/45.0)) - c[163] * c[74] * (y[5]/5.0) * (y[8]/5.0)) + c[163] * c[75] * (y[6]/5.0) - c[76] * (y[5]/5.0) * c[163])

	def RateC_e_bound_lung(self, y, w, c, t):
		return (1 / c[163]) * (c[163] * c[74] * (y[5]/5.0) * (y[8]/5.0) - c[163] * c[75] * (y[6]/5.0) - c[77] * (y[6]/5.0))

	def RateC_is_lung(self, y, w, c, t):
		return (1 / c[162]) * (((1 - c[79]) * c[35] * (y[3]/1000.0) - (1 - c[51]) * c[35] * (y[7]/45.0)) + c[77] * (1 - c[78]) * (y[6]/5.0) - c[77] * (y[7]/45.0))

	def RateFcRn_free_lung(self, y, w, c, t):
		return (1 / c[163]) * ((c[75] * (y[6]/5.0) * c[163] - c[74] * (y[5]/5.0) * (y[8]/5.0) * c[163]) + c[77] * (y[6]/5.0))

	def RateC_p_brain(self, y, w, c, t):
		return (1 / c[86]) * ((c[17] * (y[3]/1000.0) - (c[17] - c[38]) * (y[9]/31.9) - (1 - c[95]) * c[91] * (y[9]/31.9) - (1 - c[96]) * c[18] * (y[9]/31.9) - c[92] * c[90] * (y[9]/31.9)) + c[92] * c[93] * c[90] * c[78] * (y[11]/0.1) + c[92] * (1 - c[93]) * c[90] * c[78] * (y[15]/0.1))

	def RateC_BBB_unbound_brain(self, y, w, c, t):
		return (1 / c[87]) * ((c[92] * c[93] * c[90] * ((y[9]/31.9) + (y[12]/261.0)) - c[87] * c[74] * (y[10]/0.1) * c[94]) + c[87] * c[75] * (y[11]/0.1) - c[87] * c[76] * (y[10]/0.1))

	def RateC_BBB_bound_brain(self, y, w, c, t):
		return (1 / c[87]) * (-c[92] * c[93] * c[90] * (y[11]/0.1) + c[87] * c[74] * (y[10]/0.1) * c[94] - c[87] * c[75] * (y[11]/0.1))

	def RateC_is_brain(self, y, w, c, t):
		return (1 / c[88]) * (((1 - c[95]) * c[91] * (y[9]/31.9) - (1 - c[54]) * c[91] * (y[12]/261.0) - c[91] * (y[12]/261.0)) + c[91] * (y[19]/90.0) + c[92] * c[93] * c[90] * (1 - c[78]) * (y[11]/0.1) - c[92] * c[93] * c[90] * (y[12]/261.0))

	def RateC_bc_brain(self, y, w, c, t):
		return (1 / c[89]) * (c[34] * (y[4]/55.0) - c[34] * (y[13]/26.1))

	def RateC_BCSFB_unbound_brain(self, y, w, c, t):
		return (1 / c[97]) * ((c[92] * c[102] * c[90] * (y[9]/31.9) + c[104] * c[92] * (1 - c[93]) * c[90] * (y[16]/22.5) + (1 - c[104]) * c[92] * (1 - c[93]) * c[90] * (y[17]/22.5) - c[97] * c[74] * (y[14]/0.1) * c[103]) + c[97] * c[75] * (y[15]/0.1) - c[97] * c[76] * (y[14]/0.1))

	def RateC_BCSFB_bound_brain(self, y, w, c, t):
		return (1 / c[97]) * (-c[92] * (1 - c[93]) * c[90] * (y[15]/0.1) + c[97] * c[74] * (y[14]/0.1) * c[103] - c[97] * c[75] * (y[15]/0.1))

	def RateC_LV_brain(self, y, w, c, t):
		return (1 / c[98]) * (((1 - c[96]) * c[104] * c[18] * (y[9]/31.9) + c[104] * c[91] * (y[12]/261.0) - (c[104] * c[18] + c[104] * c[91]) * (y[16]/22.5) - c[104] * c[92] * (1 - c[93]) * c[90] * (y[16]/22.5)) + c[104] * c[92] * (1 - c[93]) * c[90] * (1 - c[78]) * (y[15]/0.1))

	def RateC_TFV_brain(self, y, w, c, t):
		return (1 / c[99]) * (((1 - c[96]) * (1 - c[104]) * c[18] * (y[9]/31.9) + (1 - c[104]) * c[91] * (y[12]/261.0) - (c[18] + c[91]) * (y[17]/22.5) - (1 - c[104]) * c[92] * (1 - c[93]) * c[90] * (y[17]/22.5)) + (1 - c[104]) * c[92] * (1 - c[93]) * c[90] * (1 - c[78]) * (y[15]/0.1) + (c[104] * c[18] + c[104] * c[91]) * (y[16]/22.5))

	def RateC_CM_brain(self, y, w, c, t):
		return (1 / c[100]) * (c[18] + c[91]) * ((y[17]/22.5) - (y[18]/7.5))

	def RateC_SAS_brain(self, y, w, c, t):
		return (1 / c[101]) * ((c[18] + c[91]) * (y[18]/7.5) - (1 - c[66]) * c[18] * (y[19]/90.0) - c[91] * (y[19]/90.0))

	def RateC_p_liver(self, y, w, c, t):
		return (1 / c[175]) * ((c[10] * (y[3]/1000.0) + (c[15] - c[46]) * (y[80]/221.0) + (c[13] - c[47]) * (y[86]/104.0) + (c[11] - c[44]) * (y[68]/385.0) + (c[12] - c[45]) * (y[74]/548.0) - ((c[10] - c[48]) + (c[15] - c[46]) + (c[13] - c[47]) + (c[11] - c[44]) + (c[12] - c[45])) * (y[20]/2143.0) - (1 - c[118]) * c[48] * (y[20]/2143.0) - c[117] * (y[20]/2143.0)) + c[117] * c[78] * (y[24]/10.7))

	def RateC_bc_liver(self, y, w, c, t):
		return (1 / c[176]) * (c[27] * (y[4]/55.0) + c[32] * (y[81]/26.8) + c[30] * (y[87]/5.7) + c[28] * (y[69]/6.15) + c[29] * (y[75]/8.74) - (c[27] + c[32] + c[30] + c[28] + c[29]) * (y[21]/183.0))

	def RateC_is_liver(self, y, w, c, t):
		return (1 / c[177]) * (((1 - c[118]) * c[48] * (y[20]/2143.0) - (1 - c[64]) * c[48] * (y[22]/149.0)) + c[117] * (1 - c[78]) * (y[24]/10.7) - c[117] * (y[22]/149.0))

	def RateC_e_unbound_liver(self, y, w, c, t):
		return (1 / c[178]) * ((c[117] * ((y[20]/2143.0) + (y[22]/149.0)) - c[178] * c[74] * (y[23]/10.7) * (y[25]/10.7)) + c[178] * c[75] * (y[24]/10.7) - c[76] * (y[23]/10.7) * c[178])

	def RateC_e_bound_liver(self, y, w, c, t):
		return (1 / c[178]) * (c[178] * c[74] * (y[23]/10.7) * (y[25]/10.7) - c[178] * c[75] * (y[24]/10.7) - c[117] * (y[24]/10.7))

	def RateFcRn_free_liver(self, y, w, c, t):
		return (1 / c[178]) * ((c[75] * (y[24]/10.7) * c[178] - c[74] * (y[23]/10.7) * (y[25]/10.7) * c[178]) + c[117] * (y[24]/10.7))

	def RateC_p_heart(self, y, w, c, t):
		return (1 / c[179]) * ((c[3] * (y[3]/1000.0) - c[3] * (y[26]/341.0) - (1 - c[134]) * c[36] * (y[26]/341.0) - c[133] * (y[26]/341.0)) + c[133] * c[78] * (y[30]/1.71))

	def RateC_bc_heart(self, y, w, c, t):
		return (1 / c[180]) * (c[20] * (y[4]/55.0) - c[20] * (y[27]/13.1))

	def RateC_is_heart(self, y, w, c, t):
		return (1 / c[181]) * (((1 - c[134]) * c[36] * (y[26]/341.0) - (1 - c[52]) * c[36] * (y[28]/10.8)) + c[133] * (1 - c[78]) * (y[30]/1.71) - c[133] * (y[28]/10.8))

	def RateC_e_unbound_heart(self, y, w, c, t):
		return (1 / c[182]) * ((c[133] * ((y[26]/341.0) + (y[28]/10.8)) - c[182] * c[74] * (y[29]/1.71) * (y[31]/1.71)) + c[182] * c[75] * (y[30]/1.71) - c[76] * (y[29]/1.71) * c[182])

	def RateC_e_bound_heart(self, y, w, c, t):
		return (1 / c[182]) * (c[182] * c[74] * (y[29]/1.71) * (y[31]/1.71) - c[182] * c[75] * (y[30]/1.71) - c[133] * (y[30]/1.71))

	def RateFcRn_free_heart(self, y, w, c, t):
		return (1 / c[182]) * ((c[75] * (y[30]/1.71) * c[182] - c[74] * (y[29]/1.71) * (y[31]/1.71) * c[182]) + c[133] * (y[30]/1.71))

	def RateC_p_muscle(self, y, w, c, t):
		return (1 / c[183]) * ((c[5] * (y[3]/1000.0) - c[5] * (y[32]/30078.0) - (1 - c[136]) * c[39] * (y[32]/30078.0) - c[135] * (y[32]/30078.0)) + c[135] * c[78] * (y[36]/150.0))

	def RateC_bc_muscle(self, y, w, c, t):
		return (1 / c[184]) * (c[22] * (y[4]/55.0) - c[22] * (y[33]/662.0))

	def RateC_is_muscle(self, y, w, c, t):
		return (1 / c[185]) * (((1 - c[136]) * c[39] * (y[32]/30078.0) - (1 - c[55]) * c[39] * (y[34]/541.0)) + c[135] * (1 - c[78]) * (y[36]/150.0) - c[135] * (y[34]/541.0))

	def RateC_e_unbound_muscle(self, y, w, c, t):
		return (1 / c[186]) * ((c[135] * ((y[32]/30078.0) + (y[34]/541.0)) - c[186] * c[74] * (y[35]/150.0) * (y[37]/150.0)) + c[186] * c[75] * (y[36]/150.0) - c[76] * (y[35]/150.0) * c[186])

	def RateC_e_bound_muscle(self, y, w, c, t):
		return (1 / c[186]) * (c[186] * c[74] * (y[35]/150.0) * (y[37]/150.0) - c[186] * c[75] * (y[36]/150.0) - c[135] * (y[36]/150.0))

	def RateFcRn_free_muscle(self, y, w, c, t):
		return (1 / c[186]) * ((c[75] * (y[36]/150.0) * c[186] - c[74] * (y[35]/150.0) * (y[37]/150.0) * c[186]) + c[135] * (y[36]/150.0))

	def RateC_p_kidney(self, y, w, c, t):
		return (1 / c[187]) * ((c[9] * (y[3]/1000.0) - c[9] * (y[38]/332.0) - (1 - c[138]) * c[37] * (y[38]/332.0) - c[137] * (y[38]/332.0)) + c[137] * c[78] * (y[42]/1.66))

	def RateC_bc_kidney(self, y, w, c, t):
		return (1 / c[188]) * (c[26] * (y[4]/55.0) - c[26] * (y[39]/18.2))

	def RateC_is_kidney(self, y, w, c, t):
		return (1 / c[189]) * (((1 - c[138]) * c[37] * (y[38]/332.0) - (1 - c[53]) * c[37] * (y[40]/14.9)) + c[137] * (1 - c[78]) * (y[42]/1.66) - c[137] * (y[40]/14.9))

	def RateC_e_unbound_kidney(self, y, w, c, t):
		return (1 / c[190]) * ((c[137] * ((y[38]/332.0) + (y[40]/14.9)) - c[190] * c[74] * (y[41]/1.66) * (y[43]/1.66)) + c[190] * c[75] * (y[42]/1.66) - c[76] * (y[41]/1.66) * c[190])

	def RateC_e_bound_kidney(self, y, w, c, t):
		return (1 / c[190]) * (c[190] * c[74] * (y[41]/1.66) * (y[43]/1.66) - c[190] * c[75] * (y[42]/1.66) - c[137] * (y[42]/1.66))

	def RateFcRn_free_kidney(self, y, w, c, t):
		return (1 / c[190]) * ((c[75] * (y[42]/1.66) * c[190] - c[74] * (y[41]/1.66) * (y[43]/1.66) * c[190]) + c[137] * (y[42]/1.66))

	def RateC_p_skin(self, y, w, c, t):
		return (1 / c[191]) * ((c[6] * (y[3]/1000.0) - c[6] * (y[44]/3408.0) - (1 - c[140]) * c[42] * (y[44]/3408.0) - c[139] * (y[44]/3408.0)) + c[139] * c[78] * (y[48]/17.0))

	def RateC_bc_skin(self, y, w, c, t):
		return (1 / c[192]) * (c[23] * (y[4]/55.0) - c[23] * (y[45]/127.0))

	def RateC_is_skin(self, y, w, c, t):
		return (1 / c[193]) * (((1 - c[140]) * c[42] * (y[44]/3408.0) - (1 - c[58]) * c[42] * (y[46]/104.0)) + c[139] * (1 - c[78]) * (y[48]/17.0) - c[139] * (y[46]/104.0))

	def RateC_e_unbound_skin(self, y, w, c, t):
		return (1 / c[194]) * ((c[139] * ((y[44]/3408.0) + (y[46]/104.0)) - c[194] * c[74] * (y[47]/17.0) * (y[49]/17.0)) + c[194] * c[75] * (y[48]/17.0) - c[76] * (y[47]/17.0) * c[194])

	def RateC_e_bound_skin(self, y, w, c, t):
		return (1 / c[194]) * (c[194] * c[74] * (y[47]/17.0) * (y[49]/17.0) - c[194] * c[75] * (y[48]/17.0) - c[139] * (y[48]/17.0))

	def RateFcRn_free_skin(self, y, w, c, t):
		return (1 / c[194]) * ((c[75] * (y[48]/17.0) * c[194] - c[74] * (y[47]/17.0) * (y[49]/17.0) * c[194]) + c[139] * (y[48]/17.0))

	def RateC_p_fat(self, y, w, c, t):
		return (1 / c[195]) * ((c[7] * (y[3]/1000.0) - c[7] * (y[50]/13465.0) - (1 - c[142]) * c[43] * (y[50]/13465.0) - c[141] * (y[50]/13465.0)) + c[141] * c[78] * (y[54]/67.3))

	def RateC_bc_fat(self, y, w, c, t):
		return (1 / c[196]) * (c[24] * (y[4]/55.0) - c[24] * (y[51]/148.0))

	def RateC_is_fat(self, y, w, c, t):
		return (1 / c[197]) * (((1 - c[142]) * c[43] * (y[50]/13465.0) - (1 - c[59]) * c[43] * (y[52]/121.0)) + c[141] * (1 - c[78]) * (y[54]/67.3) - c[141] * (y[52]/121.0))

	def RateC_e_unbound_fat(self, y, w, c, t):
		return (1 / c[198]) * ((c[141] * ((y[50]/13465.0) + (y[52]/121.0)) - c[198] * c[74] * (y[53]/67.3) * (y[55]/67.3)) + c[198] * c[75] * (y[54]/67.3) - c[76] * (y[53]/67.3) * c[198])

	def RateC_e_bound_fat(self, y, w, c, t):
		return (1 / c[198]) * (c[198] * c[74] * (y[53]/67.3) * (y[55]/67.3) - c[198] * c[75] * (y[54]/67.3) - c[141] * (y[54]/67.3))

	def RateFcRn_free_fat(self, y, w, c, t):
		return (1 / c[198]) * ((c[75] * (y[54]/67.3) * c[198] - c[74] * (y[53]/67.3) * (y[55]/67.3) * c[198]) + c[141] * (y[54]/67.3))

	def RateC_p_marrow(self, y, w, c, t):
		return (1 / c[199]) * ((c[8] * (y[3]/1000.0) - c[8] * (y[56]/10165.0) - (1 - c[144]) * c[40] * (y[56]/10165.0) - c[143] * (y[56]/10165.0)) + c[143] * c[78] * (y[60]/50.8))

	def RateC_bc_marrow(self, y, w, c, t):
		return (1 / c[200]) * (c[25] * (y[4]/55.0) - c[25] * (y[57]/224.0))

	def RateC_is_marrow(self, y, w, c, t):
		return (1 / c[201]) * (((1 - c[144]) * c[40] * (y[56]/10165.0) - (1 - c[56]) * c[40] * (y[58]/183.0)) + c[143] * (1 - c[78]) * (y[60]/50.8) - c[143] * (y[58]/183.0))

	def RateC_e_unbound_marrow(self, y, w, c, t):
		return (1 / c[202]) * ((c[143] * ((y[56]/10165.0) + (y[58]/183.0)) - c[202] * c[74] * (y[59]/50.8) * (y[61]/50.8)) + c[202] * c[75] * (y[60]/50.8) - c[76] * (y[59]/50.8) * c[202])

	def RateC_e_bound_marrow(self, y, w, c, t):
		return (1 / c[202]) * (c[202] * c[74] * (y[59]/50.8) * (y[61]/50.8) - c[202] * c[75] * (y[60]/50.8) - c[143] * (y[60]/50.8))

	def RateFcRn_free_marrow(self, y, w, c, t):
		return (1 / c[202]) * ((c[75] * (y[60]/50.8) * c[202] - c[74] * (y[59]/50.8) * (y[61]/50.8) * c[202]) + c[143] * (y[60]/50.8))

	def RateC_p_thymus(self, y, w, c, t):
		return (1 / c[203]) * ((c[14] * (y[3]/1000.0) - c[14] * (y[62]/6.41) - (1 - c[146]) * c[41] * (y[62]/6.41) - c[145] * (y[62]/6.41)) + c[145] * c[78] * (y[66]/0.0321))

	def RateC_bc_thymus(self, y, w, c, t):
		return (1 / c[204]) * (c[31] * (y[4]/55.0) - c[31] * (y[63]/0.353))

	def RateC_is_thymus(self, y, w, c, t):
		return (1 / c[205]) * (((1 - c[146]) * c[41] * (y[62]/6.41) - (1 - c[57]) * c[41] * (y[64]/0.288)) + c[145] * (1 - c[78]) * (y[66]/0.0321) - c[145] * (y[64]/0.288))

	def RateC_e_unbound_thymus(self, y, w, c, t):
		return (1 / c[206]) * ((c[145] * ((y[62]/6.41) + (y[64]/0.288)) - c[206] * c[74] * (y[65]/0.0321) * (y[67]/0.0321)) + c[206] * c[75] * (y[66]/0.0321) - c[76] * (y[65]/0.0321) * c[206])

	def RateC_e_bound_thymus(self, y, w, c, t):
		return (1 / c[206]) * (c[206] * c[74] * (y[65]/0.0321) * (y[67]/0.0321) - c[206] * c[75] * (y[66]/0.0321) - c[145] * (y[66]/0.0321))

	def RateFcRn_free_thymus(self, y, w, c, t):
		return (1 / c[206]) * ((c[75] * (y[66]/0.0321) * c[206] - c[74] * (y[65]/0.0321) * (y[67]/0.0321) * c[206]) + c[145] * (y[66]/0.0321))

	def RateC_p_SI(self, y, w, c, t):
		return (1 / c[207]) * ((c[11] * (y[3]/1000.0) - c[11] * (y[68]/385.0) - (1 - c[148]) * c[44] * (y[68]/385.0) - c[147] * (y[68]/385.0)) + c[147] * c[78] * (y[72]/1.93))

	def RateC_bc_SI(self, y, w, c, t):
		return (1 / c[208]) * (c[28] * (y[4]/55.0) - c[28] * (y[69]/6.15))

	def RateC_is_SI(self, y, w, c, t):
		return (1 / c[209]) * (((1 - c[148]) * c[44] * (y[68]/385.0) - (1 - c[60]) * c[44] * (y[70]/5.03)) + c[147] * (1 - c[78]) * (y[72]/1.93) - c[147] * (y[70]/5.03))

	def RateC_e_unbound_SI(self, y, w, c, t):
		return (1 / c[210]) * ((c[147] * ((y[68]/385.0) + (y[70]/5.03)) - c[210] * c[74] * (y[71]/1.93) * (y[73]/1.93)) + c[210] * c[75] * (y[72]/1.93) - c[76] * (y[71]/1.93) * c[210])

	def RateC_e_bound_SI(self, y, w, c, t):
		return (1 / c[210]) * (c[210] * c[74] * (y[71]/1.93) * (y[73]/1.93) - c[210] * c[75] * (y[72]/1.93) - c[147] * (y[72]/1.93))

	def RateFcRn_free_SI(self, y, w, c, t):
		return (1 / c[210]) * ((c[75] * (y[72]/1.93) * c[210] - c[74] * (y[71]/1.93) * (y[73]/1.93) * c[210]) + c[147] * (y[72]/1.93))

	def RateC_p_LI(self, y, w, c, t):
		return (1 / c[211]) * ((c[12] * (y[3]/1000.0) - c[12] * (y[74]/548.0) - (1 - c[150]) * c[45] * (y[74]/548.0) - c[149] * (y[74]/548.0)) + c[149] * c[78] * (y[78]/2.74))

	def RateC_bc_LI(self, y, w, c, t):
		return (1 / c[212]) * (c[29] * (y[4]/55.0) - c[29] * (y[75]/8.74))

	def RateC_is_LI(self, y, w, c, t):
		return (1 / c[213]) * (((1 - c[150]) * c[45] * (y[74]/548.0) - (1 - c[61]) * c[45] * (y[76]/7.15)) + c[149] * (1 - c[78]) * (y[78]/2.74) - c[149] * (y[76]/7.15))

	def RateC_e_unbound_LI(self, y, w, c, t):
		return (1 / c[214]) * ((c[149] * ((y[74]/548.0) + (y[76]/7.15)) - c[214] * c[74] * (y[77]/2.74) * (y[79]/2.74)) + c[214] * c[75] * (y[78]/2.74) - c[76] * (y[77]/2.74) * c[214])

	def RateC_e_bound_LI(self, y, w, c, t):
		return (1 / c[214]) * (c[214] * c[74] * (y[77]/2.74) * (y[79]/2.74) - c[214] * c[75] * (y[78]/2.74) - c[149] * (y[78]/2.74))

	def RateFcRn_free_LI(self, y, w, c, t):
		return (1 / c[214]) * ((c[75] * (y[78]/2.74) * c[214] - c[74] * (y[77]/2.74) * (y[79]/2.74) * c[214]) + c[149] * (y[78]/2.74))

	def RateC_p_spleen(self, y, w, c, t):
		return (1 / c[215]) * ((c[15] * (y[3]/1000.0) - c[15] * (y[80]/221.0) - (1 - c[152]) * c[46] * (y[80]/221.0) - c[151] * (y[80]/221.0)) + c[151] * c[78] * (y[84]/1.11))

	def RateC_bc_spleen(self, y, w, c, t):
		return (1 / c[216]) * (c[32] * (y[4]/55.0) - c[32] * (y[81]/26.8))

	def RateC_is_spleen(self, y, w, c, t):
		return (1 / c[217]) * (((1 - c[152]) * c[46] * (y[80]/221.0) - (1 - c[62]) * c[46] * (y[82]/21.9)) + c[151] * (1 - c[78]) * (y[84]/1.11) - c[151] * (y[82]/21.9))

	def RateC_e_unbound_spleen(self, y, w, c, t):
		return (1 / c[218]) * ((c[151] * ((y[80]/221.0) + (y[82]/21.9)) - c[218] * c[74] * (y[83]/1.11) * (y[85]/1.11)) + c[218] * c[75] * (y[84]/1.11) - c[76] * (y[83]/1.11) * c[218])

	def RateC_e_bound_spleen(self, y, w, c, t):
		return (1 / c[218]) * (c[218] * c[74] * (y[83]/1.11) * (y[85]/1.11) - c[218] * c[75] * (y[84]/1.11) - c[151] * (y[84]/1.11))

	def RateFcRn_free_spleen(self, y, w, c, t):
		return (1 / c[218]) * ((c[75] * (y[84]/1.11) * c[218] - c[74] * (y[83]/1.11) * (y[85]/1.11) * c[218]) + c[151] * (y[84]/1.11))

	def RateC_p_pancreas(self, y, w, c, t):
		return (1 / c[219]) * ((c[13] * (y[3]/1000.0) - c[13] * (y[86]/104.0) - (1 - c[154]) * c[47] * (y[86]/104.0) - c[153] * (y[86]/104.0)) + c[153] * c[78] * (y[90]/0.518))

	def RateC_bc_pancreas(self, y, w, c, t):
		return (1 / c[220]) * (c[30] * (y[4]/55.0) - c[30] * (y[87]/5.7))

	def RateC_is_pancreas(self, y, w, c, t):
		return (1 / c[221]) * (((1 - c[154]) * c[47] * (y[86]/104.0) - (1 - c[63]) * c[47] * (y[88]/4.66)) + c[153] * (1 - c[78]) * (y[90]/0.518) - c[153] * (y[88]/4.66))

	def RateC_e_unbound_pancreas(self, y, w, c, t):
		return (1 / c[222]) * ((c[153] * ((y[86]/104.0) + (y[88]/4.66)) - c[222] * c[74] * (y[89]/0.518) * (y[91]/0.518)) + c[222] * c[75] * (y[90]/0.518) - c[76] * (y[89]/0.518) * c[222])

	def RateC_e_bound_pancreas(self, y, w, c, t):
		return (1 / c[222]) * (c[222] * c[74] * (y[89]/0.518) * (y[91]/0.518) - c[222] * c[75] * (y[90]/0.518) - c[153] * (y[90]/0.518))

	def RateFcRn_free_pancreas(self, y, w, c, t):
		return (1 / c[222]) * ((c[75] * (y[90]/0.518) * c[222] - c[74] * (y[89]/0.518) * (y[91]/0.518) * c[222]) + c[153] * (y[90]/0.518))

	def RateC_p_other(self, y, w, c, t):
		return (1 / c[223]) * ((c[16] * (y[3]/1000.0) - c[16] * (y[92]/4852.0) - (1 - c[156]) * c[49] * (y[92]/4852.0) - c[155] * (y[92]/4852.0)) + c[155] * c[78] * (y[96]/24.3))

	def RateC_bc_other(self, y, w, c, t):
		return (1 / c[224]) * (c[33] * (y[4]/55.0) - c[33] * (y[93]/204.0))

	def RateC_is_other(self, y, w, c, t):
		return (1 / c[225]) * (((1 - c[156]) * c[49] * (y[92]/4852.0) - (1 - c[65]) * c[49] * (y[94]/167.0)) + c[155] * (1 - c[78]) * (y[96]/24.3) - c[155] * (y[94]/167.0))

	def RateC_e_unbound_other(self, y, w, c, t):
		return (1 / c[226]) * ((c[155] * ((y[92]/4852.0) + (y[94]/167.0)) - c[226] * c[74] * (y[95]/24.3) * (y[97]/24.3)) + c[226] * c[75] * (y[96]/24.3) - c[76] * (y[95]/24.3) * c[226])

	def RateC_e_bound_other(self, y, w, c, t):
		return (1 / c[226]) * (c[226] * c[74] * (y[95]/24.3) * (y[97]/24.3) - c[226] * c[75] * (y[96]/24.3) - c[155] * (y[96]/24.3))

	def RateFcRn_free_other(self, y, w, c, t):
		return (1 / c[226]) * ((c[75] * (y[96]/24.3) * c[226] - c[74] * (y[95]/24.3) * (y[97]/24.3) * c[226]) + c[155] * (y[96]/24.3))

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		return w

class ModelStep(eqx.Module):
	y_indexes: dict = eqx.static_field()
	w_indexes: dict = eqx.static_field()
	c_indexes: dict = eqx.static_field()
	ratefunc: RateofSpeciesChange
	atol: float = eqx.static_field()
	rtol: float = eqx.static_field()
	mxstep: int = eqx.static_field()
	assignmentfunc: AssignmentRule

	def __init__(self, y_indexes={'C_p': 0, 'C_bc': 1, 'C_ln': 2, 'C_p_lung': 3, 'C_bc_lung': 4, 'C_e_unbound_lung': 5, 'C_e_bound_lung': 6, 'C_is_lung': 7, 'FcRn_free_lung': 8, 'C_p_brain': 9, 'C_BBB_unbound_brain': 10, 'C_BBB_bound_brain': 11, 'C_is_brain': 12, 'C_bc_brain': 13, 'C_BCSFB_unbound_brain': 14, 'C_BCSFB_bound_brain': 15, 'C_LV_brain': 16, 'C_TFV_brain': 17, 'C_CM_brain': 18, 'C_SAS_brain': 19, 'C_p_liver': 20, 'C_bc_liver': 21, 'C_is_liver': 22, 'C_e_unbound_liver': 23, 'C_e_bound_liver': 24, 'FcRn_free_liver': 25, 'C_p_heart': 26, 'C_bc_heart': 27, 'C_is_heart': 28, 'C_e_unbound_heart': 29, 'C_e_bound_heart': 30, 'FcRn_free_heart': 31, 'C_p_muscle': 32, 'C_bc_muscle': 33, 'C_is_muscle': 34, 'C_e_unbound_muscle': 35, 'C_e_bound_muscle': 36, 'FcRn_free_muscle': 37, 'C_p_kidney': 38, 'C_bc_kidney': 39, 'C_is_kidney': 40, 'C_e_unbound_kidney': 41, 'C_e_bound_kidney': 42, 'FcRn_free_kidney': 43, 'C_p_skin': 44, 'C_bc_skin': 45, 'C_is_skin': 46, 'C_e_unbound_skin': 47, 'C_e_bound_skin': 48, 'FcRn_free_skin': 49, 'C_p_fat': 50, 'C_bc_fat': 51, 'C_is_fat': 52, 'C_e_unbound_fat': 53, 'C_e_bound_fat': 54, 'FcRn_free_fat': 55, 'C_p_marrow': 56, 'C_bc_marrow': 57, 'C_is_marrow': 58, 'C_e_unbound_marrow': 59, 'C_e_bound_marrow': 60, 'FcRn_free_marrow': 61, 'C_p_thymus': 62, 'C_bc_thymus': 63, 'C_is_thymus': 64, 'C_e_unbound_thymus': 65, 'C_e_bound_thymus': 66, 'FcRn_free_thymus': 67, 'C_p_SI': 68, 'C_bc_SI': 69, 'C_is_SI': 70, 'C_e_unbound_SI': 71, 'C_e_bound_SI': 72, 'FcRn_free_SI': 73, 'C_p_LI': 74, 'C_bc_LI': 75, 'C_is_LI': 76, 'C_e_unbound_LI': 77, 'C_e_bound_LI': 78, 'FcRn_free_LI': 79, 'C_p_spleen': 80, 'C_bc_spleen': 81, 'C_is_spleen': 82, 'C_e_unbound_spleen': 83, 'C_e_bound_spleen': 84, 'FcRn_free_spleen': 85, 'C_p_pancreas': 86, 'C_bc_pancreas': 87, 'C_is_pancreas': 88, 'C_e_unbound_pancreas': 89, 'C_e_bound_pancreas': 90, 'FcRn_free_pancreas': 91, 'C_p_other': 92, 'C_bc_other': 93, 'C_is_other': 94, 'C_e_unbound_other': 95, 'C_e_bound_other': 96, 'FcRn_free_other': 97}, w_indexes={}, c_indexes={'Vp': 0, 'Vbc': 1, 'Vlymphnode': 2, 'Q_p_heart': 3, 'Q_p_lung': 4, 'Q_p_muscle': 5, 'Q_p_skin': 6, 'Q_p_fat': 7, 'Q_p_marrow': 8, 'Q_p_kidney': 9, 'Q_p_liver': 10, 'Q_p_SI': 11, 'Q_p_LI': 12, 'Q_p_pancreas': 13, 'Q_p_thymus': 14, 'Q_p_spleen': 15, 'Q_p_other': 16, 'Q_p_brain': 17, 'Q_CSF_brain': 18, 'Q_ECF_brain': 19, 'Q_bc_heart': 20, 'Q_bc_lung': 21, 'Q_bc_muscle': 22, 'Q_bc_skin': 23, 'Q_bc_fat': 24, 'Q_bc_marrow': 25, 'Q_bc_kidney': 26, 'Q_bc_liver': 27, 'Q_bc_SI': 28, 'Q_bc_LI': 29, 'Q_bc_pancreas': 30, 'Q_bc_thymus': 31, 'Q_bc_spleen': 32, 'Q_bc_other': 33, 'Q_bc_brain': 34, 'L_lung': 35, 'L_heart': 36, 'L_kidney': 37, 'L_brain': 38, 'L_muscle': 39, 'L_marrow': 40, 'L_thymus': 41, 'L_skin': 42, 'L_fat': 43, 'L_SI': 44, 'L_LI': 45, 'L_spleen': 46, 'L_pancreas': 47, 'L_liver': 48, 'L_other': 49, 'L_LN': 50, 'sigma_L_lung': 51, 'sigma_L_heart': 52, 'sigma_L_kidney': 53, 'sigma_L_brain_ISF': 54, 'sigma_L_muscle': 55, 'sigma_L_marrow': 56, 'sigma_L_thymus': 57, 'sigma_L_skin': 58, 'sigma_L_fat': 59, 'sigma_L_SI': 60, 'sigma_L_LI': 61, 'sigma_L_spleen': 62, 'sigma_L_pancreas': 63, 'sigma_L_liver': 64, 'sigma_L_other': 65, 'sigma_L_SAS': 66, 'C_p_0': 67, 'C_bc_0': 68, 'C_ln_0': 69, 'Vp_lung': 160, 'VBC_lung': 161, 'VIS_lung': 162, 'VES_lung': 163, 'kon_FcRn': 74, 'koff_FcRn': 75, 'kdeg': 76, 'CLup_lung': 77, 'FR': 78, 'sigma_V_lung': 79, 'C_p_lung_0': 80, 'C_bc_lung_0': 81, 'C_is_lung_0': 82, 'C_e_unbound_lung_0': 83, 'C_e_bound_lung_0': 84, 'FcRn_free_lung_0': 85, 'Vp_brain': 86, 'VBBB_brain': 87, 'VIS_brain': 88, 'VBC_brain': 89, 'V_ES_brain': 90, 'Q_ISF_brain': 91, 'CLup_brain': 92, 'f_BBB': 93, 'FcRn_free_BBB': 94, 'sigma_V_BBB': 95, 'sigma_V_BCSFB': 96, 'V_BCSFB_brain': 97, 'V_LV_brain': 98, 'V_TFV_brain': 99, 'V_CM_brain': 100, 'V_SAS_brain': 101, 'f_BCSFB': 102, 'FcRn_free_BCSFB': 103, 'f_LV': 104, 'C_BCSFB_unbound_brain_0': 105, 'C_BCSFB_bound_brain_0': 106, 'C_LV_brain_0': 107, 'C_TFV_brain_0': 108, 'C_CM_brain_0': 109, 'C_SAS_brain_0': 110, 'C_p_brain_0': 111, 'C_is_brain_0': 112, 'Vp_liver': 175, 'VBC_liver': 176, 'VIS_liver': 177, 'VES_liver': 178, 'CLup_liver': 117, 'sigma_V_liver': 118, 'C_p_liver_0': 119, 'C_bc_liver_0': 120, 'C_is_liver_0': 121, 'C_e_unbound_liver_0': 122, 'C_e_bound_liver_0': 123, 'FcRn_free_liver_0': 124, 'C_p_spleen_0': 125, 'C_bc_spleen_0': 126, 'C_p_pancreas_0': 127, 'C_bc_pancreas_0': 128, 'C_p_SI_0': 129, 'C_bc_SI_0': 130, 'C_p_LI_0': 131, 'C_bc_LI_0': 132, 'CLup_heart': 133, 'sigma_V_heart': 134, 'CLup_muscle': 135, 'sigma_V_muscle': 136, 'CLup_kidney': 137, 'sigma_V_kidney': 138, 'CLup_skin': 139, 'sigma_V_skin': 140, 'CLup_fat': 141, 'sigma_V_fat': 142, 'CLup_marrow': 143, 'sigma_V_marrow': 144, 'CLup_thymus': 145, 'sigma_V_thymus': 146, 'CLup_SI': 147, 'sigma_V_SI': 148, 'CLup_LI': 149, 'sigma_V_LI': 150, 'CLup_spleen': 151, 'sigma_V_spleen': 152, 'CLup_pancreas': 153, 'sigma_V_pancreas': 154, 'CLup_other': 155, 'sigma_V_other': 156, 'plasma': 157, 'blood_cells': 158, 'lymph_node': 159, 'brain_plasma': 164, 'BBB_unbound': 165, 'BBB_bound': 166, 'brain_ISF': 167, 'brain_blood_cells': 168, 'BCSFB_unbound': 169, 'BCSFB_bound': 170, 'LV': 171, 'TFV': 172, 'CM': 173, 'SAS': 174, 'Vp_heart': 179, 'VBC_heart': 180, 'VIS_heart': 181, 'VES_heart': 182, 'Vp_muscle': 183, 'VBC_muscle': 184, 'VIS_muscle': 185, 'VES_muscle': 186, 'Vp_kidney': 187, 'VBC_kidney': 188, 'VIS_kidney': 189, 'VES_kidney': 190, 'Vp_skin': 191, 'VBC_skin': 192, 'VIS_skin': 193, 'VES_skin': 194, 'Vp_fat': 195, 'VBC_fat': 196, 'VIS_fat': 197, 'VES_fat': 198, 'Vp_marrow': 199, 'VBC_marrow': 200, 'VIS_marrow': 201, 'VES_marrow': 202, 'Vp_thymus': 203, 'VBC_thymus': 204, 'VIS_thymus': 205, 'VES_thymus': 206, 'Vp_SI': 207, 'VBC_SI': 208, 'VIS_SI': 209, 'VES_SI': 210, 'Vp_LI': 211, 'VBC_LI': 212, 'VIS_LI': 213, 'VES_LI': 214, 'Vp_spleen': 215, 'VBC_spleen': 216, 'VIS_spleen': 217, 'VES_spleen': 218, 'Vp_pancreas': 219, 'VBC_pancreas': 220, 'VIS_pancreas': 221, 'VES_pancreas': 222, 'Vp_other': 223, 'VBC_other': 224, 'VIS_other': 225, 'VES_other': 226}, atol=1e-06, rtol=1e-12, mxstep=5000000):

		self.y_indexes = y_indexes
		self.w_indexes = w_indexes
		self.c_indexes = c_indexes

		self.ratefunc = RateofSpeciesChange()
		self.rtol = rtol
		self.atol = atol
		self.mxstep = mxstep
		self.assignmentfunc = AssignmentRule()

	@jit
	def __call__(self, y, w, c, t, deltaT):
		y_new = odeint(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]	
		t_new = t + deltaT	
		w_new = self.assignmentfunc(y_new, w, c, t_new)	
		return y_new, w_new, c, t_new	

class ModelRollout(eqx.Module):
	deltaT: float = eqx.static_field()
	modelstepfunc: ModelStep

	def __init__(self, deltaT=0.1, atol=1e-06, rtol=1e-12, mxstep=5000000):

		self.deltaT = deltaT
		self.modelstepfunc = ModelStep(atol=atol, rtol=rtol, mxstep=mxstep)

	@partial(jit, static_argnames=("n_steps",))
	def __call__(self, n_steps, y0=jnp.array([14117.999999999989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000533074, 0.0, 0.0, 0.0, 0.0, 0.0, 8.51922e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0074730000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 8.27012e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00084694, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003352886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0025308559999999997, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5992219999999998e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 9.61526e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00013650680000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 5.53002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5806760000000002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001210626]), w0=jnp.array([]), c=jnp.array([3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 8343.0, 2591.0, 32402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 36402.0, 21.0, 10.5, 6350.0, 148920.0, 27410.0, 9520.0, 6830.0, 2120.0, 26530.0, 10820.0, 10130.0, 10530.0, 2500.0, 289.0, 5190.0, 4520.0, 29810.0, 364.0, 16.0, 65.0, 73.0, 67.0, 5.0, 1.0, 23.0, 17.0, 25.0, 26.0, 13.0, 6.0, 26.0, 11.0, 500.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.51631477927063, 0.0, 0.0, 1000.0, 55.0, 45.0, 5.0, 559000000.0, 23.9, 26.6, 0.55, 0.715, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 31.9, 0.1, 261.0, 26.1, 7.25, 10.5, 0.3, 0.95, 1.0, 0.95, 0.9974, 0.1, 22.5, 22.5, 7.5, 90.0, 0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2143.0, 183.0, 149.0, 10.7, 0.55, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.95, 0.55, 0.95, 0.55, 0.9, 0.55, 0.95, 0.55, 0.95, 0.55, 0.95, 0.55, 0.9, 0.55, 0.9, 0.55, 0.95, 0.55, 0.85, 0.55, 0.9, 0.55, 0.95, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 31.9, 0.1, 0.1, 261.0, 26.1, 0.1, 0.1, 22.5, 22.5, 7.5, 90.0, 2143.0, 183.0, 149.0, 10.7, 341.0, 13.1, 10.8, 1.71, 30078.0, 662.0, 541.0, 150.0, 332.0, 18.2, 14.9, 1.66, 3408.0, 127.0, 104.0, 17.0, 13465.0, 148.0, 121.0, 67.3, 10165.0, 224.0, 183.0, 50.8, 6.41, 0.353, 0.288, 0.0321, 385.0, 6.15, 5.03, 1.93, 548.0, 8.74, 7.15, 2.74, 221.0, 26.8, 21.9, 1.11, 104.0, 5.7, 4.66, 0.518, 4852.0, 204.0, 167.0, 24.3]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

