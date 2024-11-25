import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([14117.999999999989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000533074, 0.0, 0.0, 0.0, 0.0, 0.0, 8.51922e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0074730000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 8.27012e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00084694, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003352886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0025308559999999997, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5992219999999998e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 9.61526e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00013650680000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 5.53002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5806760000000002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001210626])
y_indexes = {'C_p': 0, 'C_bc': 1, 'C_ln': 2, 'C_p_lung': 3, 'C_bc_lung': 4, 'C_e_unbound_lung': 5, 'C_e_bound_lung': 6, 'C_is_lung': 7, 'FcRn_free_lung': 8, 'C_p_brain': 9, 'C_BBB_unbound_brain': 10, 'C_BBB_bound_brain': 11, 'C_is_brain': 12, 'C_BCSFB_unbound_brain': 13, 'C_BCSFB_bound_brain': 14, 'C_LV_brain': 15, 'C_TFV_brain': 16, 'C_CM_brain': 17, 'C_SAS_brain': 18, 'C_p_liver': 19, 'C_bc_liver': 20, 'C_is_liver': 21, 'C_e_unbound_liver': 22, 'C_e_bound_liver': 23, 'FcRn_free_liver': 24, 'C_p_heart': 25, 'C_bc_heart': 26, 'C_e_unbound_heart': 27, 'C_e_bound_heart': 28, 'C_is_heart': 29, 'FcRn_free_heart': 30, 'C_p_muscle': 31, 'C_bc_muscle': 32, 'C_is_muscle': 33, 'C_e_unbound_muscle': 34, 'C_e_bound_muscle': 35, 'FcRn_free_muscle': 36, 'C_p_kidney': 37, 'C_bc_kidney': 38, 'C_is_kidney': 39, 'C_e_unbound_kidney': 40, 'C_e_bound_kidney': 41, 'FcRn_free_kidney': 42, 'C_p_skin': 43, 'C_bc_skin': 44, 'C_is_skin': 45, 'C_e_unbound_skin': 46, 'C_e_bound_skin': 47, 'FcRn_free_skin': 48, 'C_p_fat': 49, 'C_bc_fat': 50, 'C_is_fat': 51, 'C_e_unbound_fat': 52, 'C_e_bound_fat': 53, 'FcRn_free_fat': 54, 'C_p_marrow': 55, 'C_bc_marrow': 56, 'C_is_marrow': 57, 'C_e_unbound_marrow': 58, 'C_e_bound_marrow': 59, 'FcRn_free_marrow': 60, 'C_p_thymus': 61, 'C_bc_thymus': 62, 'C_is_thymus': 63, 'C_e_unbound_thymus': 64, 'C_e_bound_thymus': 65, 'FcRn_free_thymus': 66, 'C_p_SI': 67, 'C_bc_SI': 68, 'C_is_SI': 69, 'C_e_unbound_SI': 70, 'C_e_bound_SI': 71, 'FcRn_free_SI': 72, 'C_p_LI': 73, 'C_bc_LI': 74, 'C_is_LI': 75, 'C_e_unbound_LI': 76, 'C_e_bound_LI': 77, 'FcRn_free_LI': 78, 'C_p_spleen': 79, 'C_bc_spleen': 80, 'C_is_spleen': 81, 'C_e_unbound_spleen': 82, 'C_e_bound_spleen': 83, 'FcRn_free_spleen': 84, 'C_p_pancreas': 85, 'C_bc_pancreas': 86, 'C_is_pancreas': 87, 'C_e_unbound_pancreas': 88, 'C_e_bound_pancreas': 89, 'FcRn_free_pancreas': 90, 'C_p_other': 91, 'C_bc_other': 92, 'C_is_other': 93, 'C_e_unbound_other': 94, 'C_e_bound_other': 95, 'FcRn_free_other': 96}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 181913.0, 148920.0, 364.0, 0.95, 0.2, 0.55, 31.9, 261.0, 0.1, 26.1, 7.25, 0.1, 22.5, 22.5, 7.5, 90.0, 36402.0, 29810.0, 73.0, 0.95, 0.2, 0.2, 0.2, 0.3, 21.0, 10.5, 0.95, 0.9974, 2143.0, 183.0, 149.0, 10.7, 13210.0, 10820.0, 26.0, 0.95, 0.2, 0.55, 16.0, 65.0, 67.0, 23.0, 17.0, 5.0, 1.0, 25.0, 26.0, 13.0, 6.0, 11.0, 500.0, 559000000.0, 23.9, 26.6, 0.715, 0.95, 0.2, 1.0, 1.0, 341.0, 13.1, 10.8, 1.71, 7752.0, 6350.0, 0.95, 0.2, 0.55, 30078.0, 662.0, 541.0, 150.0, 33469.0, 27410.0, 0.95, 0.2, 0.55, 332.0, 18.2, 14.9, 1.66, 32402.0, 26530.0, 0.9, 0.2, 0.55, 3408.0, 127.0, 104.0, 17.0, 11626.0, 9520.0, 0.95, 0.2, 0.55, 13465.0, 148.0, 121.0, 67.3, 8343.0, 6830.0, 0.95, 0.2, 0.55, 10165.0, 224.0, 183.0, 50.8, 2591.0, 2120.0, 0.95, 0.2, 0.55, 6.41, 0.353, 0.288, 0.0321, 353.0, 289.0, 0.9, 0.2, 0.55, 385.0, 6.15, 5.03, 1.93, 12368.0, 10130.0, 0.9, 0.2, 0.55, 548.0, 8.74, 7.15, 2.74, 12867.0, 10530.0, 0.95, 0.2, 0.55, 221.0, 26.8, 21.9, 1.11, 6343.0, 5190.0, 0.85, 0.2, 0.55, 104.0, 5.7, 4.66, 0.518, 3056.0, 2500.0, 0.9, 0.2, 0.55, 4852.0, 204.0, 167.0, 24.3, 5521.0, 4520.0, 0.95, 0.2, 0.55, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 31.9, 261.0, 0.1, 26.1, 7.25, 0.1, 22.5, 22.5, 7.5, 90.0, 2143.0, 183.0, 149.0, 10.7, 341.0, 13.1, 10.8, 1.71, 30078.0, 662.0, 541.0, 150.0, 332.0, 18.2, 14.9, 1.66, 3408.0, 127.0, 104.0, 17.0, 13465.0, 148.0, 121.0, 67.3, 10165.0, 224.0, 183.0, 50.8, 6.41, 0.353, 0.288, 0.0321, 385.0, 6.15, 5.03, 1.93, 548.0, 8.74, 7.15, 2.74, 221.0, 26.8, 21.9, 1.11, 104.0, 5.7, 4.66, 0.518, 4852.0, 204.0, 167.0, 24.3]) 
c_indexes = {'C_bc_brain': 0, 'Vp': 1, 'Vbc': 2, 'Vlymphnode': 3, 'Vp_lung': 178, 'VBC_lung': 179, 'VIS_lung': 180, 'VES_lung': 181, 'Q_p_lung': 8, 'Q_bc_lung': 9, 'L_lung': 10, 'sigma_V_lung': 11, 'sigma_L_lung': 12, 'CLup_lung': 13, 'Vp_brain': 182, 'VIS_brain': 183, 'VBBB_brain': 184, 'VBC_brain': 185, 'V_ES_brain': 186, 'V_BCSFB_brain': 187, 'V_LV_brain': 188, 'V_TFV_brain': 189, 'V_CM_brain': 190, 'V_SAS_brain': 191, 'Q_p_brain': 24, 'Q_bc_brain': 25, 'L_brain': 26, 'sigma_V_brain': 27, 'sigma_L_brain': 28, 'sigma_L_SAS': 29, 'sigma_L_brain_ISF': 30, 'CLup_brain': 31, 'Q_CSF_brain': 32, 'Q_ISF_brain': 33, 'sigma_V_BBB': 34, 'sigma_V_BCSFB': 35, 'Vp_liver': 192, 'VBC_liver': 193, 'VIS_liver': 194, 'VES_liver': 195, 'Q_p_liver': 40, 'Q_bc_liver': 41, 'L_liver': 42, 'sigma_V_liver': 43, 'sigma_L_liver': 44, 'CLup_liver': 45, 'L_heart': 46, 'L_kidney': 47, 'L_muscle': 48, 'L_skin': 49, 'L_fat': 50, 'L_marrow': 51, 'L_thymus': 52, 'L_SI': 53, 'L_LI': 54, 'L_spleen': 55, 'L_pancreas': 56, 'L_other': 57, 'L_LN': 58, 'kon_FcRn': 59, 'koff_FcRn': 60, 'kdeg': 61, 'FR': 62, 'f_BBB': 63, 'f_LV': 64, 'FcRn_free_BBB': 65, 'FcRn_free_BCSFB': 66, 'Vp_heart': 196, 'VBC_heart': 197, 'VIS_heart': 198, 'VES_heart': 199, 'Q_p_heart': 71, 'Q_bc_heart': 72, 'sigma_V_heart': 73, 'sigma_L_heart': 74, 'CLup_heart': 75, 'Vp_muscle': 200, 'VBC_muscle': 201, 'VIS_muscle': 202, 'VES_muscle': 203, 'Q_p_muscle': 80, 'Q_bc_muscle': 81, 'sigma_V_muscle': 82, 'sigma_L_muscle': 83, 'CLup_muscle': 84, 'Vp_kidney': 204, 'VBC_kidney': 205, 'VIS_kidney': 206, 'VES_kidney': 207, 'Q_p_kidney': 89, 'Q_bc_kidney': 90, 'sigma_V_kidney': 91, 'sigma_L_kidney': 92, 'CLup_kidney': 93, 'Vp_skin': 208, 'VBC_skin': 209, 'VIS_skin': 210, 'VES_skin': 211, 'Q_p_skin': 98, 'Q_bc_skin': 99, 'sigma_V_skin': 100, 'sigma_L_skin': 101, 'CLup_skin': 102, 'Vp_fat': 212, 'VBC_fat': 213, 'VIS_fat': 214, 'VES_fat': 215, 'Q_p_fat': 107, 'Q_bc_fat': 108, 'sigma_V_fat': 109, 'sigma_L_fat': 110, 'CLup_fat': 111, 'Vp_marrow': 216, 'VBC_marrow': 217, 'VIS_marrow': 218, 'VES_marrow': 219, 'Q_p_marrow': 116, 'Q_bc_marrow': 117, 'sigma_V_marrow': 118, 'sigma_L_marrow': 119, 'CLup_marrow': 120, 'Vp_thymus': 220, 'VBC_thymus': 221, 'VIS_thymus': 222, 'VES_thymus': 223, 'Q_p_thymus': 125, 'Q_bc_thymus': 126, 'sigma_V_thymus': 127, 'sigma_L_thymus': 128, 'CLup_thymus': 129, 'Vp_SI': 224, 'VBC_SI': 225, 'VIS_SI': 226, 'VES_SI': 227, 'Q_p_SI': 134, 'Q_bc_SI': 135, 'sigma_V_SI': 136, 'sigma_L_SI': 137, 'CLup_SI': 138, 'Vp_LI': 228, 'VBC_LI': 229, 'VIS_LI': 230, 'VES_LI': 231, 'Q_p_LI': 143, 'Q_bc_LI': 144, 'sigma_V_LI': 145, 'sigma_L_LI': 146, 'CLup_LI': 147, 'Vp_spleen': 232, 'VBC_spleen': 233, 'VIS_spleen': 234, 'VES_spleen': 235, 'Q_p_spleen': 152, 'Q_bc_spleen': 153, 'sigma_V_spleen': 154, 'sigma_L_spleen': 155, 'CLup_spleen': 156, 'Vp_pancreas': 236, 'VBC_pancreas': 237, 'VIS_pancreas': 238, 'VES_pancreas': 239, 'Q_p_pancreas': 161, 'Q_bc_pancreas': 162, 'sigma_V_pancreas': 163, 'sigma_L_pancreas': 164, 'CLup_pancreas': 165, 'Vp_other': 240, 'VBC_other': 241, 'VIS_other': 242, 'VES_other': 243, 'Q_p_other': 170, 'Q_bc_other': 171, 'sigma_V_other': 172, 'sigma_L_other': 173, 'CLup_other': 174, 'plasma': 175, 'blood_cells': 176, 'lymph_node': 177}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p(y, w, c, t), self.RateC_bc(y, w, c, t), self.RateC_ln(y, w, c, t), self.RateC_p_lung(y, w, c, t), self.RateC_bc_lung(y, w, c, t), self.RateC_e_unbound_lung(y, w, c, t), self.RateC_e_bound_lung(y, w, c, t), self.RateC_is_lung(y, w, c, t), self.RateFcRn_free_lung(y, w, c, t), self.RateC_p_brain(y, w, c, t), self.RateC_BBB_unbound_brain(y, w, c, t), self.RateC_BBB_bound_brain(y, w, c, t), self.RateC_is_brain(y, w, c, t), self.RateC_BCSFB_unbound_brain(y, w, c, t), self.RateC_BCSFB_bound_brain(y, w, c, t), self.RateC_LV_brain(y, w, c, t), self.RateC_TFV_brain(y, w, c, t), self.RateC_CM_brain(y, w, c, t), self.RateC_SAS_brain(y, w, c, t), self.RateC_p_liver(y, w, c, t), self.RateC_bc_liver(y, w, c, t), self.RateC_is_liver(y, w, c, t), self.RateC_e_unbound_liver(y, w, c, t), self.RateC_e_bound_liver(y, w, c, t), self.RateFcRn_free_liver(y, w, c, t), self.RateC_p_heart(y, w, c, t), self.RateC_bc_heart(y, w, c, t), self.RateC_e_unbound_heart(y, w, c, t), self.RateC_e_bound_heart(y, w, c, t), self.RateC_is_heart(y, w, c, t), self.RateFcRn_free_heart(y, w, c, t), self.RateC_p_muscle(y, w, c, t), self.RateC_bc_muscle(y, w, c, t), self.RateC_is_muscle(y, w, c, t), self.RateC_e_unbound_muscle(y, w, c, t), self.RateC_e_bound_muscle(y, w, c, t), self.RateFcRn_free_muscle(y, w, c, t), self.RateC_p_kidney(y, w, c, t), self.RateC_bc_kidney(y, w, c, t), self.RateC_is_kidney(y, w, c, t), self.RateC_e_unbound_kidney(y, w, c, t), self.RateC_e_bound_kidney(y, w, c, t), self.RateFcRn_free_kidney(y, w, c, t), self.RateC_p_skin(y, w, c, t), self.RateC_bc_skin(y, w, c, t), self.RateC_is_skin(y, w, c, t), self.RateC_e_unbound_skin(y, w, c, t), self.RateC_e_bound_skin(y, w, c, t), self.RateFcRn_free_skin(y, w, c, t), self.RateC_p_fat(y, w, c, t), self.RateC_bc_fat(y, w, c, t), self.RateC_is_fat(y, w, c, t), self.RateC_e_unbound_fat(y, w, c, t), self.RateC_e_bound_fat(y, w, c, t), self.RateFcRn_free_fat(y, w, c, t), self.RateC_p_marrow(y, w, c, t), self.RateC_bc_marrow(y, w, c, t), self.RateC_is_marrow(y, w, c, t), self.RateC_e_unbound_marrow(y, w, c, t), self.RateC_e_bound_marrow(y, w, c, t), self.RateFcRn_free_marrow(y, w, c, t), self.RateC_p_thymus(y, w, c, t), self.RateC_bc_thymus(y, w, c, t), self.RateC_is_thymus(y, w, c, t), self.RateC_e_unbound_thymus(y, w, c, t), self.RateC_e_bound_thymus(y, w, c, t), self.RateFcRn_free_thymus(y, w, c, t), self.RateC_p_SI(y, w, c, t), self.RateC_bc_SI(y, w, c, t), self.RateC_is_SI(y, w, c, t), self.RateC_e_unbound_SI(y, w, c, t), self.RateC_e_bound_SI(y, w, c, t), self.RateFcRn_free_SI(y, w, c, t), self.RateC_p_LI(y, w, c, t), self.RateC_bc_LI(y, w, c, t), self.RateC_is_LI(y, w, c, t), self.RateC_e_unbound_LI(y, w, c, t), self.RateC_e_bound_LI(y, w, c, t), self.RateFcRn_free_LI(y, w, c, t), self.RateC_p_spleen(y, w, c, t), self.RateC_bc_spleen(y, w, c, t), self.RateC_is_spleen(y, w, c, t), self.RateC_e_unbound_spleen(y, w, c, t), self.RateC_e_bound_spleen(y, w, c, t), self.RateFcRn_free_spleen(y, w, c, t), self.RateC_p_pancreas(y, w, c, t), self.RateC_bc_pancreas(y, w, c, t), self.RateC_is_pancreas(y, w, c, t), self.RateC_e_unbound_pancreas(y, w, c, t), self.RateC_e_bound_pancreas(y, w, c, t), self.RateFcRn_free_pancreas(y, w, c, t), self.RateC_p_other(y, w, c, t), self.RateC_bc_other(y, w, c, t), self.RateC_is_other(y, w, c, t), self.RateC_e_unbound_other(y, w, c, t), self.RateC_e_bound_other(y, w, c, t), self.RateFcRn_free_other(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p(self, y, w, c, t):
		return (1 / c[1]) * (-(c[8] + c[10]) * (y[0]/3126.0) + (c[71] - c[46]) * (y[25]/341.0) + (c[89] - c[47]) * (y[37]/332.0) + (c[24] - c[26]) * (y[9]/31.9) + (c[80] - c[48]) * (y[31]/30078.0) + (c[116] - c[51]) * (y[55]/10165.0) + (c[125] - c[52]) * (y[61]/6.41) + (c[98] - c[49]) * (y[43]/3408.0) + (c[107] - c[50]) * (y[49]/13465.0) + ((c[134] - c[53]) + (c[143] - c[54]) + (c[152] - c[55]) + (c[161] - c[56]) + (c[40] - c[42])) * (y[19]/2143.0) + (c[170] - c[57]) * (y[91]/4852.0) + c[58] * (y[2]/274.0))

	def RateC_bc(self, y, w, c, t):
		return (1 / c[2]) * (-c[9] * (y[1]/2558.0) + c[72] * (y[26]/13.1) + c[90] * (y[38]/18.2) + c[25] * (c[0]/26.1) + c[81] * (y[32]/662.0) + c[117] * (y[56]/224.0) + c[126] * (y[62]/0.353) + c[99] * (y[44]/127.0) + c[108] * (y[50]/148.0) + (c[135] + c[144] + c[153] + c[162] + c[41]) * (y[20]/183.0) + c[171] * (y[92]/204.0))

	def RateC_ln(self, y, w, c, t):
		return (1 / c[3]) * ((1 - c[12]) * c[10] * (y[7]/45.0) + (1 - c[74]) * c[46] * (y[29]/10.8) + (1 - c[92]) * c[47] * (y[39]/14.9) + (1 - c[29]) * c[32] * (y[18]/90.0) + (1 - c[30]) * c[33] * (y[12]/261.0) + (1 - c[83]) * c[48] * (y[33]/541.0) + (1 - c[119]) * c[51] * (y[57]/183.0) + (1 - c[128]) * c[52] * (y[63]/0.288) + (1 - c[101]) * c[49] * (y[45]/104.0) + (1 - c[110]) * c[50] * (y[51]/121.0) + (1 - c[137]) * c[53] * (y[69]/5.03) + (1 - c[146]) * c[54] * (y[75]/7.15) + (1 - c[155]) * c[55] * (y[81]/21.9) + (1 - c[164]) * c[56] * (y[87]/4.66) + (1 - c[44]) * c[42] * (y[21]/149.0) + (1 - c[173]) * c[57] * (y[93]/167.0) - c[58] * (y[2]/274.0))

	def RateC_p_lung(self, y, w, c, t):
		return (1 / c[178]) * ((c[8] * (y[0]/3126.0) - c[8] * (y[3]/1000.0) - (1 - c[11]) * c[10] * (y[3]/1000.0) - c[13] * (y[3]/1000.0)) + c[13] * c[62] * (y[6]/5.0))

	def RateC_bc_lung(self, y, w, c, t):
		return (1 / c[179]) * (c[9] * (y[1]/2558.0) - c[9] * (y[4]/55.0))

	def RateC_e_unbound_lung(self, y, w, c, t):
		return (1 / c[181]) * ((c[13] * ((y[3]/1000.0) + (y[7]/45.0)) - c[181] * c[59] * (y[5]/5.0) * (y[8]/5.0)) + c[181] * c[60] * (y[6]/5.0) - c[61] * (y[5]/5.0) * c[181])

	def RateC_e_bound_lung(self, y, w, c, t):
		return (1 / c[181]) * (c[181] * c[59] * (y[5]/5.0) * (y[8]/5.0) - c[181] * c[60] * (y[6]/5.0) - c[13] * (y[6]/5.0))

	def RateC_is_lung(self, y, w, c, t):
		return (1 / c[180]) * (((1 - c[11]) * c[10] * (y[3]/1000.0) - (1 - c[12]) * c[10] * (y[7]/45.0)) + c[13] * (1 - c[62]) * (y[6]/5.0) - c[13] * (y[7]/45.0))

	def RateFcRn_free_lung(self, y, w, c, t):
		return (1 / c[181]) * ((c[60] * (y[6]/5.0) * c[181] - c[59] * (y[5]/5.0) * (y[8]/5.0) * c[181]) + c[13] * (y[6]/5.0))

	def RateC_p_brain(self, y, w, c, t):
		return (1 / c[182]) * ((c[24] * (y[3]/1000.0) - (c[24] - c[26]) * (y[9]/31.9) - (1 - c[34]) * c[33] * (y[9]/31.9) - (1 - c[35]) * c[32] * (y[9]/31.9) - c[31] * c[186] * (y[9]/31.9)) + c[31] * c[63] * c[186] * c[62] * (y[11]/0.1) + c[31] * (1 - c[63]) * c[186] * c[62] * (y[14]/0.1))

	def RateC_BBB_unbound_brain(self, y, w, c, t):
		return (1 / c[184]) * ((c[31] * c[63] * c[186] * ((y[9]/31.9) + (y[12]/261.0)) - c[184] * c[59] * (y[10]/0.1) * c[65]) + c[184] * c[60] * (y[11]/0.1) - c[184] * c[61] * (y[10]/0.1))

	def RateC_BBB_bound_brain(self, y, w, c, t):
		return (1 / c[184]) * (-c[31] * c[63] * c[186] * (y[11]/0.1) + c[184] * c[59] * (y[10]/0.1) * c[65] - c[184] * c[60] * (y[11]/0.1))

	def RateC_is_brain(self, y, w, c, t):
		return (1 / c[183]) * (((1 - c[34]) * c[33] * (y[9]/31.9) - (1 - c[30]) * c[33] * (y[12]/261.0) - c[33] * (y[12]/261.0)) + c[33] * (y[18]/90.0) + c[31] * c[63] * c[186] * (1 - c[62]) * (y[11]/0.1) - c[31] * c[63] * c[186] * (y[12]/261.0))

	def RateC_BCSFB_unbound_brain(self, y, w, c, t):
		return (1 / c[187]) * ((c[31] * (1 - c[63]) * c[186] * ((y[9]/31.9) + (y[15]/22.5) + (y[16]/22.5)) - c[187] * c[59] * (y[13]/0.1) * c[66]) + c[187] * c[60] * (y[14]/0.1) - c[187] * c[61] * (y[13]/0.1))

	def RateC_BCSFB_bound_brain(self, y, w, c, t):
		return (1 / c[187]) * (-c[31] * (1 - c[63]) * c[186] * (y[14]/0.1) + c[187] * c[59] * (y[13]/0.1) * c[66] - c[187] * c[60] * (y[14]/0.1))

	def RateC_LV_brain(self, y, w, c, t):
		return (1 / c[188]) * (((1 - c[35]) * c[64] * c[32] * (y[9]/31.9) + c[64] * c[33] * (y[12]/261.0) - (c[64] * c[32] + c[64] * c[33]) * (y[15]/22.5) - c[64] * c[31] * (1 - c[63]) * c[186] * (y[15]/22.5)) + c[64] * c[31] * (1 - c[63]) * c[186] * (1 - c[62]) * (y[14]/0.1))

	def RateC_TFV_brain(self, y, w, c, t):
		return (1 / c[189]) * (((1 - c[35]) * (1 - c[64]) * c[32] * (y[9]/31.9) + (1 - c[64]) * c[33] * (y[12]/261.0) - (c[32] + c[33]) * (y[16]/22.5) - (1 - c[64]) * c[31] * (1 - c[63]) * c[186] * (y[16]/22.5)) + (1 - c[64]) * c[31] * (1 - c[63]) * c[186] * (1 - c[62]) * (y[14]/0.1) + (c[64] * c[32] + c[64] * c[33]) * (y[15]/22.5))

	def RateC_CM_brain(self, y, w, c, t):
		return (1 / c[190]) * (c[32] + c[33]) * ((y[16]/22.5) - (y[17]/7.5))

	def RateC_SAS_brain(self, y, w, c, t):
		return (1 / c[191]) * ((c[32] + c[33]) * (y[17]/7.5) - (1 - c[29]) * c[32] * (y[18]/90.0) - c[33] * (y[18]/90.0))

	def RateC_p_liver(self, y, w, c, t):
		return (1 / c[192]) * ((c[40] * (y[3]/1000.0) + c[152] * (y[79]/221.0) + c[161] * (y[85]/104.0) + c[134] * (y[67]/385.0) + c[143] * (y[73]/548.0) - ((c[40] - c[42]) + (c[152] - c[55]) + (c[161] - c[56]) + (c[134] - c[53]) + (c[143] - c[54])) * (y[19]/2143.0) - (1 - c[43]) * c[42] * (y[19]/2143.0) - c[45] * (y[19]/2143.0)) + c[45] * c[62] * (y[23]/10.7))

	def RateC_bc_liver(self, y, w, c, t):
		return (1 / c[193]) * (c[41] * (y[4]/55.0) + c[153] * (y[80]/26.8) + c[162] * (y[86]/5.7) + c[135] * (y[68]/6.15) + c[144] * (y[74]/8.74) - (c[41] + c[153] + c[162] + c[135] + c[144]) * (y[20]/183.0))

	def RateC_is_liver(self, y, w, c, t):
		return (1 / c[194]) * (((1 - c[43]) * c[42] * (y[19]/2143.0) - (1 - c[44]) * c[42] * (y[21]/149.0)) + c[45] * (1 - c[62]) * (y[23]/10.7) - c[45] * (y[21]/149.0))

	def RateC_e_unbound_liver(self, y, w, c, t):
		return (1 / c[195]) * ((c[45] * ((y[19]/2143.0) + (y[21]/149.0)) - c[195] * c[59] * (y[22]/10.7) * (y[24]/10.7)) + c[195] * c[60] * (y[23]/10.7) - c[61] * (y[22]/10.7) * c[195])

	def RateC_e_bound_liver(self, y, w, c, t):
		return (1 / c[195]) * (c[195] * c[59] * (y[22]/10.7) * (y[24]/10.7) - c[195] * c[60] * (y[23]/10.7) - c[45] * (y[23]/10.7))

	def RateFcRn_free_liver(self, y, w, c, t):
		return (1 / c[195]) * ((c[60] * (y[23]/10.7) * c[195] - c[59] * (y[22]/10.7) * (y[24]/10.7) * c[195]) + c[45] * (y[23]/10.7))

	def RateC_p_heart(self, y, w, c, t):
		return (1 / c[196]) * ((c[71] * (y[3]/1000.0) - c[71] * (y[25]/341.0) - (1 - c[73]) * c[46] * (y[25]/341.0) - c[75] * (y[25]/341.0)) + c[75] * c[62] * (y[28]/1.71))

	def RateC_bc_heart(self, y, w, c, t):
		return (1 / c[197]) * (c[72] * (y[4]/55.0) - c[72] * (y[26]/13.1))

	def RateC_e_unbound_heart(self, y, w, c, t):
		return (1 / c[199]) * ((c[75] * ((y[25]/341.0) + (y[29]/10.8)) - c[199] * c[59] * (y[27]/1.71) * (y[30]/1.71)) + c[199] * c[60] * (y[28]/1.71) - c[61] * (y[27]/1.71) * c[199])

	def RateC_e_bound_heart(self, y, w, c, t):
		return (1 / c[199]) * (c[199] * c[59] * (y[27]/1.71) * (y[30]/1.71) - c[199] * c[60] * (y[28]/1.71) - c[75] * (y[28]/1.71))

	def RateC_is_heart(self, y, w, c, t):
		return (1 / c[198]) * (((1 - c[73]) * c[46] * (y[25]/341.0) - (1 - c[74]) * c[46] * (y[29]/10.8)) + c[75] * (1 - c[62]) * (y[28]/1.71) - c[75] * (y[29]/10.8))

	def RateFcRn_free_heart(self, y, w, c, t):
		return (1 / c[199]) * ((c[60] * (y[28]/1.71) * c[199] - c[59] * (y[27]/1.71) * (y[30]/1.71) * c[199]) + c[75] * (y[28]/1.71))

	def RateC_p_muscle(self, y, w, c, t):
		return (1 / c[200]) * ((c[80] * (y[3]/1000.0) - c[80] * (y[31]/30078.0) - (1 - c[82]) * c[48] * (y[31]/30078.0) - c[84] * (y[31]/30078.0)) + c[84] * c[62] * (y[35]/150.0))

	def RateC_bc_muscle(self, y, w, c, t):
		return (1 / c[201]) * (c[81] * (y[4]/55.0) - c[81] * (y[32]/662.0))

	def RateC_is_muscle(self, y, w, c, t):
		return (1 / c[202]) * (((1 - c[82]) * c[48] * (y[31]/30078.0) - (1 - c[83]) * c[48] * (y[33]/541.0)) + c[84] * (1 - c[62]) * (y[35]/150.0) - c[84] * (y[33]/541.0))

	def RateC_e_unbound_muscle(self, y, w, c, t):
		return (1 / c[203]) * ((c[84] * ((y[31]/30078.0) + (y[33]/541.0)) - c[203] * c[59] * (y[34]/150.0) * (y[36]/150.0)) + c[203] * c[60] * (y[35]/150.0) - c[61] * (y[34]/150.0) * c[203])

	def RateC_e_bound_muscle(self, y, w, c, t):
		return (1 / c[203]) * (c[203] * c[59] * (y[34]/150.0) * (y[36]/150.0) - c[203] * c[60] * (y[35]/150.0) - c[84] * (y[35]/150.0))

	def RateFcRn_free_muscle(self, y, w, c, t):
		return (1 / c[203]) * ((c[60] * (y[35]/150.0) * c[203] - c[59] * (y[34]/150.0) * (y[36]/150.0) * c[203]) + c[84] * (y[35]/150.0))

	def RateC_p_heart(self, y, w, c, t):
		return (1 / c[196]) * ((c[71] * (y[3]/1000.0) - c[71] * (y[25]/341.0) - (1 - c[73]) * c[46] * (y[25]/341.0) - c[75] * (y[25]/341.0)) + c[75] * c[62] * (y[28]/1.71))

	def RateC_bc_heart(self, y, w, c, t):
		return (1 / c[197]) * (c[72] * (y[4]/55.0) - c[72] * (y[26]/13.1))

	def RateC_is_heart(self, y, w, c, t):
		return (1 / c[198]) * (((1 - c[73]) * c[46] * (y[25]/341.0) - (1 - c[74]) * c[46] * (y[29]/10.8)) + c[75] * (1 - c[62]) * (y[28]/1.71) - c[75] * (y[29]/10.8))

	def RateC_e_unbound_heart(self, y, w, c, t):
		return (1 / c[199]) * ((c[75] * ((y[25]/341.0) + (y[29]/10.8)) - c[199] * c[59] * (y[27]/1.71) * (y[30]/1.71)) + c[199] * c[60] * (y[28]/1.71) - c[61] * (y[27]/1.71) * c[199])

	def RateC_e_bound_heart(self, y, w, c, t):
		return (1 / c[199]) * (c[199] * c[59] * (y[27]/1.71) * (y[30]/1.71) - c[199] * c[60] * (y[28]/1.71) - c[75] * (y[28]/1.71))

	def RateFcRn_free_heart(self, y, w, c, t):
		return (1 / c[199]) * ((c[60] * (y[28]/1.71) * c[199] - c[59] * (y[27]/1.71) * (y[30]/1.71) * c[199]) + c[75] * (y[28]/1.71))

	def RateC_p_muscle(self, y, w, c, t):
		return (1 / c[200]) * ((c[80] * (y[3]/1000.0) - c[80] * (y[31]/30078.0) - (1 - c[82]) * c[48] * (y[31]/30078.0) - c[84] * (y[31]/30078.0)) + c[84] * c[62] * (y[35]/150.0))

	def RateC_bc_muscle(self, y, w, c, t):
		return (1 / c[201]) * (c[81] * (y[4]/55.0) - c[81] * (y[32]/662.0))

	def RateC_is_muscle(self, y, w, c, t):
		return (1 / c[202]) * (((1 - c[82]) * c[48] * (y[31]/30078.0) - (1 - c[83]) * c[48] * (y[33]/541.0)) + c[84] * (1 - c[62]) * (y[35]/150.0) - c[84] * (y[33]/541.0))

	def RateC_e_unbound_muscle(self, y, w, c, t):
		return (1 / c[203]) * ((c[84] * ((y[31]/30078.0) + (y[33]/541.0)) - c[203] * c[59] * (y[34]/150.0) * (y[36]/150.0)) + c[203] * c[60] * (y[35]/150.0) - c[61] * (y[34]/150.0) * c[203])

	def RateC_e_bound_muscle(self, y, w, c, t):
		return (1 / c[203]) * (c[203] * c[59] * (y[34]/150.0) * (y[36]/150.0) - c[203] * c[60] * (y[35]/150.0) - c[84] * (y[35]/150.0))

	def RateFcRn_free_muscle(self, y, w, c, t):
		return (1 / c[203]) * ((c[60] * (y[35]/150.0) * c[203] - c[59] * (y[34]/150.0) * (y[36]/150.0) * c[203]) + c[84] * (y[35]/150.0))

	def RateC_p_kidney(self, y, w, c, t):
		return (1 / c[204]) * ((c[89] * (y[3]/1000.0) - c[89] * (y[37]/332.0) - (1 - c[91]) * c[47] * (y[37]/332.0) - c[93] * (y[37]/332.0)) + c[93] * c[62] * (y[41]/1.66))

	def RateC_bc_kidney(self, y, w, c, t):
		return (1 / c[205]) * (c[90] * (y[4]/55.0) - c[90] * (y[38]/18.2))

	def RateC_is_kidney(self, y, w, c, t):
		return (1 / c[206]) * (((1 - c[91]) * c[47] * (y[37]/332.0) - (1 - c[92]) * c[47] * (y[39]/14.9)) + c[93] * (1 - c[62]) * (y[41]/1.66) - c[93] * (y[39]/14.9))

	def RateC_e_unbound_kidney(self, y, w, c, t):
		return (1 / c[207]) * ((c[93] * ((y[37]/332.0) + (y[39]/14.9)) - c[207] * c[59] * (y[40]/1.66) * (y[42]/1.66)) + c[207] * c[60] * (y[41]/1.66) - c[61] * (y[40]/1.66) * c[207])

	def RateC_e_bound_kidney(self, y, w, c, t):
		return (1 / c[207]) * (c[207] * c[59] * (y[40]/1.66) * (y[42]/1.66) - c[207] * c[60] * (y[41]/1.66) - c[93] * (y[41]/1.66))

	def RateFcRn_free_kidney(self, y, w, c, t):
		return (1 / c[207]) * ((c[60] * (y[41]/1.66) * c[207] - c[59] * (y[40]/1.66) * (y[42]/1.66) * c[207]) + c[93] * (y[41]/1.66))

	def RateC_p_skin(self, y, w, c, t):
		return (1 / c[208]) * ((c[98] * (y[3]/1000.0) - c[98] * (y[43]/3408.0) - (1 - c[100]) * c[49] * (y[43]/3408.0) - c[102] * (y[43]/3408.0)) + c[102] * c[62] * (y[47]/17.0))

	def RateC_bc_skin(self, y, w, c, t):
		return (1 / c[209]) * (c[99] * (y[4]/55.0) - c[99] * (y[44]/127.0))

	def RateC_is_skin(self, y, w, c, t):
		return (1 / c[210]) * (((1 - c[100]) * c[49] * (y[43]/3408.0) - (1 - c[101]) * c[49] * (y[45]/104.0)) + c[102] * (1 - c[62]) * (y[47]/17.0) - c[102] * (y[45]/104.0))

	def RateC_e_unbound_skin(self, y, w, c, t):
		return (1 / c[211]) * ((c[102] * ((y[43]/3408.0) + (y[45]/104.0)) - c[211] * c[59] * (y[46]/17.0) * (y[48]/17.0)) + c[211] * c[60] * (y[47]/17.0) - c[61] * (y[46]/17.0) * c[211])

	def RateC_e_bound_skin(self, y, w, c, t):
		return (1 / c[211]) * (c[211] * c[59] * (y[46]/17.0) * (y[48]/17.0) - c[211] * c[60] * (y[47]/17.0) - c[102] * (y[47]/17.0))

	def RateFcRn_free_skin(self, y, w, c, t):
		return (1 / c[211]) * ((c[60] * (y[47]/17.0) * c[211] - c[59] * (y[46]/17.0) * (y[48]/17.0) * c[211]) + c[102] * (y[47]/17.0))

	def RateC_p_fat(self, y, w, c, t):
		return (1 / c[212]) * ((c[107] * (y[3]/1000.0) - c[107] * (y[49]/13465.0) - (1 - c[109]) * c[50] * (y[49]/13465.0) - c[111] * (y[49]/13465.0)) + c[111] * c[62] * (y[53]/67.3))

	def RateC_bc_fat(self, y, w, c, t):
		return (1 / c[213]) * (c[108] * (y[4]/55.0) - c[108] * (y[50]/148.0))

	def RateC_is_fat(self, y, w, c, t):
		return (1 / c[214]) * (((1 - c[109]) * c[50] * (y[49]/13465.0) - (1 - c[110]) * c[50] * (y[51]/121.0)) + c[111] * (1 - c[62]) * (y[53]/67.3) - c[111] * (y[51]/121.0))

	def RateC_e_unbound_fat(self, y, w, c, t):
		return (1 / c[215]) * ((c[111] * ((y[49]/13465.0) + (y[51]/121.0)) - c[215] * c[59] * (y[52]/67.3) * (y[54]/67.3)) + c[215] * c[60] * (y[53]/67.3) - c[61] * (y[52]/67.3) * c[215])

	def RateC_e_bound_fat(self, y, w, c, t):
		return (1 / c[215]) * (c[215] * c[59] * (y[52]/67.3) * (y[54]/67.3) - c[215] * c[60] * (y[53]/67.3) - c[111] * (y[53]/67.3))

	def RateFcRn_free_fat(self, y, w, c, t):
		return (1 / c[215]) * ((c[60] * (y[53]/67.3) * c[215] - c[59] * (y[52]/67.3) * (y[54]/67.3) * c[215]) + c[111] * (y[53]/67.3))

	def RateC_p_marrow(self, y, w, c, t):
		return (1 / c[216]) * ((c[116] * (y[3]/1000.0) - c[116] * (y[55]/10165.0) - (1 - c[118]) * c[51] * (y[55]/10165.0) - c[120] * (y[55]/10165.0)) + c[120] * c[62] * (y[59]/50.8))

	def RateC_bc_marrow(self, y, w, c, t):
		return (1 / c[217]) * (c[117] * (y[4]/55.0) - c[117] * (y[56]/224.0))

	def RateC_is_marrow(self, y, w, c, t):
		return (1 / c[218]) * (((1 - c[118]) * c[51] * (y[55]/10165.0) - (1 - c[119]) * c[51] * (y[57]/183.0)) + c[120] * (1 - c[62]) * (y[59]/50.8) - c[120] * (y[57]/183.0))

	def RateC_e_unbound_marrow(self, y, w, c, t):
		return (1 / c[219]) * ((c[120] * ((y[55]/10165.0) + (y[57]/183.0)) - c[219] * c[59] * (y[58]/50.8) * (y[60]/50.8)) + c[219] * c[60] * (y[59]/50.8) - c[61] * (y[58]/50.8) * c[219])

	def RateC_e_bound_marrow(self, y, w, c, t):
		return (1 / c[219]) * (c[219] * c[59] * (y[58]/50.8) * (y[60]/50.8) - c[219] * c[60] * (y[59]/50.8) - c[120] * (y[59]/50.8))

	def RateFcRn_free_marrow(self, y, w, c, t):
		return (1 / c[219]) * ((c[60] * (y[59]/50.8) * c[219] - c[59] * (y[58]/50.8) * (y[60]/50.8) * c[219]) + c[120] * (y[59]/50.8))

	def RateC_p_thymus(self, y, w, c, t):
		return (1 / c[220]) * ((c[125] * (y[3]/1000.0) - c[125] * (y[61]/6.41) - (1 - c[127]) * c[52] * (y[61]/6.41) - c[129] * (y[61]/6.41)) + c[129] * c[62] * (y[65]/0.0321))

	def RateC_bc_thymus(self, y, w, c, t):
		return (1 / c[221]) * (c[126] * (y[4]/55.0) - c[126] * (y[62]/0.353))

	def RateC_is_thymus(self, y, w, c, t):
		return (1 / c[222]) * (((1 - c[127]) * c[52] * (y[61]/6.41) - (1 - c[128]) * c[52] * (y[63]/0.288)) + c[129] * (1 - c[62]) * (y[65]/0.0321) - c[129] * (y[63]/0.288))

	def RateC_e_unbound_thymus(self, y, w, c, t):
		return (1 / c[223]) * ((c[129] * ((y[61]/6.41) + (y[63]/0.288)) - c[223] * c[59] * (y[64]/0.0321) * (y[66]/0.0321)) + c[223] * c[60] * (y[65]/0.0321) - c[61] * (y[64]/0.0321) * c[223])

	def RateC_e_bound_thymus(self, y, w, c, t):
		return (1 / c[223]) * (c[223] * c[59] * (y[64]/0.0321) * (y[66]/0.0321) - c[223] * c[60] * (y[65]/0.0321) - c[129] * (y[65]/0.0321))

	def RateFcRn_free_thymus(self, y, w, c, t):
		return (1 / c[223]) * ((c[60] * (y[65]/0.0321) * c[223] - c[59] * (y[64]/0.0321) * (y[66]/0.0321) * c[223]) + c[129] * (y[65]/0.0321))

	def RateC_p_SI(self, y, w, c, t):
		return (1 / c[224]) * ((c[134] * (y[3]/1000.0) - c[134] * (y[67]/385.0) - (1 - c[136]) * c[53] * (y[67]/385.0) - c[138] * (y[67]/385.0)) + c[138] * c[62] * (y[71]/1.93))

	def RateC_bc_SI(self, y, w, c, t):
		return (1 / c[225]) * (c[135] * (y[4]/55.0) - c[135] * (y[68]/6.15))

	def RateC_is_SI(self, y, w, c, t):
		return (1 / c[226]) * (((1 - c[136]) * c[53] * (y[67]/385.0) - (1 - c[137]) * c[53] * (y[69]/5.03)) + c[138] * (1 - c[62]) * (y[71]/1.93) - c[138] * (y[69]/5.03))

	def RateC_e_unbound_SI(self, y, w, c, t):
		return (1 / c[227]) * ((c[138] * ((y[67]/385.0) + (y[69]/5.03)) - c[227] * c[59] * (y[70]/1.93) * (y[72]/1.93)) + c[227] * c[60] * (y[71]/1.93) - c[61] * (y[70]/1.93) * c[227])

	def RateC_e_bound_SI(self, y, w, c, t):
		return (1 / c[227]) * (c[227] * c[59] * (y[70]/1.93) * (y[72]/1.93) - c[227] * c[60] * (y[71]/1.93) - c[138] * (y[71]/1.93))

	def RateFcRn_free_SI(self, y, w, c, t):
		return (1 / c[227]) * ((c[60] * (y[71]/1.93) * c[227] - c[59] * (y[70]/1.93) * (y[72]/1.93) * c[227]) + c[138] * (y[71]/1.93))

	def RateC_p_LI(self, y, w, c, t):
		return (1 / c[228]) * ((c[143] * (y[3]/1000.0) - c[143] * (y[73]/548.0) - (1 - c[145]) * c[54] * (y[73]/548.0) - c[147] * (y[73]/548.0)) + c[147] * c[62] * (y[77]/2.74))

	def RateC_bc_LI(self, y, w, c, t):
		return (1 / c[229]) * (c[144] * (y[4]/55.0) - c[144] * (y[74]/8.74))

	def RateC_is_LI(self, y, w, c, t):
		return (1 / c[230]) * (((1 - c[145]) * c[54] * (y[73]/548.0) - (1 - c[146]) * c[54] * (y[75]/7.15)) + c[147] * (1 - c[62]) * (y[77]/2.74) - c[147] * (y[75]/7.15))

	def RateC_e_unbound_LI(self, y, w, c, t):
		return (1 / c[231]) * ((c[147] * ((y[73]/548.0) + (y[75]/7.15)) - c[231] * c[59] * (y[76]/2.74) * (y[78]/2.74)) + c[231] * c[60] * (y[77]/2.74) - c[61] * (y[76]/2.74) * c[231])

	def RateC_e_bound_LI(self, y, w, c, t):
		return (1 / c[231]) * (c[231] * c[59] * (y[76]/2.74) * (y[78]/2.74) - c[231] * c[60] * (y[77]/2.74) - c[147] * (y[77]/2.74))

	def RateFcRn_free_LI(self, y, w, c, t):
		return (1 / c[231]) * ((c[60] * (y[77]/2.74) * c[231] - c[59] * (y[76]/2.74) * (y[78]/2.74) * c[231]) + c[147] * (y[77]/2.74))

	def RateC_p_spleen(self, y, w, c, t):
		return (1 / c[232]) * ((c[152] * (y[3]/1000.0) - c[152] * (y[79]/221.0) - (1 - c[154]) * c[55] * (y[79]/221.0) - c[156] * (y[79]/221.0)) + c[156] * c[62] * (y[83]/1.11))

	def RateC_bc_spleen(self, y, w, c, t):
		return (1 / c[233]) * (c[153] * (y[4]/55.0) - c[153] * (y[80]/26.8))

	def RateC_is_spleen(self, y, w, c, t):
		return (1 / c[234]) * (((1 - c[154]) * c[55] * (y[79]/221.0) - (1 - c[155]) * c[55] * (y[81]/21.9)) + c[156] * (1 - c[62]) * (y[83]/1.11) - c[156] * (y[81]/21.9))

	def RateC_e_unbound_spleen(self, y, w, c, t):
		return (1 / c[235]) * ((c[156] * ((y[79]/221.0) + (y[81]/21.9)) - c[235] * c[59] * (y[82]/1.11) * (y[84]/1.11)) + c[235] * c[60] * (y[83]/1.11) - c[61] * (y[82]/1.11) * c[235])

	def RateC_e_bound_spleen(self, y, w, c, t):
		return (1 / c[235]) * (c[235] * c[59] * (y[82]/1.11) * (y[84]/1.11) - c[235] * c[60] * (y[83]/1.11) - c[156] * (y[83]/1.11))

	def RateFcRn_free_spleen(self, y, w, c, t):
		return (1 / c[235]) * ((c[60] * (y[83]/1.11) * c[235] - c[59] * (y[82]/1.11) * (y[84]/1.11) * c[235]) + c[156] * (y[83]/1.11))

	def RateC_p_pancreas(self, y, w, c, t):
		return (1 / c[236]) * ((c[161] * (y[3]/1000.0) - c[161] * (y[85]/104.0) - (1 - c[163]) * c[56] * (y[85]/104.0) - c[165] * (y[85]/104.0)) + c[165] * c[62] * (y[89]/0.518))

	def RateC_bc_pancreas(self, y, w, c, t):
		return (1 / c[237]) * (c[162] * (y[4]/55.0) - c[162] * (y[86]/5.7))

	def RateC_is_pancreas(self, y, w, c, t):
		return (1 / c[238]) * (((1 - c[163]) * c[56] * (y[85]/104.0) - (1 - c[164]) * c[56] * (y[87]/4.66)) + c[165] * (1 - c[62]) * (y[89]/0.518) - c[165] * (y[87]/4.66))

	def RateC_e_unbound_pancreas(self, y, w, c, t):
		return (1 / c[239]) * ((c[165] * ((y[85]/104.0) + (y[87]/4.66)) - c[239] * c[59] * (y[88]/0.518) * (y[90]/0.518)) + c[239] * c[60] * (y[89]/0.518) - c[61] * (y[88]/0.518) * c[239])

	def RateC_e_bound_pancreas(self, y, w, c, t):
		return (1 / c[239]) * (c[239] * c[59] * (y[88]/0.518) * (y[90]/0.518) - c[239] * c[60] * (y[89]/0.518) - c[165] * (y[89]/0.518))

	def RateFcRn_free_pancreas(self, y, w, c, t):
		return (1 / c[239]) * ((c[60] * (y[89]/0.518) * c[239] - c[59] * (y[88]/0.518) * (y[90]/0.518) * c[239]) + c[165] * (y[89]/0.518))

	def RateC_p_other(self, y, w, c, t):
		return (1 / c[240]) * ((c[170] * (y[3]/1000.0) - c[170] * (y[91]/4852.0) - (1 - c[172]) * c[57] * (y[91]/4852.0) - c[174] * (y[91]/4852.0)) + c[174] * c[62] * (y[95]/24.3))

	def RateC_bc_other(self, y, w, c, t):
		return (1 / c[241]) * (c[171] * (y[4]/55.0) - c[171] * (y[92]/204.0))

	def RateC_is_other(self, y, w, c, t):
		return (1 / c[242]) * (((1 - c[172]) * c[57] * (y[91]/4852.0) - (1 - c[173]) * c[57] * (y[93]/167.0)) + c[174] * (1 - c[62]) * (y[95]/24.3) - c[174] * (y[93]/167.0))

	def RateC_e_unbound_other(self, y, w, c, t):
		return (1 / c[243]) * ((c[174] * ((y[91]/4852.0) + (y[93]/167.0)) - c[243] * c[59] * (y[94]/24.3) * (y[96]/24.3)) + c[243] * c[60] * (y[95]/24.3) - c[61] * (y[94]/24.3) * c[243])

	def RateC_e_bound_other(self, y, w, c, t):
		return (1 / c[243]) * (c[243] * c[59] * (y[94]/24.3) * (y[96]/24.3) - c[243] * c[60] * (y[95]/24.3) - c[174] * (y[95]/24.3))

	def RateFcRn_free_other(self, y, w, c, t):
		return (1 / c[243]) * ((c[60] * (y[95]/24.3) * c[243] - c[59] * (y[94]/24.3) * (y[96]/24.3) * c[243]) + c[174] * (y[95]/24.3))

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

	def __init__(self, y_indexes={'C_p': 0, 'C_bc': 1, 'C_ln': 2, 'C_p_lung': 3, 'C_bc_lung': 4, 'C_e_unbound_lung': 5, 'C_e_bound_lung': 6, 'C_is_lung': 7, 'FcRn_free_lung': 8, 'C_p_brain': 9, 'C_BBB_unbound_brain': 10, 'C_BBB_bound_brain': 11, 'C_is_brain': 12, 'C_BCSFB_unbound_brain': 13, 'C_BCSFB_bound_brain': 14, 'C_LV_brain': 15, 'C_TFV_brain': 16, 'C_CM_brain': 17, 'C_SAS_brain': 18, 'C_p_liver': 19, 'C_bc_liver': 20, 'C_is_liver': 21, 'C_e_unbound_liver': 22, 'C_e_bound_liver': 23, 'FcRn_free_liver': 24, 'C_p_heart': 25, 'C_bc_heart': 26, 'C_e_unbound_heart': 27, 'C_e_bound_heart': 28, 'C_is_heart': 29, 'FcRn_free_heart': 30, 'C_p_muscle': 31, 'C_bc_muscle': 32, 'C_is_muscle': 33, 'C_e_unbound_muscle': 34, 'C_e_bound_muscle': 35, 'FcRn_free_muscle': 36, 'C_p_kidney': 37, 'C_bc_kidney': 38, 'C_is_kidney': 39, 'C_e_unbound_kidney': 40, 'C_e_bound_kidney': 41, 'FcRn_free_kidney': 42, 'C_p_skin': 43, 'C_bc_skin': 44, 'C_is_skin': 45, 'C_e_unbound_skin': 46, 'C_e_bound_skin': 47, 'FcRn_free_skin': 48, 'C_p_fat': 49, 'C_bc_fat': 50, 'C_is_fat': 51, 'C_e_unbound_fat': 52, 'C_e_bound_fat': 53, 'FcRn_free_fat': 54, 'C_p_marrow': 55, 'C_bc_marrow': 56, 'C_is_marrow': 57, 'C_e_unbound_marrow': 58, 'C_e_bound_marrow': 59, 'FcRn_free_marrow': 60, 'C_p_thymus': 61, 'C_bc_thymus': 62, 'C_is_thymus': 63, 'C_e_unbound_thymus': 64, 'C_e_bound_thymus': 65, 'FcRn_free_thymus': 66, 'C_p_SI': 67, 'C_bc_SI': 68, 'C_is_SI': 69, 'C_e_unbound_SI': 70, 'C_e_bound_SI': 71, 'FcRn_free_SI': 72, 'C_p_LI': 73, 'C_bc_LI': 74, 'C_is_LI': 75, 'C_e_unbound_LI': 76, 'C_e_bound_LI': 77, 'FcRn_free_LI': 78, 'C_p_spleen': 79, 'C_bc_spleen': 80, 'C_is_spleen': 81, 'C_e_unbound_spleen': 82, 'C_e_bound_spleen': 83, 'FcRn_free_spleen': 84, 'C_p_pancreas': 85, 'C_bc_pancreas': 86, 'C_is_pancreas': 87, 'C_e_unbound_pancreas': 88, 'C_e_bound_pancreas': 89, 'FcRn_free_pancreas': 90, 'C_p_other': 91, 'C_bc_other': 92, 'C_is_other': 93, 'C_e_unbound_other': 94, 'C_e_bound_other': 95, 'FcRn_free_other': 96}, w_indexes={}, c_indexes={'C_bc_brain': 0, 'Vp': 1, 'Vbc': 2, 'Vlymphnode': 3, 'Vp_lung': 178, 'VBC_lung': 179, 'VIS_lung': 180, 'VES_lung': 181, 'Q_p_lung': 8, 'Q_bc_lung': 9, 'L_lung': 10, 'sigma_V_lung': 11, 'sigma_L_lung': 12, 'CLup_lung': 13, 'Vp_brain': 182, 'VIS_brain': 183, 'VBBB_brain': 184, 'VBC_brain': 185, 'V_ES_brain': 186, 'V_BCSFB_brain': 187, 'V_LV_brain': 188, 'V_TFV_brain': 189, 'V_CM_brain': 190, 'V_SAS_brain': 191, 'Q_p_brain': 24, 'Q_bc_brain': 25, 'L_brain': 26, 'sigma_V_brain': 27, 'sigma_L_brain': 28, 'sigma_L_SAS': 29, 'sigma_L_brain_ISF': 30, 'CLup_brain': 31, 'Q_CSF_brain': 32, 'Q_ISF_brain': 33, 'sigma_V_BBB': 34, 'sigma_V_BCSFB': 35, 'Vp_liver': 192, 'VBC_liver': 193, 'VIS_liver': 194, 'VES_liver': 195, 'Q_p_liver': 40, 'Q_bc_liver': 41, 'L_liver': 42, 'sigma_V_liver': 43, 'sigma_L_liver': 44, 'CLup_liver': 45, 'L_heart': 46, 'L_kidney': 47, 'L_muscle': 48, 'L_skin': 49, 'L_fat': 50, 'L_marrow': 51, 'L_thymus': 52, 'L_SI': 53, 'L_LI': 54, 'L_spleen': 55, 'L_pancreas': 56, 'L_other': 57, 'L_LN': 58, 'kon_FcRn': 59, 'koff_FcRn': 60, 'kdeg': 61, 'FR': 62, 'f_BBB': 63, 'f_LV': 64, 'FcRn_free_BBB': 65, 'FcRn_free_BCSFB': 66, 'Vp_heart': 196, 'VBC_heart': 197, 'VIS_heart': 198, 'VES_heart': 199, 'Q_p_heart': 71, 'Q_bc_heart': 72, 'sigma_V_heart': 73, 'sigma_L_heart': 74, 'CLup_heart': 75, 'Vp_muscle': 200, 'VBC_muscle': 201, 'VIS_muscle': 202, 'VES_muscle': 203, 'Q_p_muscle': 80, 'Q_bc_muscle': 81, 'sigma_V_muscle': 82, 'sigma_L_muscle': 83, 'CLup_muscle': 84, 'Vp_kidney': 204, 'VBC_kidney': 205, 'VIS_kidney': 206, 'VES_kidney': 207, 'Q_p_kidney': 89, 'Q_bc_kidney': 90, 'sigma_V_kidney': 91, 'sigma_L_kidney': 92, 'CLup_kidney': 93, 'Vp_skin': 208, 'VBC_skin': 209, 'VIS_skin': 210, 'VES_skin': 211, 'Q_p_skin': 98, 'Q_bc_skin': 99, 'sigma_V_skin': 100, 'sigma_L_skin': 101, 'CLup_skin': 102, 'Vp_fat': 212, 'VBC_fat': 213, 'VIS_fat': 214, 'VES_fat': 215, 'Q_p_fat': 107, 'Q_bc_fat': 108, 'sigma_V_fat': 109, 'sigma_L_fat': 110, 'CLup_fat': 111, 'Vp_marrow': 216, 'VBC_marrow': 217, 'VIS_marrow': 218, 'VES_marrow': 219, 'Q_p_marrow': 116, 'Q_bc_marrow': 117, 'sigma_V_marrow': 118, 'sigma_L_marrow': 119, 'CLup_marrow': 120, 'Vp_thymus': 220, 'VBC_thymus': 221, 'VIS_thymus': 222, 'VES_thymus': 223, 'Q_p_thymus': 125, 'Q_bc_thymus': 126, 'sigma_V_thymus': 127, 'sigma_L_thymus': 128, 'CLup_thymus': 129, 'Vp_SI': 224, 'VBC_SI': 225, 'VIS_SI': 226, 'VES_SI': 227, 'Q_p_SI': 134, 'Q_bc_SI': 135, 'sigma_V_SI': 136, 'sigma_L_SI': 137, 'CLup_SI': 138, 'Vp_LI': 228, 'VBC_LI': 229, 'VIS_LI': 230, 'VES_LI': 231, 'Q_p_LI': 143, 'Q_bc_LI': 144, 'sigma_V_LI': 145, 'sigma_L_LI': 146, 'CLup_LI': 147, 'Vp_spleen': 232, 'VBC_spleen': 233, 'VIS_spleen': 234, 'VES_spleen': 235, 'Q_p_spleen': 152, 'Q_bc_spleen': 153, 'sigma_V_spleen': 154, 'sigma_L_spleen': 155, 'CLup_spleen': 156, 'Vp_pancreas': 236, 'VBC_pancreas': 237, 'VIS_pancreas': 238, 'VES_pancreas': 239, 'Q_p_pancreas': 161, 'Q_bc_pancreas': 162, 'sigma_V_pancreas': 163, 'sigma_L_pancreas': 164, 'CLup_pancreas': 165, 'Vp_other': 240, 'VBC_other': 241, 'VIS_other': 242, 'VES_other': 243, 'Q_p_other': 170, 'Q_bc_other': 171, 'sigma_V_other': 172, 'sigma_L_other': 173, 'CLup_other': 174, 'plasma': 175, 'blood_cells': 176, 'lymph_node': 177}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([14117.999999999989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000533074, 0.0, 0.0, 0.0, 0.0, 0.0, 8.51922e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0074730000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 8.27012e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00084694, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003352886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0025308559999999997, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5992219999999998e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 9.61526e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00013650680000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 5.53002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5806760000000002e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001210626]), w0=jnp.array([]), c=jnp.array([0.0, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 181913.0, 148920.0, 364.0, 0.95, 0.2, 0.55, 31.9, 261.0, 0.1, 26.1, 7.25, 0.1, 22.5, 22.5, 7.5, 90.0, 36402.0, 29810.0, 73.0, 0.95, 0.2, 0.2, 0.2, 0.3, 21.0, 10.5, 0.95, 0.9974, 2143.0, 183.0, 149.0, 10.7, 13210.0, 10820.0, 26.0, 0.95, 0.2, 0.55, 16.0, 65.0, 67.0, 23.0, 17.0, 5.0, 1.0, 25.0, 26.0, 13.0, 6.0, 11.0, 500.0, 559000000.0, 23.9, 26.6, 0.715, 0.95, 0.2, 1.0, 1.0, 341.0, 13.1, 10.8, 1.71, 7752.0, 6350.0, 0.95, 0.2, 0.55, 30078.0, 662.0, 541.0, 150.0, 33469.0, 27410.0, 0.95, 0.2, 0.55, 332.0, 18.2, 14.9, 1.66, 32402.0, 26530.0, 0.9, 0.2, 0.55, 3408.0, 127.0, 104.0, 17.0, 11626.0, 9520.0, 0.95, 0.2, 0.55, 13465.0, 148.0, 121.0, 67.3, 8343.0, 6830.0, 0.95, 0.2, 0.55, 10165.0, 224.0, 183.0, 50.8, 2591.0, 2120.0, 0.95, 0.2, 0.55, 6.41, 0.353, 0.288, 0.0321, 353.0, 289.0, 0.9, 0.2, 0.55, 385.0, 6.15, 5.03, 1.93, 12368.0, 10130.0, 0.9, 0.2, 0.55, 548.0, 8.74, 7.15, 2.74, 12867.0, 10530.0, 0.95, 0.2, 0.55, 221.0, 26.8, 21.9, 1.11, 6343.0, 5190.0, 0.85, 0.2, 0.55, 104.0, 5.7, 4.66, 0.518, 3056.0, 2500.0, 0.9, 0.2, 0.55, 4852.0, 204.0, 167.0, 24.3, 5521.0, 4520.0, 0.95, 0.2, 0.55, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 31.9, 261.0, 0.1, 26.1, 7.25, 0.1, 22.5, 22.5, 7.5, 90.0, 2143.0, 183.0, 149.0, 10.7, 341.0, 13.1, 10.8, 1.71, 30078.0, 662.0, 541.0, 150.0, 332.0, 18.2, 14.9, 1.66, 3408.0, 127.0, 104.0, 17.0, 13465.0, 148.0, 121.0, 67.3, 10165.0, 224.0, 183.0, 50.8, 6.41, 0.353, 0.288, 0.0321, 385.0, 6.15, 5.03, 1.93, 548.0, 8.74, 7.15, 2.74, 221.0, 26.8, 21.9, 1.11, 104.0, 5.7, 4.66, 0.518, 4852.0, 204.0, 167.0, 24.3]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

