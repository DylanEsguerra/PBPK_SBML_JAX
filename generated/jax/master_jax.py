import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([14117.999999999989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000533074])
y_indexes = {'C_p': 0, 'C_bc': 1, 'C_ln': 2, 'C_p_lung': 3, 'C_bc_lung': 4, 'C_e_unbound_lung': 5, 'C_e_bound_lung': 6, 'C_is_lung': 7, 'FcRn_free_lung': 8, 'C_p_brain': 9, 'C_BBB_unbound_brain': 10, 'C_BBB_bound_brain': 11, 'C_is_brain': 12, 'C_bc_brain': 13, 'C_BCSFB_unbound_brain': 14, 'C_BCSFB_bound_brain': 15, 'C_LV_brain': 16, 'C_TFV_brain': 17, 'C_CM_brain': 18, 'C_SAS_brain': 19, 'C_p_liver': 20, 'C_bc_liver': 21, 'C_is_liver': 22, 'C_e_unbound_liver': 23, 'C_e_bound_liver': 24, 'FcRn_free_liver': 25}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 8343.0, 2591.0, 32402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 36402.0, 21.0, 10.5, 6350.0, 148920.0, 27410.0, 9520.0, 6830.0, 2120.0, 26530.0, 10820.0, 10130.0, 10530.0, 2500.0, 289.0, 5190.0, 4520.0, 29810.0, 364.0, 16.0, 65.0, 73.0, 67.0, 5.0, 1.0, 23.0, 17.0, 25.0, 26.0, 13.0, 6.0, 26.0, 11.0, 500.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.51631477927063, 0.0, 0.0, 1000.0, 55.0, 45.0, 5.0, 559000000.0, 23.9, 26.6, 0.55, 0.715, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 31.9, 0.1, 261.0, 26.1, 7.25, 10.5, 0.3, 0.95, 1.0, 0.95, 0.9974, 0.1, 22.5, 22.5, 7.5, 90.0, 0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2143.0, 183.0, 149.0, 10.7, 0.55, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 31.9, 0.1, 0.1, 261.0, 26.1, 0.1, 0.1, 22.5, 22.5, 7.5, 90.0, 2143.0, 183.0, 149.0, 10.7]) 
c_indexes = {'C_p_heart': 0, 'C_p_kidney': 1, 'C_p_muscle': 2, 'C_p_marrow': 3, 'C_p_thymus': 4, 'C_p_skin': 5, 'C_p_fat': 6, 'C_p_other': 7, 'C_bc_heart': 8, 'C_bc_kidney': 9, 'C_bc_muscle': 10, 'C_bc_marrow': 11, 'C_bc_thymus': 12, 'C_bc_skin': 13, 'C_bc_fat': 14, 'C_bc_other': 15, 'C_is_heart': 16, 'C_is_kidney': 17, 'C_is_muscle': 18, 'C_is_marrow': 19, 'C_is_thymus': 20, 'C_is_skin': 21, 'C_is_fat': 22, 'C_is_SI': 23, 'C_is_LI': 24, 'C_is_spleen': 25, 'C_is_pancreas': 26, 'C_is_other': 27, 'Vp': 28, 'Vbc': 29, 'Vlymphnode': 30, 'Q_p_heart': 31, 'Q_p_lung': 32, 'Q_p_muscle': 33, 'Q_p_skin': 34, 'Q_p_fat': 35, 'Q_p_marrow': 36, 'Q_p_kidney': 37, 'Q_p_liver': 38, 'Q_p_SI': 39, 'Q_p_LI': 40, 'Q_p_pancreas': 41, 'Q_p_thymus': 42, 'Q_p_spleen': 43, 'Q_p_other': 44, 'Q_p_brain': 45, 'Q_CSF_brain': 46, 'Q_ECF_brain': 47, 'Q_bc_heart': 48, 'Q_bc_lung': 49, 'Q_bc_muscle': 50, 'Q_bc_skin': 51, 'Q_bc_fat': 52, 'Q_bc_marrow': 53, 'Q_bc_kidney': 54, 'Q_bc_liver': 55, 'Q_bc_SI': 56, 'Q_bc_LI': 57, 'Q_bc_pancreas': 58, 'Q_bc_thymus': 59, 'Q_bc_spleen': 60, 'Q_bc_other': 61, 'Q_bc_brain': 62, 'L_lung': 63, 'L_heart': 64, 'L_kidney': 65, 'L_brain': 66, 'L_muscle': 67, 'L_marrow': 68, 'L_thymus': 69, 'L_skin': 70, 'L_fat': 71, 'L_SI': 72, 'L_LI': 73, 'L_spleen': 74, 'L_pancreas': 75, 'L_liver': 76, 'L_other': 77, 'L_LN': 78, 'sigma_L_lung': 79, 'sigma_L_heart': 80, 'sigma_L_kidney': 81, 'sigma_L_brain_ISF': 82, 'sigma_L_muscle': 83, 'sigma_L_marrow': 84, 'sigma_L_thymus': 85, 'sigma_L_skin': 86, 'sigma_L_fat': 87, 'sigma_L_SI': 88, 'sigma_L_LI': 89, 'sigma_L_spleen': 90, 'sigma_L_pancreas': 91, 'sigma_L_liver': 92, 'sigma_L_other': 93, 'sigma_L_SAS': 94, 'C_p_0': 95, 'C_bc_0': 96, 'C_ln_0': 97, 'Vp_lung': 172, 'VBC_lung': 173, 'VIS_lung': 174, 'VES_lung': 175, 'kon_FcRn': 102, 'koff_FcRn': 103, 'kdeg': 104, 'CLup_lung': 105, 'FR': 106, 'sigma_V_lung': 107, 'C_p_lung_0': 108, 'C_bc_lung_0': 109, 'C_is_lung_0': 110, 'C_e_unbound_lung_0': 111, 'C_e_bound_lung_0': 112, 'FcRn_free_lung_0': 113, 'Vp_brain': 114, 'VBBB_brain': 115, 'VIS_brain': 116, 'VBC_brain': 117, 'V_ES_brain': 118, 'Q_ISF_brain': 119, 'CLup_brain': 120, 'f_BBB': 121, 'FcRn_free_BBB': 122, 'sigma_V_BBB': 123, 'sigma_V_BCSFB': 124, 'V_BCSFB_brain': 125, 'V_LV_brain': 126, 'V_TFV_brain': 127, 'V_CM_brain': 128, 'V_SAS_brain': 129, 'f_BCSFB': 130, 'FcRn_free_BCSFB': 131, 'f_LV': 132, 'C_BCSFB_unbound_brain_0': 133, 'C_BCSFB_bound_brain_0': 134, 'C_LV_brain_0': 135, 'C_TFV_brain_0': 136, 'C_CM_brain_0': 137, 'C_SAS_brain_0': 138, 'C_p_brain_0': 139, 'C_is_brain_0': 140, 'Vp_liver': 187, 'VBC_liver': 188, 'VIS_liver': 189, 'VES_liver': 190, 'CLup_liver': 145, 'sigma_V_liver': 146, 'C_p_liver_0': 147, 'C_bc_liver_0': 148, 'C_is_liver_0': 149, 'C_e_unbound_liver_0': 150, 'C_e_bound_liver_0': 151, 'FcRn_free_liver_0': 152, 'C_p_spleen_0': 153, 'C_bc_spleen_0': 154, 'C_p_pancreas_0': 155, 'C_bc_pancreas_0': 156, 'C_p_SI_0': 157, 'C_bc_SI_0': 158, 'C_p_LI_0': 159, 'C_bc_LI_0': 160, 'C_p_spleen': 161, 'C_bc_spleen': 162, 'C_p_pancreas': 163, 'C_bc_pancreas': 164, 'C_p_SI': 165, 'C_bc_SI': 166, 'C_p_LI': 167, 'C_bc_LI': 168, 'plasma': 169, 'blood_cells': 170, 'lymph_node': 171, 'brain_plasma': 176, 'BBB_unbound': 177, 'BBB_bound': 178, 'brain_ISF': 179, 'brain_blood_cells': 180, 'BCSFB_unbound': 181, 'BCSFB_bound': 182, 'LV': 183, 'TFV': 184, 'CM': 185, 'SAS': 186}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p(y, w, c, t), self.RateC_bc(y, w, c, t), self.RateC_ln(y, w, c, t), self.RateC_p_lung(y, w, c, t), self.RateC_bc_lung(y, w, c, t), self.RateC_e_unbound_lung(y, w, c, t), self.RateC_e_bound_lung(y, w, c, t), self.RateC_is_lung(y, w, c, t), self.RateFcRn_free_lung(y, w, c, t), self.RateC_p_brain(y, w, c, t), self.RateC_BBB_unbound_brain(y, w, c, t), self.RateC_BBB_bound_brain(y, w, c, t), self.RateC_is_brain(y, w, c, t), self.RateC_bc_brain(y, w, c, t), self.RateC_BCSFB_unbound_brain(y, w, c, t), self.RateC_BCSFB_bound_brain(y, w, c, t), self.RateC_LV_brain(y, w, c, t), self.RateC_TFV_brain(y, w, c, t), self.RateC_CM_brain(y, w, c, t), self.RateC_SAS_brain(y, w, c, t), self.RateC_p_liver(y, w, c, t), self.RateC_bc_liver(y, w, c, t), self.RateC_is_liver(y, w, c, t), self.RateC_e_unbound_liver(y, w, c, t), self.RateC_e_bound_liver(y, w, c, t), self.RateFcRn_free_liver(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p(self, y, w, c, t):
		return (1 / c[28]) * (-(c[32] + c[63]) * (y[0]/3126.0) + (c[31] - c[64]) * c[0] + (c[37] - c[65]) * c[1] + (c[45] - c[66]) * (y[9]/31.9) + (c[33] - c[67]) * c[2] + (c[36] - c[68]) * c[3] + (c[42] - c[69]) * c[4] + (c[34] - c[70]) * c[5] + (c[35] - c[71]) * c[6] + ((c[39] - c[72]) + (c[40] - c[73]) + (c[43] - c[74]) + (c[41] - c[75]) + (c[38] - c[76])) * (y[20]/2143.0) + (c[44] - c[77]) * c[7] + c[78] * (y[2]/274.0))

	def RateC_bc(self, y, w, c, t):
		return (1 / c[29]) * (-c[49] * (y[1]/2558.0) + c[48] * c[8] + c[54] * c[9] + c[62] * (y[13]/26.1) + c[50] * c[10] + c[53] * c[11] + c[59] * c[12] + c[51] * c[13] + c[52] * c[14] + (c[56] + c[57] + c[60] + c[58] + c[55]) * (y[21]/183.0) + c[61] * c[15])

	def RateC_ln(self, y, w, c, t):
		return (1 / c[30]) * ((1 - c[79]) * c[63] * (y[7]/45.0) + (1 - c[80]) * c[64] * c[16] + (1 - c[81]) * c[65] * c[17] + (1 - c[94]) * c[46] * (y[19]/90.0) + (1 - c[82]) * c[47] * (y[12]/261.0) + (1 - c[83]) * c[67] * c[18] + (1 - c[84]) * c[68] * c[19] + (1 - c[85]) * c[69] * c[20] + (1 - c[86]) * c[70] * c[21] + (1 - c[87]) * c[71] * c[22] + (1 - c[88]) * c[72] * c[23] + (1 - c[89]) * c[73] * c[24] + (1 - c[90]) * c[74] * c[25] + (1 - c[91]) * c[75] * c[26] + (1 - c[92]) * c[76] * (y[22]/149.0) + (1 - c[93]) * c[77] * c[27] - c[78] * (y[2]/274.0))

	def RateC_p_lung(self, y, w, c, t):
		return (1 / c[172]) * ((c[32] * (y[0]/3126.0) - (c[32] - c[63]) * (y[3]/1000.0) - (1 - c[107]) * c[63] * (y[3]/1000.0) - c[105] * (y[3]/1000.0)) + c[105] * c[106] * (y[6]/5.0))

	def RateC_bc_lung(self, y, w, c, t):
		return (1 / c[173]) * (c[49] * (y[1]/2558.0) - c[49] * (y[4]/55.0))

	def RateC_e_unbound_lung(self, y, w, c, t):
		return (1 / c[175]) * ((c[105] * ((y[3]/1000.0) + (y[7]/45.0)) - c[175] * c[102] * (y[5]/5.0) * (y[8]/5.0)) + c[175] * c[103] * (y[6]/5.0) - c[104] * (y[5]/5.0) * c[175])

	def RateC_e_bound_lung(self, y, w, c, t):
		return (1 / c[175]) * (c[175] * c[102] * (y[5]/5.0) * (y[8]/5.0) - c[175] * c[103] * (y[6]/5.0) - c[105] * (y[6]/5.0))

	def RateC_is_lung(self, y, w, c, t):
		return (1 / c[174]) * (((1 - c[107]) * c[63] * (y[3]/1000.0) - (1 - c[79]) * c[63] * (y[7]/45.0)) + c[105] * (1 - c[106]) * (y[6]/5.0) - c[105] * (y[7]/45.0))

	def RateFcRn_free_lung(self, y, w, c, t):
		return (1 / c[175]) * ((c[103] * (y[6]/5.0) * c[175] - c[102] * (y[5]/5.0) * (y[8]/5.0) * c[175]) + c[105] * (y[6]/5.0))

	def RateC_p_brain(self, y, w, c, t):
		return (1 / c[114]) * ((c[45] * (y[3]/1000.0) - (c[45] - c[66]) * (y[9]/31.9) - (1 - c[123]) * c[119] * (y[9]/31.9) - (1 - c[124]) * c[46] * (y[9]/31.9) - c[120] * c[118] * (y[9]/31.9)) + c[120] * c[121] * c[118] * c[106] * (y[11]/0.1) + c[120] * (1 - c[121]) * c[118] * c[106] * (y[15]/0.1))

	def RateC_BBB_unbound_brain(self, y, w, c, t):
		return (1 / c[115]) * ((c[120] * c[121] * c[118] * ((y[9]/31.9) + (y[12]/261.0)) - c[115] * c[102] * (y[10]/0.1) * c[122]) + c[115] * c[103] * (y[11]/0.1) - c[115] * c[104] * (y[10]/0.1))

	def RateC_BBB_bound_brain(self, y, w, c, t):
		return (1 / c[115]) * (-c[120] * c[121] * c[118] * (y[11]/0.1) + c[115] * c[102] * (y[10]/0.1) * c[122] - c[115] * c[103] * (y[11]/0.1))

	def RateC_is_brain(self, y, w, c, t):
		return (1 / c[116]) * (((1 - c[123]) * c[119] * (y[9]/31.9) - (1 - c[82]) * c[119] * (y[12]/261.0) - c[119] * (y[12]/261.0)) + c[119] * (y[19]/90.0) + c[120] * c[121] * c[118] * (1 - c[106]) * (y[11]/0.1) - c[120] * c[121] * c[118] * (y[12]/261.0))

	def RateC_bc_brain(self, y, w, c, t):
		return (1 / c[117]) * (c[62] * (y[4]/55.0) - c[62] * (y[13]/26.1))

	def RateC_BCSFB_unbound_brain(self, y, w, c, t):
		return (1 / c[125]) * ((c[120] * c[130] * c[118] * (y[9]/31.9) + c[132] * c[120] * (1 - c[121]) * c[118] * (y[16]/22.5) + (1 - c[132]) * c[120] * (1 - c[121]) * c[118] * (y[17]/22.5) - c[125] * c[102] * (y[14]/0.1) * c[131]) + c[125] * c[103] * (y[15]/0.1) - c[125] * c[104] * (y[14]/0.1))

	def RateC_BCSFB_bound_brain(self, y, w, c, t):
		return (1 / c[125]) * (-c[120] * (1 - c[121]) * c[118] * (y[15]/0.1) + c[125] * c[102] * (y[14]/0.1) * c[131] - c[125] * c[103] * (y[15]/0.1))

	def RateC_LV_brain(self, y, w, c, t):
		return (1 / c[126]) * (((1 - c[124]) * c[132] * c[46] * (y[9]/31.9) + c[132] * c[119] * (y[12]/261.0) - (c[132] * c[46] + c[132] * c[119]) * (y[16]/22.5) - c[132] * c[120] * (1 - c[121]) * c[118] * (y[16]/22.5)) + c[132] * c[120] * (1 - c[121]) * c[118] * (1 - c[106]) * (y[15]/0.1))

	def RateC_TFV_brain(self, y, w, c, t):
		return (1 / c[127]) * (((1 - c[124]) * (1 - c[132]) * c[46] * (y[9]/31.9) + (1 - c[132]) * c[119] * (y[12]/261.0) - (c[46] + c[119]) * (y[17]/22.5) - (1 - c[132]) * c[120] * (1 - c[121]) * c[118] * (y[17]/22.5)) + (1 - c[132]) * c[120] * (1 - c[121]) * c[118] * (1 - c[106]) * (y[15]/0.1) + (c[132] * c[46] + c[132] * c[119]) * (y[16]/22.5))

	def RateC_CM_brain(self, y, w, c, t):
		return (1 / c[128]) * (c[46] + c[119]) * ((y[17]/22.5) - (y[18]/7.5))

	def RateC_SAS_brain(self, y, w, c, t):
		return (1 / c[129]) * ((c[46] + c[119]) * (y[18]/7.5) - (1 - c[94]) * c[46] * (y[19]/90.0) - c[119] * (y[19]/90.0))

	def RateC_p_liver(self, y, w, c, t):
		return (1 / c[187]) * ((c[38] * (y[3]/1000.0) + (c[43] - c[74]) * c[161] + (c[41] - c[75]) * c[163] + (c[39] - c[72]) * c[165] + (c[40] - c[73]) * c[167] - ((c[38] - c[76]) + (c[43] - c[74]) + (c[41] - c[75]) + (c[39] - c[72]) + (c[40] - c[73])) * (y[20]/2143.0) - (1 - c[146]) * c[76] * (y[20]/2143.0) - c[145] * (y[20]/2143.0)) + c[145] * c[106] * (y[24]/10.7))

	def RateC_bc_liver(self, y, w, c, t):
		return (1 / c[188]) * (c[55] * (y[4]/55.0) + c[60] * c[162] + c[58] * c[164] + c[56] * c[166] + c[57] * c[168] - (c[55] + c[60] + c[58] + c[56] + c[57]) * (y[21]/183.0))

	def RateC_is_liver(self, y, w, c, t):
		return (1 / c[189]) * (((1 - c[146]) * c[76] * (y[20]/2143.0) - (1 - c[92]) * c[76] * (y[22]/149.0)) + c[145] * (1 - c[106]) * (y[24]/10.7) - c[145] * (y[22]/149.0))

	def RateC_e_unbound_liver(self, y, w, c, t):
		return (1 / c[190]) * ((c[145] * ((y[20]/2143.0) + (y[22]/149.0)) - c[190] * c[102] * (y[23]/10.7) * (y[25]/10.7)) + c[190] * c[103] * (y[24]/10.7) - c[104] * (y[23]/10.7) * c[190])

	def RateC_e_bound_liver(self, y, w, c, t):
		return (1 / c[190]) * (c[190] * c[102] * (y[23]/10.7) * (y[25]/10.7) - c[190] * c[103] * (y[24]/10.7) - c[145] * (y[24]/10.7))

	def RateFcRn_free_liver(self, y, w, c, t):
		return (1 / c[190]) * ((c[103] * (y[24]/10.7) * c[190] - c[102] * (y[23]/10.7) * (y[25]/10.7) * c[190]) + c[145] * (y[24]/10.7))

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

	def __init__(self, y_indexes={'C_p': 0, 'C_bc': 1, 'C_ln': 2, 'C_p_lung': 3, 'C_bc_lung': 4, 'C_e_unbound_lung': 5, 'C_e_bound_lung': 6, 'C_is_lung': 7, 'FcRn_free_lung': 8, 'C_p_brain': 9, 'C_BBB_unbound_brain': 10, 'C_BBB_bound_brain': 11, 'C_is_brain': 12, 'C_bc_brain': 13, 'C_BCSFB_unbound_brain': 14, 'C_BCSFB_bound_brain': 15, 'C_LV_brain': 16, 'C_TFV_brain': 17, 'C_CM_brain': 18, 'C_SAS_brain': 19, 'C_p_liver': 20, 'C_bc_liver': 21, 'C_is_liver': 22, 'C_e_unbound_liver': 23, 'C_e_bound_liver': 24, 'FcRn_free_liver': 25}, w_indexes={}, c_indexes={'C_p_heart': 0, 'C_p_kidney': 1, 'C_p_muscle': 2, 'C_p_marrow': 3, 'C_p_thymus': 4, 'C_p_skin': 5, 'C_p_fat': 6, 'C_p_other': 7, 'C_bc_heart': 8, 'C_bc_kidney': 9, 'C_bc_muscle': 10, 'C_bc_marrow': 11, 'C_bc_thymus': 12, 'C_bc_skin': 13, 'C_bc_fat': 14, 'C_bc_other': 15, 'C_is_heart': 16, 'C_is_kidney': 17, 'C_is_muscle': 18, 'C_is_marrow': 19, 'C_is_thymus': 20, 'C_is_skin': 21, 'C_is_fat': 22, 'C_is_SI': 23, 'C_is_LI': 24, 'C_is_spleen': 25, 'C_is_pancreas': 26, 'C_is_other': 27, 'Vp': 28, 'Vbc': 29, 'Vlymphnode': 30, 'Q_p_heart': 31, 'Q_p_lung': 32, 'Q_p_muscle': 33, 'Q_p_skin': 34, 'Q_p_fat': 35, 'Q_p_marrow': 36, 'Q_p_kidney': 37, 'Q_p_liver': 38, 'Q_p_SI': 39, 'Q_p_LI': 40, 'Q_p_pancreas': 41, 'Q_p_thymus': 42, 'Q_p_spleen': 43, 'Q_p_other': 44, 'Q_p_brain': 45, 'Q_CSF_brain': 46, 'Q_ECF_brain': 47, 'Q_bc_heart': 48, 'Q_bc_lung': 49, 'Q_bc_muscle': 50, 'Q_bc_skin': 51, 'Q_bc_fat': 52, 'Q_bc_marrow': 53, 'Q_bc_kidney': 54, 'Q_bc_liver': 55, 'Q_bc_SI': 56, 'Q_bc_LI': 57, 'Q_bc_pancreas': 58, 'Q_bc_thymus': 59, 'Q_bc_spleen': 60, 'Q_bc_other': 61, 'Q_bc_brain': 62, 'L_lung': 63, 'L_heart': 64, 'L_kidney': 65, 'L_brain': 66, 'L_muscle': 67, 'L_marrow': 68, 'L_thymus': 69, 'L_skin': 70, 'L_fat': 71, 'L_SI': 72, 'L_LI': 73, 'L_spleen': 74, 'L_pancreas': 75, 'L_liver': 76, 'L_other': 77, 'L_LN': 78, 'sigma_L_lung': 79, 'sigma_L_heart': 80, 'sigma_L_kidney': 81, 'sigma_L_brain_ISF': 82, 'sigma_L_muscle': 83, 'sigma_L_marrow': 84, 'sigma_L_thymus': 85, 'sigma_L_skin': 86, 'sigma_L_fat': 87, 'sigma_L_SI': 88, 'sigma_L_LI': 89, 'sigma_L_spleen': 90, 'sigma_L_pancreas': 91, 'sigma_L_liver': 92, 'sigma_L_other': 93, 'sigma_L_SAS': 94, 'C_p_0': 95, 'C_bc_0': 96, 'C_ln_0': 97, 'Vp_lung': 172, 'VBC_lung': 173, 'VIS_lung': 174, 'VES_lung': 175, 'kon_FcRn': 102, 'koff_FcRn': 103, 'kdeg': 104, 'CLup_lung': 105, 'FR': 106, 'sigma_V_lung': 107, 'C_p_lung_0': 108, 'C_bc_lung_0': 109, 'C_is_lung_0': 110, 'C_e_unbound_lung_0': 111, 'C_e_bound_lung_0': 112, 'FcRn_free_lung_0': 113, 'Vp_brain': 114, 'VBBB_brain': 115, 'VIS_brain': 116, 'VBC_brain': 117, 'V_ES_brain': 118, 'Q_ISF_brain': 119, 'CLup_brain': 120, 'f_BBB': 121, 'FcRn_free_BBB': 122, 'sigma_V_BBB': 123, 'sigma_V_BCSFB': 124, 'V_BCSFB_brain': 125, 'V_LV_brain': 126, 'V_TFV_brain': 127, 'V_CM_brain': 128, 'V_SAS_brain': 129, 'f_BCSFB': 130, 'FcRn_free_BCSFB': 131, 'f_LV': 132, 'C_BCSFB_unbound_brain_0': 133, 'C_BCSFB_bound_brain_0': 134, 'C_LV_brain_0': 135, 'C_TFV_brain_0': 136, 'C_CM_brain_0': 137, 'C_SAS_brain_0': 138, 'C_p_brain_0': 139, 'C_is_brain_0': 140, 'Vp_liver': 187, 'VBC_liver': 188, 'VIS_liver': 189, 'VES_liver': 190, 'CLup_liver': 145, 'sigma_V_liver': 146, 'C_p_liver_0': 147, 'C_bc_liver_0': 148, 'C_is_liver_0': 149, 'C_e_unbound_liver_0': 150, 'C_e_bound_liver_0': 151, 'FcRn_free_liver_0': 152, 'C_p_spleen_0': 153, 'C_bc_spleen_0': 154, 'C_p_pancreas_0': 155, 'C_bc_pancreas_0': 156, 'C_p_SI_0': 157, 'C_bc_SI_0': 158, 'C_p_LI_0': 159, 'C_bc_LI_0': 160, 'C_p_spleen': 161, 'C_bc_spleen': 162, 'C_p_pancreas': 163, 'C_bc_pancreas': 164, 'C_p_SI': 165, 'C_bc_SI': 166, 'C_p_LI': 167, 'C_bc_LI': 168, 'plasma': 169, 'blood_cells': 170, 'lymph_node': 171, 'brain_plasma': 176, 'BBB_unbound': 177, 'BBB_bound': 178, 'brain_ISF': 179, 'brain_blood_cells': 180, 'BCSFB_unbound': 181, 'BCSFB_bound': 182, 'LV': 183, 'TFV': 184, 'CM': 185, 'SAS': 186}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([14117.999999999989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000533074]), w0=jnp.array([]), c=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 8343.0, 2591.0, 32402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 36402.0, 21.0, 10.5, 6350.0, 148920.0, 27410.0, 9520.0, 6830.0, 2120.0, 26530.0, 10820.0, 10130.0, 10530.0, 2500.0, 289.0, 5190.0, 4520.0, 29810.0, 364.0, 16.0, 65.0, 73.0, 67.0, 5.0, 1.0, 23.0, 17.0, 25.0, 26.0, 13.0, 6.0, 26.0, 11.0, 500.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.51631477927063, 0.0, 0.0, 1000.0, 55.0, 45.0, 5.0, 559000000.0, 23.9, 26.6, 0.55, 0.715, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 31.9, 0.1, 261.0, 26.1, 7.25, 10.5, 0.3, 0.95, 1.0, 0.95, 0.9974, 0.1, 22.5, 22.5, 7.5, 90.0, 0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2143.0, 183.0, 149.0, 10.7, 0.55, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0, 1000.0, 55.0, 45.0, 5.0, 31.9, 0.1, 0.1, 261.0, 26.1, 0.1, 0.1, 22.5, 22.5, 7.5, 90.0, 2143.0, 183.0, 149.0, 10.7]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

