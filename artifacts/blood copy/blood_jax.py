import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([31260.0, 25580.0, 2740.0])
y_indexes = {'C_p': 0, 'C_bc': 1, 'C_ln': 2}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 11233.0, 2591.0, 36402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 21453.0, 24.0, 10.5, 6342.0, 148838.0, 27383.0, 9512.0, 9191.0, 2120.0, 29784.0, 10808.0, 10120.0, 10527.0, 2500.0, 289.0, 5189.0, 4517.0, 17553.0, 183.0, 78.0, 364.0, 215.0, 335.0, 26.0, 4.0, 116.0, 112.0, 124.0, 129.0, 63.0, 31.0, 132.0, 55.0, 2000.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 10.0, 10.0, 10.0, 3126.0, 2558.0, 274.0]) 
c_indexes = {'C_p_lung': 0, 'C_p_heart': 1, 'C_p_kidney': 2, 'C_p_brain': 3, 'C_p_muscle': 4, 'C_p_marrow': 5, 'C_p_thymus': 6, 'C_p_skin': 7, 'C_p_fat': 8, 'C_p_liver': 9, 'C_p_other': 10, 'C_bc_lung': 11, 'C_bc_heart': 12, 'C_bc_kidney': 13, 'C_bc_brain': 14, 'C_bc_muscle': 15, 'C_bc_marrow': 16, 'C_bc_thymus': 17, 'C_bc_skin': 18, 'C_bc_fat': 19, 'C_bc_liver': 20, 'C_bc_other': 21, 'C_is_lung': 22, 'C_is_heart': 23, 'C_is_kidney': 24, 'C_is_brain': 25, 'C_is_muscle': 26, 'C_is_marrow': 27, 'C_is_thymus': 28, 'C_is_skin': 29, 'C_is_fat': 30, 'C_is_SI': 31, 'C_is_LI': 32, 'C_is_spleen': 33, 'C_is_pancreas': 34, 'C_is_liver': 35, 'C_is_other': 36, 'C_SAS_brain': 37, 'Vp': 38, 'Vbc': 39, 'Vlymphnode': 40, 'Q_p_heart': 41, 'Q_p_lung': 42, 'Q_p_muscle': 43, 'Q_p_skin': 44, 'Q_p_fat': 45, 'Q_p_marrow': 46, 'Q_p_kidney': 47, 'Q_p_liver': 48, 'Q_p_SI': 49, 'Q_p_LI': 50, 'Q_p_pancreas': 51, 'Q_p_thymus': 52, 'Q_p_spleen': 53, 'Q_p_other': 54, 'Q_p_brain': 55, 'Q_CSF_brain': 56, 'Q_ECF_brain': 57, 'Q_bc_heart': 58, 'Q_bc_lung': 59, 'Q_bc_muscle': 60, 'Q_bc_skin': 61, 'Q_bc_fat': 62, 'Q_bc_marrow': 63, 'Q_bc_kidney': 64, 'Q_bc_liver': 65, 'Q_bc_SI': 66, 'Q_bc_LI': 67, 'Q_bc_pancreas': 68, 'Q_bc_thymus': 69, 'Q_bc_spleen': 70, 'Q_bc_other': 71, 'Q_bc_brain': 72, 'L_lung': 73, 'L_heart': 74, 'L_kidney': 75, 'L_brain': 76, 'L_muscle': 77, 'L_marrow': 78, 'L_thymus': 79, 'L_skin': 80, 'L_fat': 81, 'L_SI': 82, 'L_LI': 83, 'L_spleen': 84, 'L_pancreas': 85, 'L_liver': 86, 'L_other': 87, 'L_LN': 88, 'sigma_L_lung': 89, 'sigma_L_heart': 90, 'sigma_L_kidney': 91, 'sigma_L_brain_ISF': 92, 'sigma_L_muscle': 93, 'sigma_L_marrow': 94, 'sigma_L_thymus': 95, 'sigma_L_skin': 96, 'sigma_L_fat': 97, 'sigma_L_SI': 98, 'sigma_L_LI': 99, 'sigma_L_spleen': 100, 'sigma_L_pancreas': 101, 'sigma_L_liver': 102, 'sigma_L_other': 103, 'sigma_L_SAS': 104, 'C_p_0': 105, 'C_bc_0': 106, 'C_ln_0': 107, 'plasma': 108, 'blood_cells': 109, 'lymph_node': 110}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p(y, w, c, t), self.RateC_bc(y, w, c, t), self.RateC_ln(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p(self, y, w, c, t):
		return (1 / c[38]) * (-(c[42] + c[73]) * (y[0]/3126.0) + (c[41] - c[74]) * c[1] + (c[47] - c[75]) * c[2] + (c[55] - c[76]) * c[3] + (c[43] - c[77]) * c[4] + (c[46] - c[78]) * c[5] + (c[52] - c[79]) * c[6] + (c[44] - c[80]) * c[7] + (c[45] - c[81]) * c[8] + ((c[49] - c[82]) + (c[50] - c[83]) + (c[53] - c[84]) + (c[51] - c[85]) + (c[48] - c[86])) * c[9] + (c[54] - c[87]) * c[10] + c[88] * (y[2]/274.0))

	def RateC_bc(self, y, w, c, t):
		return (1 / c[39]) * (-c[59] * (y[1]/2558.0) + c[58] * c[12] + c[64] * c[13] + c[72] * c[14] + c[60] * c[15] + c[63] * c[16] + c[69] * c[17] + c[61] * c[18] + c[62] * c[19] + (c[66] + c[67] + c[70] + c[68] + c[65]) * c[20] + c[71] * c[21])

	def RateC_ln(self, y, w, c, t):
		return (1 / c[40]) * ((1 - c[89]) * c[73] * c[22] + (1 - c[90]) * c[74] * c[23] + (1 - c[91]) * c[75] * c[24] + (1 - c[104]) * c[56] * c[37] + (1 - c[92]) * c[57] * c[25] + (1 - c[93]) * c[77] * c[26] + (1 - c[94]) * c[78] * c[27] + (1 - c[95]) * c[79] * c[28] + (1 - c[96]) * c[80] * c[29] + (1 - c[97]) * c[81] * c[30] + (1 - c[98]) * c[82] * c[31] + (1 - c[99]) * c[83] * c[32] + (1 - c[100]) * c[84] * c[33] + (1 - c[101]) * c[85] * c[34] + (1 - c[102]) * c[86] * c[35] + (1 - c[103]) * c[87] * c[36] - c[88] * (y[2]/274.0))

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

	def __init__(self, y_indexes={'C_p': 0, 'C_bc': 1, 'C_ln': 2}, w_indexes={}, c_indexes={'C_p_lung': 0, 'C_p_heart': 1, 'C_p_kidney': 2, 'C_p_brain': 3, 'C_p_muscle': 4, 'C_p_marrow': 5, 'C_p_thymus': 6, 'C_p_skin': 7, 'C_p_fat': 8, 'C_p_liver': 9, 'C_p_other': 10, 'C_bc_lung': 11, 'C_bc_heart': 12, 'C_bc_kidney': 13, 'C_bc_brain': 14, 'C_bc_muscle': 15, 'C_bc_marrow': 16, 'C_bc_thymus': 17, 'C_bc_skin': 18, 'C_bc_fat': 19, 'C_bc_liver': 20, 'C_bc_other': 21, 'C_is_lung': 22, 'C_is_heart': 23, 'C_is_kidney': 24, 'C_is_brain': 25, 'C_is_muscle': 26, 'C_is_marrow': 27, 'C_is_thymus': 28, 'C_is_skin': 29, 'C_is_fat': 30, 'C_is_SI': 31, 'C_is_LI': 32, 'C_is_spleen': 33, 'C_is_pancreas': 34, 'C_is_liver': 35, 'C_is_other': 36, 'C_SAS_brain': 37, 'Vp': 38, 'Vbc': 39, 'Vlymphnode': 40, 'Q_p_heart': 41, 'Q_p_lung': 42, 'Q_p_muscle': 43, 'Q_p_skin': 44, 'Q_p_fat': 45, 'Q_p_marrow': 46, 'Q_p_kidney': 47, 'Q_p_liver': 48, 'Q_p_SI': 49, 'Q_p_LI': 50, 'Q_p_pancreas': 51, 'Q_p_thymus': 52, 'Q_p_spleen': 53, 'Q_p_other': 54, 'Q_p_brain': 55, 'Q_CSF_brain': 56, 'Q_ECF_brain': 57, 'Q_bc_heart': 58, 'Q_bc_lung': 59, 'Q_bc_muscle': 60, 'Q_bc_skin': 61, 'Q_bc_fat': 62, 'Q_bc_marrow': 63, 'Q_bc_kidney': 64, 'Q_bc_liver': 65, 'Q_bc_SI': 66, 'Q_bc_LI': 67, 'Q_bc_pancreas': 68, 'Q_bc_thymus': 69, 'Q_bc_spleen': 70, 'Q_bc_other': 71, 'Q_bc_brain': 72, 'L_lung': 73, 'L_heart': 74, 'L_kidney': 75, 'L_brain': 76, 'L_muscle': 77, 'L_marrow': 78, 'L_thymus': 79, 'L_skin': 80, 'L_fat': 81, 'L_SI': 82, 'L_LI': 83, 'L_spleen': 84, 'L_pancreas': 85, 'L_liver': 86, 'L_other': 87, 'L_LN': 88, 'sigma_L_lung': 89, 'sigma_L_heart': 90, 'sigma_L_kidney': 91, 'sigma_L_brain_ISF': 92, 'sigma_L_muscle': 93, 'sigma_L_marrow': 94, 'sigma_L_thymus': 95, 'sigma_L_skin': 96, 'sigma_L_fat': 97, 'sigma_L_SI': 98, 'sigma_L_LI': 99, 'sigma_L_spleen': 100, 'sigma_L_pancreas': 101, 'sigma_L_liver': 102, 'sigma_L_other': 103, 'sigma_L_SAS': 104, 'C_p_0': 105, 'C_bc_0': 106, 'C_ln_0': 107, 'plasma': 108, 'blood_cells': 109, 'lymph_node': 110}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([31260.0, 25580.0, 2740.0]), w0=jnp.array([]), c=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 11233.0, 2591.0, 36402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 21453.0, 24.0, 10.5, 6342.0, 148838.0, 27383.0, 9512.0, 9191.0, 2120.0, 29784.0, 10808.0, 10120.0, 10527.0, 2500.0, 289.0, 5189.0, 4517.0, 17553.0, 183.0, 78.0, 364.0, 215.0, 335.0, 26.0, 4.0, 116.0, 112.0, 124.0, 129.0, 63.0, 31.0, 132.0, 55.0, 2000.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 10.0, 10.0, 10.0, 3126.0, 2558.0, 274.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

