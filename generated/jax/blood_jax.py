import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0])
y_indexes = {'C_p': 0, 'C_bc': 1, 'C_ln': 2}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 8343.0, 2591.0, 32402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 36402.0, 21.0, 10.5, 6350.0, 148920.0, 27410.0, 9520.0, 6830.0, 2120.0, 26530.0, 10820.0, 10130.0, 10530.0, 2500.0, 289.0, 5190.0, 4520.0, 29810.0, 364.0, 16.0, 65.0, 73.0, 67.0, 5.0, 1.0, 23.0, 17.0, 25.0, 26.0, 13.0, 6.0, 26.0, 11.0, 500.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0]) 
c_indexes = {'C_is_lung': 0, 'C_p_heart': 1, 'C_p_kidney': 2, 'C_p_brain': 3, 'C_p_muscle': 4, 'C_p_marrow': 5, 'C_p_thymus': 6, 'C_p_skin': 7, 'C_p_fat': 8, 'C_p_other': 9, 'C_bc_heart': 10, 'C_bc_kidney': 11, 'C_bc_brain': 12, 'C_bc_muscle': 13, 'C_bc_marrow': 14, 'C_bc_thymus': 15, 'C_bc_skin': 16, 'C_bc_fat': 17, 'C_bc_other': 18, 'C_is_heart': 19, 'C_is_kidney': 20, 'C_is_brain': 21, 'C_is_muscle': 22, 'C_is_marrow': 23, 'C_is_thymus': 24, 'C_is_skin': 25, 'C_is_fat': 26, 'C_is_SI': 27, 'C_is_LI': 28, 'C_is_spleen': 29, 'C_is_pancreas': 30, 'C_is_other': 31, 'C_SAS_brain': 32, 'C_is_liver': 33, 'C_p_liver': 34, 'C_bc_liver': 35, 'Vp': 36, 'Vbc': 37, 'Vlymphnode': 38, 'Q_p_heart': 39, 'Q_p_lung': 40, 'Q_p_muscle': 41, 'Q_p_skin': 42, 'Q_p_fat': 43, 'Q_p_marrow': 44, 'Q_p_kidney': 45, 'Q_p_liver': 46, 'Q_p_SI': 47, 'Q_p_LI': 48, 'Q_p_pancreas': 49, 'Q_p_thymus': 50, 'Q_p_spleen': 51, 'Q_p_other': 52, 'Q_p_brain': 53, 'Q_CSF_brain': 54, 'Q_ECF_brain': 55, 'Q_bc_heart': 56, 'Q_bc_lung': 57, 'Q_bc_muscle': 58, 'Q_bc_skin': 59, 'Q_bc_fat': 60, 'Q_bc_marrow': 61, 'Q_bc_kidney': 62, 'Q_bc_liver': 63, 'Q_bc_SI': 64, 'Q_bc_LI': 65, 'Q_bc_pancreas': 66, 'Q_bc_thymus': 67, 'Q_bc_spleen': 68, 'Q_bc_other': 69, 'Q_bc_brain': 70, 'L_lung': 71, 'L_heart': 72, 'L_kidney': 73, 'L_brain': 74, 'L_muscle': 75, 'L_marrow': 76, 'L_thymus': 77, 'L_skin': 78, 'L_fat': 79, 'L_SI': 80, 'L_LI': 81, 'L_spleen': 82, 'L_pancreas': 83, 'L_liver': 84, 'L_other': 85, 'L_LN': 86, 'sigma_L_lung': 87, 'sigma_L_heart': 88, 'sigma_L_kidney': 89, 'sigma_L_brain_ISF': 90, 'sigma_L_muscle': 91, 'sigma_L_marrow': 92, 'sigma_L_thymus': 93, 'sigma_L_skin': 94, 'sigma_L_fat': 95, 'sigma_L_SI': 96, 'sigma_L_LI': 97, 'sigma_L_spleen': 98, 'sigma_L_pancreas': 99, 'sigma_L_liver': 100, 'sigma_L_other': 101, 'sigma_L_SAS': 102, 'C_p_0': 103, 'C_bc_0': 104, 'C_ln_0': 105, 'plasma': 106, 'blood_cells': 107, 'lymph_node': 108}

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
		return (1 / c[36]) * (-(c[40] + c[71]) * (y[0]/3126.0) + (c[39] - c[72]) * c[1] + (c[45] - c[73]) * c[2] + (c[53] - c[74]) * c[3] + (c[41] - c[75]) * c[4] + (c[44] - c[76]) * c[5] + (c[50] - c[77]) * c[6] + (c[42] - c[78]) * c[7] + (c[43] - c[79]) * c[8] + ((c[47] - c[80]) + (c[48] - c[81]) + (c[51] - c[82]) + (c[49] - c[83]) + (c[46] - c[84])) * c[34] + (c[52] - c[85]) * c[9] + c[86] * (y[2]/274.0))

	def RateC_bc(self, y, w, c, t):
		return (1 / c[37]) * (-c[57] * (y[1]/2558.0) + c[56] * c[10] + c[62] * c[11] + c[70] * c[12] + c[58] * c[13] + c[61] * c[14] + c[67] * c[15] + c[59] * c[16] + c[60] * c[17] + (c[64] + c[65] + c[68] + c[66] + c[63]) * c[35] + c[69] * c[18])

	def RateC_ln(self, y, w, c, t):
		return (1 / c[38]) * ((1 - c[87]) * c[71] * c[0] + (1 - c[88]) * c[72] * c[19] + (1 - c[89]) * c[73] * c[20] + (1 - c[102]) * c[54] * c[32] + (1 - c[90]) * c[55] * c[21] + (1 - c[91]) * c[75] * c[22] + (1 - c[92]) * c[76] * c[23] + (1 - c[93]) * c[77] * c[24] + (1 - c[94]) * c[78] * c[25] + (1 - c[95]) * c[79] * c[26] + (1 - c[96]) * c[80] * c[27] + (1 - c[97]) * c[81] * c[28] + (1 - c[98]) * c[82] * c[29] + (1 - c[99]) * c[83] * c[30] + (1 - c[100]) * c[84] * c[33] + (1 - c[101]) * c[85] * c[31] - c[86] * (y[2]/274.0))

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

	def __init__(self, y_indexes={'C_p': 0, 'C_bc': 1, 'C_ln': 2}, w_indexes={}, c_indexes={'C_is_lung': 0, 'C_p_heart': 1, 'C_p_kidney': 2, 'C_p_brain': 3, 'C_p_muscle': 4, 'C_p_marrow': 5, 'C_p_thymus': 6, 'C_p_skin': 7, 'C_p_fat': 8, 'C_p_other': 9, 'C_bc_heart': 10, 'C_bc_kidney': 11, 'C_bc_brain': 12, 'C_bc_muscle': 13, 'C_bc_marrow': 14, 'C_bc_thymus': 15, 'C_bc_skin': 16, 'C_bc_fat': 17, 'C_bc_other': 18, 'C_is_heart': 19, 'C_is_kidney': 20, 'C_is_brain': 21, 'C_is_muscle': 22, 'C_is_marrow': 23, 'C_is_thymus': 24, 'C_is_skin': 25, 'C_is_fat': 26, 'C_is_SI': 27, 'C_is_LI': 28, 'C_is_spleen': 29, 'C_is_pancreas': 30, 'C_is_other': 31, 'C_SAS_brain': 32, 'C_is_liver': 33, 'C_p_liver': 34, 'C_bc_liver': 35, 'Vp': 36, 'Vbc': 37, 'Vlymphnode': 38, 'Q_p_heart': 39, 'Q_p_lung': 40, 'Q_p_muscle': 41, 'Q_p_skin': 42, 'Q_p_fat': 43, 'Q_p_marrow': 44, 'Q_p_kidney': 45, 'Q_p_liver': 46, 'Q_p_SI': 47, 'Q_p_LI': 48, 'Q_p_pancreas': 49, 'Q_p_thymus': 50, 'Q_p_spleen': 51, 'Q_p_other': 52, 'Q_p_brain': 53, 'Q_CSF_brain': 54, 'Q_ECF_brain': 55, 'Q_bc_heart': 56, 'Q_bc_lung': 57, 'Q_bc_muscle': 58, 'Q_bc_skin': 59, 'Q_bc_fat': 60, 'Q_bc_marrow': 61, 'Q_bc_kidney': 62, 'Q_bc_liver': 63, 'Q_bc_SI': 64, 'Q_bc_LI': 65, 'Q_bc_pancreas': 66, 'Q_bc_thymus': 67, 'Q_bc_spleen': 68, 'Q_bc_other': 69, 'Q_bc_brain': 70, 'L_lung': 71, 'L_heart': 72, 'L_kidney': 73, 'L_brain': 74, 'L_muscle': 75, 'L_marrow': 76, 'L_thymus': 77, 'L_skin': 78, 'L_fat': 79, 'L_SI': 80, 'L_LI': 81, 'L_spleen': 82, 'L_pancreas': 83, 'L_liver': 84, 'L_other': 85, 'L_LN': 86, 'sigma_L_lung': 87, 'sigma_L_heart': 88, 'sigma_L_kidney': 89, 'sigma_L_brain_ISF': 90, 'sigma_L_muscle': 91, 'sigma_L_marrow': 92, 'sigma_L_thymus': 93, 'sigma_L_skin': 94, 'sigma_L_fat': 95, 'sigma_L_SI': 96, 'sigma_L_LI': 97, 'sigma_L_spleen': 98, 'sigma_L_pancreas': 99, 'sigma_L_liver': 100, 'sigma_L_other': 101, 'sigma_L_SAS': 102, 'C_p_0': 103, 'C_bc_0': 104, 'C_ln_0': 105, 'plasma': 106, 'blood_cells': 107, 'lymph_node': 108}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0, 7752.0, 181913.0, 33469.0, 11626.0, 8343.0, 2591.0, 32402.0, 13210.0, 12368.0, 12867.0, 3056.0, 353.0, 6343.0, 5521.0, 36402.0, 21.0, 10.5, 6350.0, 148920.0, 27410.0, 9520.0, 6830.0, 2120.0, 26530.0, 10820.0, 10130.0, 10530.0, 2500.0, 289.0, 5190.0, 4520.0, 29810.0, 364.0, 16.0, 65.0, 73.0, 67.0, 5.0, 1.0, 23.0, 17.0, 25.0, 26.0, 13.0, 6.0, 26.0, 11.0, 500.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 3126.0, 2558.0, 274.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

