import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'C_p_brain': 0, 'C_BBB_unbound_brain': 1, 'C_BBB_bound_brain': 2, 'C_is_brain': 3, 'C_bc_brain': 4, 'C_BCSFB_unbound_brain': 5, 'C_BCSFB_bound_brain': 6, 'C_LV_brain': 7, 'C_TFV_brain': 8, 'C_CM_brain': 9, 'C_SAS_brain': 10}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([31.9, 0.1, 261.0, 26.1, 7.25, 36402.0, 29810.0, 10.5, 21.0, 559000000.0, 23.9, 26.6, 0.3, 0.95, 0.715, 1.0, 0.95, 0.9974, 0.2, 73.0, 0.1, 22.5, 22.5, 7.5, 90.0, 0.2, 0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.51631477927063, 0.0, 4.51631477927063, 0.0, 31.9, 0.1, 0.1, 261.0, 26.1, 0.1, 0.1, 22.5, 22.5, 7.5, 90.0]) 
c_indexes = {'Vp_brain': 0, 'VBBB_brain': 1, 'VIS_brain': 2, 'VBC_brain': 3, 'V_ES_brain': 4, 'Q_p_brain': 5, 'Q_bc_brain': 6, 'Q_ISF_brain': 7, 'Q_CSF_brain': 8, 'kon_FcRn': 9, 'koff_FcRn': 10, 'kdeg': 11, 'CLup_brain': 12, 'f_BBB': 13, 'FR': 14, 'FcRn_free_BBB': 15, 'sigma_V_BBB': 16, 'sigma_V_BCSFB': 17, 'sigma_L_brain_ISF': 18, 'L_brain': 19, 'V_BCSFB_brain': 20, 'V_LV_brain': 21, 'V_TFV_brain': 22, 'V_CM_brain': 23, 'V_SAS_brain': 24, 'f_BCSFB': 25, 'sigma_L_SAS': 26, 'FcRn_free_BCSFB': 27, 'f_LV': 28, 'C_LV_brain_0': 29, 'C_TFV_brain_0': 30, 'C_CM_brain_0': 31, 'C_SAS_brain_0': 32, 'C_p_brain_0': 33, 'C_is_brain_0': 34, 'C_p_lung_0': 35, 'C_bc_lung_0': 36, 'C_p_lung': 37, 'C_bc_lung': 38, 'brain_plasma': 39, 'BBB_unbound': 40, 'BBB_bound': 41, 'brain_ISF': 42, 'brain_blood_cells': 43, 'BCSFB_unbound': 44, 'BCSFB_bound': 45, 'LV': 46, 'TFV': 47, 'CM': 48, 'SAS': 49}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p_brain(y, w, c, t), self.RateC_BBB_unbound_brain(y, w, c, t), self.RateC_BBB_bound_brain(y, w, c, t), self.RateC_is_brain(y, w, c, t), self.RateC_bc_brain(y, w, c, t), self.RateC_BCSFB_unbound_brain(y, w, c, t), self.RateC_BCSFB_bound_brain(y, w, c, t), self.RateC_LV_brain(y, w, c, t), self.RateC_TFV_brain(y, w, c, t), self.RateC_CM_brain(y, w, c, t), self.RateC_SAS_brain(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p_brain(self, y, w, c, t):
		return (1 / c[0]) * ((c[5] * c[37] - (c[5] - c[19]) * (y[0]/31.9) - (1 - c[16]) * c[7] * (y[0]/31.9) - (1 - c[17]) * c[8] * (y[0]/31.9) - c[12] * c[4] * (y[0]/31.9)) + c[12] * c[13] * c[4] * c[14] * (y[2]/0.1) + c[12] * (1 - c[13]) * c[4] * c[14] * (y[6]/0.1))

	def RateC_BBB_unbound_brain(self, y, w, c, t):
		return (1 / c[1]) * ((c[12] * c[13] * c[4] * ((y[0]/31.9) + (y[3]/261.0)) - c[1] * c[9] * (y[1]/0.1) * c[15]) + c[1] * c[10] * (y[2]/0.1) - c[1] * c[11] * (y[1]/0.1))

	def RateC_BBB_bound_brain(self, y, w, c, t):
		return (1 / c[1]) * (-c[12] * c[13] * c[4] * (y[2]/0.1) + c[1] * c[9] * (y[1]/0.1) * c[15] - c[1] * c[10] * (y[2]/0.1))

	def RateC_is_brain(self, y, w, c, t):
		return (1 / c[2]) * (((1 - c[16]) * c[7] * (y[0]/31.9) - (1 - c[18]) * c[7] * (y[3]/261.0) - c[7] * (y[3]/261.0)) + c[7] * (y[10]/90.0) + c[12] * c[13] * c[4] * (1 - c[14]) * (y[2]/0.1) - c[12] * c[13] * c[4] * (y[3]/261.0))

	def RateC_bc_brain(self, y, w, c, t):
		return (1 / c[3]) * (c[6] * c[38] - c[6] * (y[4]/26.1))

	def RateC_BCSFB_unbound_brain(self, y, w, c, t):
		return (1 / c[20]) * ((c[12] * c[25] * c[4] * (y[0]/31.9) + c[28] * c[12] * (1 - c[13]) * c[4] * (y[7]/22.5) + (1 - c[28]) * c[12] * (1 - c[13]) * c[4] * (y[8]/22.5) - c[20] * c[9] * (y[5]/0.1) * c[27]) + c[20] * c[10] * (y[6]/0.1) - c[20] * c[11] * (y[5]/0.1))

	def RateC_BCSFB_bound_brain(self, y, w, c, t):
		return (1 / c[20]) * (-c[12] * (1 - c[13]) * c[4] * (y[6]/0.1) + c[20] * c[9] * (y[5]/0.1) * c[27] - c[20] * c[10] * (y[6]/0.1))

	def RateC_LV_brain(self, y, w, c, t):
		return (1 / c[21]) * (((1 - c[17]) * c[28] * c[8] * (y[0]/31.9) + c[28] * c[7] * (y[3]/261.0) - (c[28] * c[8] + c[28] * c[7]) * (y[7]/22.5) - c[28] * c[12] * (1 - c[13]) * c[4] * (y[7]/22.5)) + c[28] * c[12] * (1 - c[13]) * c[4] * (1 - c[14]) * (y[6]/0.1))

	def RateC_TFV_brain(self, y, w, c, t):
		return (1 / c[22]) * (((1 - c[17]) * (1 - c[28]) * c[8] * (y[0]/31.9) + (1 - c[28]) * c[7] * (y[3]/261.0) - (c[8] + c[7]) * (y[8]/22.5) - (1 - c[28]) * c[12] * (1 - c[13]) * c[4] * (y[8]/22.5)) + (1 - c[28]) * c[12] * (1 - c[13]) * c[4] * (1 - c[14]) * (y[6]/0.1) + (c[28] * c[8] + c[28] * c[7]) * (y[7]/22.5))

	def RateC_CM_brain(self, y, w, c, t):
		return (1 / c[23]) * (c[8] + c[7]) * ((y[8]/22.5) - (y[9]/7.5))

	def RateC_SAS_brain(self, y, w, c, t):
		return (1 / c[24]) * ((c[8] + c[7]) * (y[9]/7.5) - (1 - c[26]) * c[8] * (y[10]/90.0) - c[7] * (y[10]/90.0))

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

	def __init__(self, y_indexes={'C_p_brain': 0, 'C_BBB_unbound_brain': 1, 'C_BBB_bound_brain': 2, 'C_is_brain': 3, 'C_bc_brain': 4, 'C_BCSFB_unbound_brain': 5, 'C_BCSFB_bound_brain': 6, 'C_LV_brain': 7, 'C_TFV_brain': 8, 'C_CM_brain': 9, 'C_SAS_brain': 10}, w_indexes={}, c_indexes={'Vp_brain': 0, 'VBBB_brain': 1, 'VIS_brain': 2, 'VBC_brain': 3, 'V_ES_brain': 4, 'Q_p_brain': 5, 'Q_bc_brain': 6, 'Q_ISF_brain': 7, 'Q_CSF_brain': 8, 'kon_FcRn': 9, 'koff_FcRn': 10, 'kdeg': 11, 'CLup_brain': 12, 'f_BBB': 13, 'FR': 14, 'FcRn_free_BBB': 15, 'sigma_V_BBB': 16, 'sigma_V_BCSFB': 17, 'sigma_L_brain_ISF': 18, 'L_brain': 19, 'V_BCSFB_brain': 20, 'V_LV_brain': 21, 'V_TFV_brain': 22, 'V_CM_brain': 23, 'V_SAS_brain': 24, 'f_BCSFB': 25, 'sigma_L_SAS': 26, 'FcRn_free_BCSFB': 27, 'f_LV': 28, 'C_LV_brain_0': 29, 'C_TFV_brain_0': 30, 'C_CM_brain_0': 31, 'C_SAS_brain_0': 32, 'C_p_brain_0': 33, 'C_is_brain_0': 34, 'C_p_lung_0': 35, 'C_bc_lung_0': 36, 'C_p_lung': 37, 'C_bc_lung': 38, 'brain_plasma': 39, 'BBB_unbound': 40, 'BBB_bound': 41, 'brain_ISF': 42, 'brain_blood_cells': 43, 'BCSFB_unbound': 44, 'BCSFB_bound': 45, 'LV': 46, 'TFV': 47, 'CM': 48, 'SAS': 49}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([31.9, 0.1, 261.0, 26.1, 7.25, 36402.0, 29810.0, 10.5, 21.0, 559000000.0, 23.9, 26.6, 0.3, 0.95, 0.715, 1.0, 0.95, 0.9974, 0.2, 73.0, 0.1, 22.5, 22.5, 7.5, 90.0, 0.2, 0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.51631477927063, 0.0, 4.51631477927063, 0.0, 31.9, 0.1, 0.1, 261.0, 26.1, 0.1, 0.1, 22.5, 22.5, 7.5, 90.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

