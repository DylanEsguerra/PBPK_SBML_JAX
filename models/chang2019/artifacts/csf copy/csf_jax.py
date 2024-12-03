import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([525.625, 525.625, 5062.5, 5062.5, 562.5, 81000.0])
y_indexes = {'C_BCSFB_unbound_brain': 0, 'C_BCSFB_bound_brain': 1, 'C_LV_brain': 2, 'C_TFV_brain': 3, 'C_CM_brain': 4, 'C_SAS_brain': 5}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([72.5, 225.0, 7.25, 22.5, 22.5, 7.5, 90.0, 7.25, 24.0, 10.5, 0.1, 0.15, 0.15, 0.15, 0.5, 0.2, 0.1, 0.01, 0.001, 1.0, 0.5, 72.5, 72.5, 225.0, 225.0, 75.0, 900.0, 72.5, 225.0, 7.25, 7.25, 22.5, 22.5, 7.5, 90.0]) 
c_indexes = {'C_p_brain': 0, 'C_IS_brain': 1, 'V_BCSFB_brain': 2, 'V_LV_brain': 3, 'V_TFV_brain': 4, 'V_CM_brain': 5, 'V_SAS_brain': 6, 'V_ES_brain': 7, 'Q_CSF_brain': 8, 'Q_ISF_brain': 9, 'CLup_brain': 10, 'f_BBB': 11, 'f_BCSFB': 12, 'FR': 13, 'sigma_V_BCSFB': 14, 'sigma_L_SAS': 15, 'kon_FcRn': 16, 'koff_FcRn': 17, 'kdeg': 18, 'FcRn_free_BCSFB': 19, 'f_LV': 20, 'C_BCSFB_unbound_brain_0': 21, 'C_BCSFB_bound_brain_0': 22, 'C_LV_brain_0': 23, 'C_TFV_brain_0': 24, 'C_CM_brain_0': 25, 'C_SAS_brain_0': 26, 'C_p_brain_0': 27, 'C_IS_brain_0': 28, 'BCSFB_unbound': 29, 'BCSFB_bound': 30, 'LV': 31, 'TFV': 32, 'CM': 33, 'SAS': 34}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_BCSFB_unbound_brain(y, w, c, t), self.RateC_BCSFB_bound_brain(y, w, c, t), self.RateC_LV_brain(y, w, c, t), self.RateC_TFV_brain(y, w, c, t), self.RateC_CM_brain(y, w, c, t), self.RateC_SAS_brain(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_BCSFB_unbound_brain(self, y, w, c, t):
		return (1 / c[2]) * ((c[10] * c[12] * c[7] * c[0] + c[20] * c[10] * (1 - c[11]) * c[7] * (y[2]/22.5) + (1 - c[20]) * c[10] * (1 - c[11]) * c[7] * (y[3]/22.5) - c[2] * c[16] * (y[0]/7.25) * c[19]) + c[2] * c[17] * (y[1]/7.25) - c[2] * c[18] * (y[0]/7.25))

	def RateC_BCSFB_bound_brain(self, y, w, c, t):
		return (1 / c[2]) * (-c[10] * (1 - c[11]) * c[7] * (y[1]/7.25) + c[2] * c[16] * (y[0]/7.25) * c[19] - c[2] * c[17] * (y[1]/7.25))

	def RateC_LV_brain(self, y, w, c, t):
		return (1 / c[3]) * (((1 - c[14]) * c[20] * c[8] * c[0] + c[20] * c[9] * c[1] - (c[20] * c[8] + c[20] * c[9]) * (y[2]/22.5) - c[20] * c[10] * (1 - c[11]) * c[7] * (y[2]/22.5)) + c[20] * c[10] * (1 - c[11]) * c[7] * (1 - c[13]) * (y[1]/7.25))

	def RateC_TFV_brain(self, y, w, c, t):
		return (1 / c[4]) * (((1 - c[14]) * (1 - c[20]) * c[8] * c[0] + (1 - c[20]) * c[9] * c[1] - (c[8] + c[9]) * (y[3]/22.5) - (1 - c[20]) * c[10] * (1 - c[11]) * c[7] * (y[3]/22.5)) + (1 - c[20]) * c[10] * (1 - c[11]) * c[7] * (1 - c[13]) * (y[1]/7.25) + (c[20] * c[8] + c[20] * c[9]) * (y[2]/22.5))

	def RateC_CM_brain(self, y, w, c, t):
		return (1 / c[5]) * (c[8] + c[9]) * ((y[3]/22.5) - (y[4]/7.5))

	def RateC_SAS_brain(self, y, w, c, t):
		return (1 / c[6]) * ((c[8] + c[9]) * (y[4]/7.5) - (1 - c[15]) * c[8] * (y[5]/90.0) - c[9] * (y[5]/90.0))

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

	def __init__(self, y_indexes={'C_BCSFB_unbound_brain': 0, 'C_BCSFB_bound_brain': 1, 'C_LV_brain': 2, 'C_TFV_brain': 3, 'C_CM_brain': 4, 'C_SAS_brain': 5}, w_indexes={}, c_indexes={'C_p_brain': 0, 'C_IS_brain': 1, 'V_BCSFB_brain': 2, 'V_LV_brain': 3, 'V_TFV_brain': 4, 'V_CM_brain': 5, 'V_SAS_brain': 6, 'V_ES_brain': 7, 'Q_CSF_brain': 8, 'Q_ISF_brain': 9, 'CLup_brain': 10, 'f_BBB': 11, 'f_BCSFB': 12, 'FR': 13, 'sigma_V_BCSFB': 14, 'sigma_L_SAS': 15, 'kon_FcRn': 16, 'koff_FcRn': 17, 'kdeg': 18, 'FcRn_free_BCSFB': 19, 'f_LV': 20, 'C_BCSFB_unbound_brain_0': 21, 'C_BCSFB_bound_brain_0': 22, 'C_LV_brain_0': 23, 'C_TFV_brain_0': 24, 'C_CM_brain_0': 25, 'C_SAS_brain_0': 26, 'C_p_brain_0': 27, 'C_IS_brain_0': 28, 'BCSFB_unbound': 29, 'BCSFB_bound': 30, 'LV': 31, 'TFV': 32, 'CM': 33, 'SAS': 34}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([525.625, 525.625, 5062.5, 5062.5, 562.5, 81000.0]), w0=jnp.array([]), c=jnp.array([72.5, 225.0, 7.25, 22.5, 22.5, 7.5, 90.0, 7.25, 24.0, 10.5, 0.1, 0.15, 0.15, 0.15, 0.5, 0.2, 0.1, 0.01, 0.001, 1.0, 0.5, 72.5, 72.5, 225.0, 225.0, 75.0, 900.0, 72.5, 225.0, 7.25, 7.25, 22.5, 22.5, 7.5, 90.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

