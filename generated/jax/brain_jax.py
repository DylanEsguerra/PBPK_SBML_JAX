import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'C_p_brain': 0, 'C_BBB_unbound_brain': 1, 'C_BBB_bound_brain': 2, 'C_is_brain': 3, 'C_bc_brain': 4}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([4.51631477927063, 0.0, 0.0, 0.0, 31.9, 0.1, 261.0, 26.1, 7.25, 36402.0, 29810.0, 10.5, 21.0, 559000000.0, 23.9, 26.6, 0.3, 0.95, 0.715, 1.0, 0.95, 0.9974, 0.2, 73.0, 31.9, 0.1, 0.1, 261.0, 26.1]) 
c_indexes = {'C_p_lung': 0, 'C_bc_lung': 1, 'C_SAS_brain': 2, 'C_BCSFB_bound_brain': 3, 'Vp_brain': 4, 'VBBB_brain': 5, 'VIS_brain': 6, 'VBC_brain': 7, 'V_ES_brain': 8, 'Q_p_brain': 9, 'Q_bc_brain': 10, 'Q_ISF_brain': 11, 'Q_CSF_brain': 12, 'kon_FcRn': 13, 'koff_FcRn': 14, 'kdeg': 15, 'CLup_brain': 16, 'f_BBB': 17, 'FR': 18, 'FcRn_free_BBB': 19, 'sigma_V_BBB': 20, 'sigma_V_BCSFB': 21, 'sigma_L_brain_ISF': 22, 'L_brain': 23, 'brain_plasma': 24, 'BBB_unbound': 25, 'BBB_bound': 26, 'brain_ISF': 27, 'brain_blood_cells': 28}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p_brain(y, w, c, t), self.RateC_BBB_unbound_brain(y, w, c, t), self.RateC_BBB_bound_brain(y, w, c, t), self.RateC_is_brain(y, w, c, t), self.RateC_bc_brain(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p_brain(self, y, w, c, t):
		return (1 / c[4]) * ((c[9] * c[0] - (c[9] - c[23]) * (y[0]/31.9) - (1 - c[20]) * c[11] * (y[0]/31.9) - (1 - c[21]) * c[12] * (y[0]/31.9) - c[16] * c[8] * (y[0]/31.9)) + c[16] * c[17] * c[8] * c[18] * (y[2]/0.1) + c[16] * (1 - c[17]) * c[8] * c[18] * c[3])

	def RateC_BBB_unbound_brain(self, y, w, c, t):
		return (1 / c[5]) * ((c[16] * c[17] * c[8] * ((y[0]/31.9) + (y[3]/261.0)) - c[5] * c[13] * (y[1]/0.1) * c[19]) + c[5] * c[14] * (y[2]/0.1) - c[5] * c[15] * (y[1]/0.1))

	def RateC_BBB_bound_brain(self, y, w, c, t):
		return (1 / c[5]) * (-c[16] * c[17] * c[8] * (y[2]/0.1) + c[5] * c[13] * (y[1]/0.1) * c[19] - c[5] * c[14] * (y[2]/0.1))

	def RateC_is_brain(self, y, w, c, t):
		return (1 / c[6]) * (((1 - c[20]) * c[11] * (y[0]/31.9) - (1 - c[22]) * c[11] * (y[3]/261.0) - c[11] * (y[3]/261.0)) + c[11] * c[2] + c[16] * c[17] * c[8] * (1 - c[18]) * (y[2]/0.1) - c[16] * c[17] * c[8] * (y[3]/261.0))

	def RateC_bc_brain(self, y, w, c, t):
		return (1 / c[7]) * (c[10] * c[1] - c[10] * (y[4]/26.1))

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

	def __init__(self, y_indexes={'C_p_brain': 0, 'C_BBB_unbound_brain': 1, 'C_BBB_bound_brain': 2, 'C_is_brain': 3, 'C_bc_brain': 4}, w_indexes={}, c_indexes={'C_p_lung': 0, 'C_bc_lung': 1, 'C_SAS_brain': 2, 'C_BCSFB_bound_brain': 3, 'Vp_brain': 4, 'VBBB_brain': 5, 'VIS_brain': 6, 'VBC_brain': 7, 'V_ES_brain': 8, 'Q_p_brain': 9, 'Q_bc_brain': 10, 'Q_ISF_brain': 11, 'Q_CSF_brain': 12, 'kon_FcRn': 13, 'koff_FcRn': 14, 'kdeg': 15, 'CLup_brain': 16, 'f_BBB': 17, 'FR': 18, 'FcRn_free_BBB': 19, 'sigma_V_BBB': 20, 'sigma_V_BCSFB': 21, 'sigma_L_brain_ISF': 22, 'L_brain': 23, 'brain_plasma': 24, 'BBB_unbound': 25, 'BBB_bound': 26, 'brain_ISF': 27, 'brain_blood_cells': 28}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([4.51631477927063, 0.0, 0.0, 0.0, 31.9, 0.1, 261.0, 26.1, 7.25, 36402.0, 29810.0, 10.5, 21.0, 559000000.0, 23.9, 26.6, 0.3, 0.95, 0.715, 1.0, 0.95, 0.9974, 0.2, 73.0, 31.9, 0.1, 0.1, 261.0, 26.1]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

