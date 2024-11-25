import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([15551.25, 525.625, 525.625, 61875.0, 12723.75])
y_indexes = {'C_p_brain': 0, 'C_BBB_unbound_brain': 1, 'C_BBB_bound_brain': 2, 'C_IS_brain': 3, 'C_BC_brain': 4}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 0.0, 900.0, 72.5, 214.5, 7.25, 275.0, 175.5, 7.25, 21453.0, 17553.0, 10.5, 24.0, 0.1, 0.01, 0.001, 0.1, 0.15, 0.15, 1.0, 0.5, 0.5, 0.2, 215.0, 214.5, 7.25, 7.25, 275.0, 175.5]) 
c_indexes = {'C_p_lung': 0, 'C_BC_lung': 1, 'C_SAS_brain': 2, 'C_BCSFB_bound_brain': 3, 'Vp_brain': 4, 'VBBB_brain': 5, 'VIS_brain': 6, 'VBC_brain': 7, 'V_ES_brain': 8, 'Q_p_brain': 9, 'Q_bc_brain': 10, 'Q_ISF_brain': 11, 'Q_CSF_brain': 12, 'kon_FcRn': 13, 'koff_FcRn': 14, 'kdeg': 15, 'CLup_brain': 16, 'f_BBB': 17, 'FR': 18, 'FcRn_free_BBB': 19, 'sigma_V_BBB': 20, 'sigma_V_BCSFB': 21, 'sigma_L_brain_ISF': 22, 'L_brain': 23, 'brain_plasma': 24, 'BBB_unbound': 25, 'BBB_bound': 26, 'brain_ISF': 27, 'brain_blood_cells': 28}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p_brain(y, w, c, t), self.RateC_BBB_unbound_brain(y, w, c, t), self.RateC_BBB_bound_brain(y, w, c, t), self.RateC_IS_brain(y, w, c, t), self.RateC_BC_brain(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p_brain(self, y, w, c, t):
		return (1 / c[4]) * ((c[9] * c[0] - (c[9] - c[23]) * (y[0]/214.5) - (1 - c[20]) * c[11] * (y[0]/214.5) - (1 - c[21]) * c[12] * (y[0]/214.5) - c[16] * c[8] * (y[0]/214.5)) + c[16] * c[17] * c[8] * c[18] * (y[2]/7.25) + c[16] * (1 - c[17]) * c[8] * c[18] * c[3])

	def RateC_BBB_unbound_brain(self, y, w, c, t):
		return (1 / c[5]) * ((c[16] * c[17] * c[8] * ((y[0]/214.5) + (y[3]/275.0)) - c[5] * c[13] * (y[1]/7.25) * c[19]) + c[5] * c[14] * (y[2]/7.25) - c[5] * c[15] * (y[1]/7.25))

	def RateC_BBB_bound_brain(self, y, w, c, t):
		return (1 / c[5]) * (-c[16] * c[17] * c[8] * (y[2]/7.25) + c[5] * c[13] * (y[1]/7.25) * c[19] - c[5] * c[14] * (y[2]/7.25))

	def RateC_IS_brain(self, y, w, c, t):
		return (1 / c[6]) * (((1 - c[20]) * c[11] * (y[0]/214.5) - (1 - c[22]) * c[11] * (y[3]/275.0) - c[11] * (y[3]/275.0)) + c[11] * c[2] + c[16] * c[17] * c[8] * (1 - c[18]) * (y[2]/7.25) - c[16] * c[17] * c[8] * (y[3]/275.0))

	def RateC_BC_brain(self, y, w, c, t):
		return (1 / c[7]) * (c[10] * c[1] - c[10] * (y[4]/175.5))

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

	def __init__(self, y_indexes={'C_p_brain': 0, 'C_BBB_unbound_brain': 1, 'C_BBB_bound_brain': 2, 'C_IS_brain': 3, 'C_BC_brain': 4}, w_indexes={}, c_indexes={'C_p_lung': 0, 'C_BC_lung': 1, 'C_SAS_brain': 2, 'C_BCSFB_bound_brain': 3, 'Vp_brain': 4, 'VBBB_brain': 5, 'VIS_brain': 6, 'VBC_brain': 7, 'V_ES_brain': 8, 'Q_p_brain': 9, 'Q_bc_brain': 10, 'Q_ISF_brain': 11, 'Q_CSF_brain': 12, 'kon_FcRn': 13, 'koff_FcRn': 14, 'kdeg': 15, 'CLup_brain': 16, 'f_BBB': 17, 'FR': 18, 'FcRn_free_BBB': 19, 'sigma_V_BBB': 20, 'sigma_V_BCSFB': 21, 'sigma_L_brain_ISF': 22, 'L_brain': 23, 'brain_plasma': 24, 'BBB_unbound': 25, 'BBB_bound': 26, 'brain_ISF': 27, 'brain_blood_cells': 28}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([15551.25, 525.625, 525.625, 61875.0, 12723.75]), w0=jnp.array([]), c=jnp.array([0.0, 0.0, 900.0, 72.5, 214.5, 7.25, 275.0, 175.5, 7.25, 21453.0, 17553.0, 10.5, 24.0, 0.1, 0.01, 0.001, 0.1, 0.15, 0.15, 1.0, 0.5, 0.5, 0.2, 215.0, 214.5, 7.25, 7.25, 275.0, 175.5]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

