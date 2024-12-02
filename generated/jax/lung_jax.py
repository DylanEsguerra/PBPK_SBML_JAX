import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([4516.3147792706295, 0.0, 0.0, 0.0, 0.0, 0.0002491])
y_indexes = {'C_p_lung': 0, 'C_bc_lung': 1, 'C_e_unbound_lung': 2, 'C_e_bound_lung': 3, 'C_is_lung': 4, 'FcRn_free_lung': 5}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 0.0, 1000.0, 55.0, 45.0, 5.0, 181913.0, 148920.0, 364.0, 559000000.0, 23.9, 26.6, 0.55, 0.715, 0.95, 0.2, 4.51631477927063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 1000.0, 55.0, 45.0, 5.0]) 
c_indexes = {'C_p': 0, 'C_bc': 1, 'Vp_lung': 24, 'VBC_lung': 25, 'VIS_lung': 26, 'VES_lung': 27, 'Q_p_lung': 6, 'Q_bc_lung': 7, 'L_lung': 8, 'kon_FcRn': 9, 'koff_FcRn': 10, 'kdeg': 11, 'CLup_lung': 12, 'FR': 13, 'sigma_V_lung': 14, 'sigma_L_lung': 15, 'C_p_lung_0': 16, 'C_bc_lung_0': 17, 'C_is_lung_0': 18, 'C_e_unbound_lung_0': 19, 'C_e_bound_lung_0': 20, 'C_p_0': 21, 'C_bc_0': 22, 'FcRn_free_lung_0': 23}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p_lung(y, w, c, t), self.RateC_bc_lung(y, w, c, t), self.RateC_e_unbound_lung(y, w, c, t), self.RateC_e_bound_lung(y, w, c, t), self.RateC_is_lung(y, w, c, t), self.RateFcRn_free_lung(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p_lung(self, y, w, c, t):
		return (1 / c[24]) * ((c[6] * c[0] - (c[6] - c[8]) * (y[0]/1000.0) - (1 - c[14]) * c[8] * (y[0]/1000.0) - c[12] * (y[0]/1000.0)) + c[12] * c[13] * (y[3]/5.0))

	def RateC_bc_lung(self, y, w, c, t):
		return (1 / c[25]) * (c[7] * c[1] - c[7] * (y[1]/55.0))

	def RateC_e_unbound_lung(self, y, w, c, t):
		return (1 / c[27]) * ((c[12] * ((y[0]/1000.0) + (y[4]/45.0)) - c[27] * c[9] * (y[2]/5.0) * (y[5]/5.0)) + c[27] * c[10] * (y[3]/5.0) - c[11] * (y[2]/5.0) * c[27])

	def RateC_e_bound_lung(self, y, w, c, t):
		return (1 / c[27]) * (c[27] * c[9] * (y[2]/5.0) * (y[5]/5.0) - c[27] * c[10] * (y[3]/5.0) - c[12] * (y[3]/5.0))

	def RateC_is_lung(self, y, w, c, t):
		return (1 / c[26]) * (((1 - c[14]) * c[8] * (y[0]/1000.0) - (1 - c[15]) * c[8] * (y[4]/45.0)) + c[12] * (1 - c[13]) * (y[3]/5.0) - c[12] * (y[4]/45.0))

	def RateFcRn_free_lung(self, y, w, c, t):
		return (1 / c[27]) * ((c[10] * (y[3]/5.0) * c[27] - c[9] * (y[2]/5.0) * (y[5]/5.0) * c[27]) + c[12] * (y[3]/5.0))

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

	def __init__(self, y_indexes={'C_p_lung': 0, 'C_bc_lung': 1, 'C_e_unbound_lung': 2, 'C_e_bound_lung': 3, 'C_is_lung': 4, 'FcRn_free_lung': 5}, w_indexes={}, c_indexes={'C_p': 0, 'C_bc': 1, 'Vp_lung': 24, 'VBC_lung': 25, 'VIS_lung': 26, 'VES_lung': 27, 'Q_p_lung': 6, 'Q_bc_lung': 7, 'L_lung': 8, 'kon_FcRn': 9, 'koff_FcRn': 10, 'kdeg': 11, 'CLup_lung': 12, 'FR': 13, 'sigma_V_lung': 14, 'sigma_L_lung': 15, 'C_p_lung_0': 16, 'C_bc_lung_0': 17, 'C_is_lung_0': 18, 'C_e_unbound_lung_0': 19, 'C_e_bound_lung_0': 20, 'C_p_0': 21, 'C_bc_0': 22, 'FcRn_free_lung_0': 23}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([4516.3147792706295, 0.0, 0.0, 0.0, 0.0, 0.0002491]), w0=jnp.array([]), c=jnp.array([0.0, 0.0, 1000.0, 55.0, 45.0, 5.0, 181913.0, 148920.0, 364.0, 559000000.0, 23.9, 26.6, 0.55, 0.715, 0.95, 0.2, 4.51631477927063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.982e-05, 1000.0, 55.0, 45.0, 5.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

