import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.8])
y_indexes = {'C_p_liver': 0, 'C_bc_liver': 1, 'C_is_liver': 2, 'C_e_unbound_liver': 3, 'C_e_bound_liver': 4, 'FcRn_free_liver': 5}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([108.0, 88.5, 435.0, 10.8, 43200.0, 35400.0, 1080.0, 885.0, 1080.0, 885.0, 4320.0, 3540.0, 4320.0, 3540.0, 432.0, 10.8, 10.8, 43.2, 43.2, 0.1, 0.01, 26.6, 0.05, 0.95, 0.2, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 108.0, 88.5, 435.0, 10.8]) 
c_indexes = {'Vp_liver': 52, 'VBC_liver': 53, 'VIS_liver': 54, 'VES_liver': 55, 'Q_p_liver': 4, 'Q_bc_liver': 5, 'Q_p_spleen': 6, 'Q_bc_spleen': 7, 'Q_p_pancreas': 8, 'Q_bc_pancreas': 9, 'Q_p_SI': 10, 'Q_bc_SI': 11, 'Q_p_LI': 12, 'Q_bc_LI': 13, 'L_liver': 14, 'L_spleen': 15, 'L_pancreas': 16, 'L_SI': 17, 'L_LI': 18, 'kon_FcRn': 19, 'koff_FcRn': 20, 'kdeg': 21, 'CLup_liver': 22, 'sigma_V_liver': 23, 'sigma_L_liver': 24, 'FR': 25, 'C_p_liver_0': 26, 'C_bc_liver_0': 27, 'C_is_liver_0': 28, 'C_e_unbound_liver_0': 29, 'C_e_bound_liver_0': 30, 'FcRn_free_liver_0': 31, 'C_p_lung_0': 32, 'C_bc_lung_0': 33, 'C_p_spleen_0': 34, 'C_bc_spleen_0': 35, 'C_p_pancreas_0': 36, 'C_bc_pancreas_0': 37, 'C_p_SI_0': 38, 'C_bc_SI_0': 39, 'C_p_LI_0': 40, 'C_bc_LI_0': 41, 'C_p_lung': 42, 'C_bc_lung': 43, 'C_p_spleen': 44, 'C_bc_spleen': 45, 'C_p_pancreas': 46, 'C_bc_pancreas': 47, 'C_p_SI': 48, 'C_bc_SI': 49, 'C_p_LI': 50, 'C_bc_LI': 51}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC_p_liver(y, w, c, t), self.RateC_bc_liver(y, w, c, t), self.RateC_is_liver(y, w, c, t), self.RateC_e_unbound_liver(y, w, c, t), self.RateC_e_bound_liver(y, w, c, t), self.RateFcRn_free_liver(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC_p_liver(self, y, w, c, t):
		return (1 / c[52]) * ((c[4] * c[42] + (c[6] - c[15]) * c[44] + (c[8] - c[16]) * c[46] + (c[10] - c[17]) * c[48] + (c[12] - c[18]) * c[50] - ((c[4] - c[14]) + (c[6] - c[15]) + (c[8] - c[16]) + (c[10] - c[17]) + (c[12] - c[18])) * (y[0]/108.0) - (1 - c[23]) * c[14] * (y[0]/108.0) - c[22] * (y[0]/108.0)) + c[22] * c[25] * (y[4]/10.8))

	def RateC_bc_liver(self, y, w, c, t):
		return (1 / c[53]) * (c[5] * c[43] + c[7] * c[45] + c[9] * c[47] + c[11] * c[49] + c[13] * c[51] - (c[5] + c[7] + c[9] + c[11] + c[13]) * (y[1]/88.5))

	def RateC_is_liver(self, y, w, c, t):
		return (1 / c[54]) * (((1 - c[23]) * c[14] * (y[0]/108.0) - (1 - c[24]) * c[14] * (y[2]/435.0)) + c[22] * (1 - c[25]) * (y[4]/10.8) - c[22] * (y[2]/435.0))

	def RateC_e_unbound_liver(self, y, w, c, t):
		return (1 / c[55]) * ((c[22] * ((y[0]/108.0) + (y[2]/435.0)) - c[55] * c[19] * (y[3]/10.8) * (y[5]/10.8)) + c[55] * c[20] * (y[4]/10.8) - c[21] * (y[3]/10.8) * c[55])

	def RateC_e_bound_liver(self, y, w, c, t):
		return (1 / c[55]) * (c[55] * c[19] * (y[3]/10.8) * (y[5]/10.8) - c[55] * c[20] * (y[4]/10.8) - c[22] * (y[4]/10.8))

	def RateFcRn_free_liver(self, y, w, c, t):
		return (1 / c[55]) * ((c[20] * (y[4]/10.8) * c[55] - c[19] * (y[3]/10.8) * (y[5]/10.8) * c[55]) + c[22] * (y[4]/10.8))

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

	def __init__(self, y_indexes={'C_p_liver': 0, 'C_bc_liver': 1, 'C_is_liver': 2, 'C_e_unbound_liver': 3, 'C_e_bound_liver': 4, 'FcRn_free_liver': 5}, w_indexes={}, c_indexes={'Vp_liver': 52, 'VBC_liver': 53, 'VIS_liver': 54, 'VES_liver': 55, 'Q_p_liver': 4, 'Q_bc_liver': 5, 'Q_p_spleen': 6, 'Q_bc_spleen': 7, 'Q_p_pancreas': 8, 'Q_bc_pancreas': 9, 'Q_p_SI': 10, 'Q_bc_SI': 11, 'Q_p_LI': 12, 'Q_bc_LI': 13, 'L_liver': 14, 'L_spleen': 15, 'L_pancreas': 16, 'L_SI': 17, 'L_LI': 18, 'kon_FcRn': 19, 'koff_FcRn': 20, 'kdeg': 21, 'CLup_liver': 22, 'sigma_V_liver': 23, 'sigma_L_liver': 24, 'FR': 25, 'C_p_liver_0': 26, 'C_bc_liver_0': 27, 'C_is_liver_0': 28, 'C_e_unbound_liver_0': 29, 'C_e_bound_liver_0': 30, 'FcRn_free_liver_0': 31, 'C_p_lung_0': 32, 'C_bc_lung_0': 33, 'C_p_spleen_0': 34, 'C_bc_spleen_0': 35, 'C_p_pancreas_0': 36, 'C_bc_pancreas_0': 37, 'C_p_SI_0': 38, 'C_bc_SI_0': 39, 'C_p_LI_0': 40, 'C_bc_LI_0': 41, 'C_p_lung': 42, 'C_bc_lung': 43, 'C_p_spleen': 44, 'C_bc_spleen': 45, 'C_p_pancreas': 46, 'C_bc_pancreas': 47, 'C_p_SI': 48, 'C_bc_SI': 49, 'C_p_LI': 50, 'C_bc_LI': 51}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.8]), w0=jnp.array([]), c=jnp.array([108.0, 88.5, 435.0, 10.8, 43200.0, 35400.0, 1080.0, 885.0, 1080.0, 885.0, 4320.0, 3540.0, 4320.0, 3540.0, 432.0, 10.8, 10.8, 43.2, 43.2, 0.1, 0.01, 26.6, 0.05, 0.95, 0.2, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 108.0, 88.5, 435.0, 10.8]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

