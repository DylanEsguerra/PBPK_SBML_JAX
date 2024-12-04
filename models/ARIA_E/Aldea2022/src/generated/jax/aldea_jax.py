import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 3.2, 0.0])
y_indexes = {'A': 0, 'C': 1, 'Cp': 2, 'A_beta': 3, 'VWD': 4}

w0 = jnp.array([450.0, 0.0, 0.0])
w_indexes = {'Dsc': 0, 'Div': 1, 'BGTS': 2}

c = jnp.array([0.494, 0.0821, 0.22, 0.336, 3.52, 0.869, 6.38, 0.000126, 0.0124, 3.2, 0.0, 60.0, 1.0, 3.72, 1.0, 3.52, 6.38, 1.0, 1.0]) 
c_indexes = {'F': 0, 'D1': 1, 'KA': 2, 'CL': 3, 'Vc': 4, 'Q': 5, 'Vp': 6, 'alpha_removal': 7, 'k_repair': 8, 'A_beta0': 9, 'VWD0': 10, 'BGTS_max': 11, 'EG50': 12, 'pow': 13, 'absorption_site': 14, 'central': 15, 'peripheral': 16, 'local_amyloid': 17, 'VWD_compartment': 18}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.absorption(y, w, c, t), self.sc_dosing(y, w, c, t), self.iv_dosing(y, w, c, t), self.central_to_peripheral(y, w, c, t), self.peripheral_to_central(y, w, c, t), self.elimination(y, w, c, t), self.amyloid_degradation(y, w, c, t), self.vwd_production(y, w, c, t), self.vwd_degradation(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def absorption(self, y, w, c, t):
		return c[2] * y[0]


	def sc_dosing(self, y, w, c, t):
		return c[0] * w[0] / c[1]


	def iv_dosing(self, y, w, c, t):
		return w[1]


	def central_to_peripheral(self, y, w, c, t):
		return c[5] * (y[1]/3.52)


	def peripheral_to_central(self, y, w, c, t):
		return c[5] * (y[2]/6.38)


	def elimination(self, y, w, c, t):
		return c[3] * (y[1]/3.52)


	def amyloid_degradation(self, y, w, c, t):
		return c[7] * (y[1]/3.52) * y[3]


	def vwd_production(self, y, w, c, t):
		return c[7] * (y[1]/3.52) * y[3]


	def vwd_degradation(self, y, w, c, t):
		return c[8] * y[4]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set((jaxfuncs.piecewise(450, (t >= 0) & (t < (0 + c[1])), 450, (t >= 28) & (t < (28 + c[1])), 900, (t >= 56) & (t < (56 + c[1])), 900, (t >= 84) & (t < (84 + c[1])), 1200, (t >= 112) & (t < (112 + c[1])), 1200, (t >= 140) & (t < (140 + c[1])), 1200, (t >= 168) & (t < (168 + c[1])), 1200, (t >= 196) & (t < (196 + c[1])), 1200, (t >= 224) & (t < (224 + c[1])), 1200, (t >= 252) & (t < (252 + c[1])), 1200, (t >= 280) & (t < (280 + c[1])), 1200, (t >= 308) & (t < (308 + c[1])), 1200, (t >= 336) & (t < (336 + c[1])), 1200, (t >= 364) & (t < (364 + c[1])), 1200, (t >= 392) & (t < (392 + c[1])), 1200, (t >= 420) & (t < (420 + c[1])), 1200, (t >= 448) & (t < (448 + c[1])), 1200, (t >= 476) & (t < (476 + c[1])), 1200, (t >= 504) & (t < (504 + c[1])), 1200, (t >= 532) & (t < (532 + c[1])), 1200, (t >= 560) & (t < (560 + c[1])), 1200, (t >= 588) & (t < (588 + c[1])), 1200, (t >= 616) & (t < (616 + c[1])), 1200, (t >= 644) & (t < (644 + c[1])), 1200, (t >= 672) & (t < (672 + c[1])), 1200, (t >= 700) & (t < (700 + c[1])), 1200, (t >= 728) & (t < (728 + c[1])), 1200, (t >= 756) & (t < (756 + c[1])), 1200, (t >= 784) & (t < (784 + c[1])), 1200, (t >= 812) & (t < (812 + c[1])), 1200, (t >= 840) & (t < (840 + c[1])), 0)))

		w = w.at[1].set((jaxfuncs.piecewise(0, (t >= 0) & (t < (0 + 0.001)), 0, (t >= 28) & (t < (28 + 0.001)), 0, (t >= 56) & (t < (56 + 0.001)), 0, (t >= 84) & (t < (84 + 0.001)), 0, (t >= 112) & (t < (112 + 0.001)), 0, (t >= 140) & (t < (140 + 0.001)), 0, (t >= 168) & (t < (168 + 0.001)), 0, (t >= 196) & (t < (196 + 0.001)), 0, (t >= 224) & (t < (224 + 0.001)), 0, (t >= 252) & (t < (252 + 0.001)), 0, (t >= 280) & (t < (280 + 0.001)), 0, (t >= 308) & (t < (308 + 0.001)), 0, (t >= 336) & (t < (336 + 0.001)), 0, (t >= 364) & (t < (364 + 0.001)), 0, (t >= 392) & (t < (392 + 0.001)), 0, (t >= 420) & (t < (420 + 0.001)), 0, (t >= 448) & (t < (448 + 0.001)), 0, (t >= 476) & (t < (476 + 0.001)), 0, (t >= 504) & (t < (504 + 0.001)), 0, (t >= 532) & (t < (532 + 0.001)), 0, (t >= 560) & (t < (560 + 0.001)), 0, (t >= 588) & (t < (588 + 0.001)), 0, (t >= 616) & (t < (616 + 0.001)), 0, (t >= 644) & (t < (644 + 0.001)), 0, (t >= 672) & (t < (672 + 0.001)), 0, (t >= 700) & (t < (700 + 0.001)), 0, (t >= 728) & (t < (728 + 0.001)), 0, (t >= 756) & (t < (756 + 0.001)), 0, (t >= 784) & (t < (784 + 0.001)), 0, (t >= 812) & (t < (812 + 0.001)), 0, (t >= 840) & (t < (840 + 0.001)), 0)))

		w = w.at[2].set((c[11] * (y[4] / c[12])**c[13] / (1 + (y[4] / c[12])**c[13])))

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

	def __init__(self, y_indexes={'A': 0, 'C': 1, 'Cp': 2, 'A_beta': 3, 'VWD': 4}, w_indexes={'Dsc': 0, 'Div': 1, 'BGTS': 2}, c_indexes={'F': 0, 'D1': 1, 'KA': 2, 'CL': 3, 'Vc': 4, 'Q': 5, 'Vp': 6, 'alpha_removal': 7, 'k_repair': 8, 'A_beta0': 9, 'VWD0': 10, 'BGTS_max': 11, 'EG50': 12, 'pow': 13, 'absorption_site': 14, 'central': 15, 'peripheral': 16, 'local_amyloid': 17, 'VWD_compartment': 18}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 3.2, 0.0]), w0=jnp.array([450.0, 0.0, 0.0]), c=jnp.array([0.494, 0.0821, 0.22, 0.336, 3.52, 0.869, 6.38, 0.000126, 0.0124, 3.2, 0.0, 60.0, 1.0, 3.72, 1.0, 3.52, 6.38, 1.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

