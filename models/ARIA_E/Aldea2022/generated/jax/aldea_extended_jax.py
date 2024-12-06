import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 3.2, 0.0, 1.0, 1.0, 1.0, 1.0])
y_indexes = {'A': 0, 'C': 1, 'Cp': 2, 'A_beta': 3, 'VWD': 4, 'APP': 5, 'C83': 6, 'C99': 7, 'p3': 8}

w0 = jnp.array([450.0, 0.0, 0.0])
w_indexes = {'Dsc': 0, 'Div': 1, 'BGTS': 2}

c = jnp.array([0.494, 0.0821, 0.22, 0.336, 3.52, 0.869, 6.38, 0.000126, 0.0124, 3.2, 0.0, 60.0, 1.0, 3.72, 0.8, 1.1, 0.153, 14.6, 1.0, 0.0223, 0.0186, 1.64, 28.8, 0.915, 0.0672, 1.0, 1.0, 1.0, 1.0, 1.0, 3.52, 6.38, 1.0, 1.0]) 
c_indexes = {'F': 0, 'D1': 1, 'KA': 2, 'CL': 3, 'Vc': 4, 'Q': 5, 'Vp': 6, 'alpha_removal': 7, 'k_repair': 8, 'A_beta0': 9, 'VWD0': 10, 'BGTS_max': 11, 'EG50': 12, 'pow': 13, 'vr0': 14, 'Vm1': 15, 'Vm2': 16, 'Vm3': 17, 'Vm4': 18, 'Vm5': 19, 'Km1': 20, 'Km2': 21, 'Km3': 22, 'Km4': 23, 'Km5': 24, 'APP0': 25, 'C83_0': 26, 'C99_0': 27, 'p3_0': 28, 'absorption_site': 29, 'central': 30, 'peripheral': 31, 'local_amyloid': 32, 'VWD_compartment': 33}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.absorption(y, w, c, t), self.sc_dosing(y, w, c, t), self.iv_dosing(y, w, c, t), self.central_to_peripheral(y, w, c, t), self.peripheral_to_central(y, w, c, t), self.elimination(y, w, c, t), self.amyloid_degradation(y, w, c, t), self.vwd_production(y, w, c, t), self.vwd_degradation(y, w, c, t), self.app_production(y, w, c, t), self.app_to_c83(y, w, c, t), self.app_to_c99(y, w, c, t), self.c83_to_p3(y, w, c, t), self.c99_to_abeta(y, w, c, t), self.c99_to_c83(y, w, c, t)], dtype=jnp.float32)

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


	def app_production(self, y, w, c, t):
		return c[14]


	def app_to_c83(self, y, w, c, t):
		return c[15] * y[5] / c[20] / (1 + y[5] / c[20] + y[7] / c[24])


	def app_to_c99(self, y, w, c, t):
		return c[16] * y[5] / c[21] / (1 + y[5] / c[21])


	def c83_to_p3(self, y, w, c, t):
		return c[17] * y[6] / c[22] / (1 + y[6] / c[22] + y[7] / c[23])


	def c99_to_abeta(self, y, w, c, t):
		return c[18] * y[7] / c[23] / (1 + y[6] / c[22] + y[7] / c[23])


	def c99_to_c83(self, y, w, c, t):
		return c[19] * y[7] / c[24] / (1 + y[5] / c[20] + y[7] / c[24])

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[1].set((jaxfuncs.piecewise(0, (t >= 0) & (t < (0 + 0.001)), 0, (t >= 28) & (t < (28 + 0.001)), 0, (t >= 56) & (t < (56 + 0.001)), 0, (t >= 84) & (t < (84 + 0.001)), 0, (t >= 112) & (t < (112 + 0.001)), 0, (t >= 140) & (t < (140 + 0.001)), 0, (t >= 168) & (t < (168 + 0.001)), 0, (t >= 196) & (t < (196 + 0.001)), 0, (t >= 224) & (t < (224 + 0.001)), 0, (t >= 252) & (t < (252 + 0.001)), 0, (t >= 280) & (t < (280 + 0.001)), 0, (t >= 308) & (t < (308 + 0.001)), 0, (t >= 336) & (t < (336 + 0.001)), 0, (t >= 364) & (t < (364 + 0.001)), 0, (t >= 392) & (t < (392 + 0.001)), 0, (t >= 420) & (t < (420 + 0.001)), 0, (t >= 448) & (t < (448 + 0.001)), 0, (t >= 476) & (t < (476 + 0.001)), 0, (t >= 504) & (t < (504 + 0.001)), 0, (t >= 532) & (t < (532 + 0.001)), 0, (t >= 560) & (t < (560 + 0.001)), 0, (t >= 588) & (t < (588 + 0.001)), 0, (t >= 616) & (t < (616 + 0.001)), 0, (t >= 644) & (t < (644 + 0.001)), 0, (t >= 672) & (t < (672 + 0.001)), 0, (t >= 700) & (t < (700 + 0.001)), 0, (t >= 728) & (t < (728 + 0.001)), 0, (t >= 756) & (t < (756 + 0.001)), 0, (t >= 784) & (t < (784 + 0.001)), 0, (t >= 812) & (t < (812 + 0.001)), 0, (t >= 840) & (t < (840 + 0.001)), 0)))

		w = w.at[2].set((c[11] * (y[4] / c[12])**c[13] / (1 + (y[4] / c[12])**c[13])))

		w = w.at[0].set((jaxfuncs.piecewise(0, w[2] > 4, jaxfuncs.piecewise(450, (t >= 0) & (t < (0 + c[1])), 0, (t >= 7) & (t < (7 + c[1])), 0, (t >= 14) & (t < (14 + c[1])), 0, (t >= 21) & (t < (21 + c[1])), 450, (t >= 28) & (t < (28 + c[1])), 0, (t >= 35) & (t < (35 + c[1])), 0, (t >= 42) & (t < (42 + c[1])), 0, (t >= 49) & (t < (49 + c[1])), 900, (t >= 56) & (t < (56 + c[1])), 0, (t >= 63) & (t < (63 + c[1])), 0, (t >= 70) & (t < (70 + c[1])), 0, (t >= 77) & (t < (77 + c[1])), 900, (t >= 84) & (t < (84 + c[1])), 0, (t >= 91) & (t < (91 + c[1])), 0, (t >= 98) & (t < (98 + c[1])), 0, (t >= 105) & (t < (105 + c[1])), 1200, (t >= 112) & (t < (112 + c[1])), 0, (t >= 119) & (t < (119 + c[1])), 0, (t >= 126) & (t < (126 + c[1])), 0, (t >= 133) & (t < (133 + c[1])), 1200, (t >= 140) & (t < (140 + c[1])), 0, (t >= 147) & (t < (147 + c[1])), 0, (t >= 154) & (t < (154 + c[1])), 0, (t >= 161) & (t < (161 + c[1])), 1200, (t >= 168) & (t < (168 + c[1])), 0, (t >= 175) & (t < (175 + c[1])), 0, (t >= 182) & (t < (182 + c[1])), 0, (t >= 189) & (t < (189 + c[1])), 1200, (t >= 196) & (t < (196 + c[1])), 0, (t >= 203) & (t < (203 + c[1])), 0, (t >= 210) & (t < (210 + c[1])), 0, (t >= 217) & (t < (217 + c[1])), 1200, (t >= 224) & (t < (224 + c[1])), 0, (t >= 231) & (t < (231 + c[1])), 0, (t >= 238) & (t < (238 + c[1])), 0, (t >= 245) & (t < (245 + c[1])), 1200, (t >= 252) & (t < (252 + c[1])), 0, (t >= 259) & (t < (259 + c[1])), 0, (t >= 266) & (t < (266 + c[1])), 0, (t >= 273) & (t < (273 + c[1])), 1200, (t >= 280) & (t < (280 + c[1])), 0, (t >= 287) & (t < (287 + c[1])), 0, (t >= 294) & (t < (294 + c[1])), 0, (t >= 301) & (t < (301 + c[1])), 1200, (t >= 308) & (t < (308 + c[1])), 0, (t >= 315) & (t < (315 + c[1])), 0, (t >= 322) & (t < (322 + c[1])), 0, (t >= 329) & (t < (329 + c[1])), 1200, (t >= 336) & (t < (336 + c[1])), 0, (t >= 343) & (t < (343 + c[1])), 0, (t >= 350) & (t < (350 + c[1])), 0, (t >= 357) & (t < (357 + c[1])), 1200, (t >= 364) & (t < (364 + c[1])), 0, (t >= 371) & (t < (371 + c[1])), 0, (t >= 378) & (t < (378 + c[1])), 0, (t >= 385) & (t < (385 + c[1])), 1200, (t >= 392) & (t < (392 + c[1])), 0, (t >= 399) & (t < (399 + c[1])), 0, (t >= 406) & (t < (406 + c[1])), 0, (t >= 413) & (t < (413 + c[1])), 1200, (t >= 420) & (t < (420 + c[1])), 0, (t >= 427) & (t < (427 + c[1])), 0, (t >= 434) & (t < (434 + c[1])), 0, (t >= 441) & (t < (441 + c[1])), 1200, (t >= 448) & (t < (448 + c[1])), 0, (t >= 455) & (t < (455 + c[1])), 0, (t >= 462) & (t < (462 + c[1])), 0, (t >= 469) & (t < (469 + c[1])), 1200, (t >= 476) & (t < (476 + c[1])), 0, (t >= 483) & (t < (483 + c[1])), 0, (t >= 490) & (t < (490 + c[1])), 0, (t >= 497) & (t < (497 + c[1])), 1200, (t >= 504) & (t < (504 + c[1])), 0, (t >= 511) & (t < (511 + c[1])), 0, (t >= 518) & (t < (518 + c[1])), 0, (t >= 525) & (t < (525 + c[1])), 1200, (t >= 532) & (t < (532 + c[1])), 0, (t >= 539) & (t < (539 + c[1])), 0, (t >= 546) & (t < (546 + c[1])), 0, (t >= 553) & (t < (553 + c[1])), 1200, (t >= 560) & (t < (560 + c[1])), 0, (t >= 567) & (t < (567 + c[1])), 0, (t >= 574) & (t < (574 + c[1])), 0, (t >= 581) & (t < (581 + c[1])), 1200, (t >= 588) & (t < (588 + c[1])), 0, (t >= 595) & (t < (595 + c[1])), 0, (t >= 602) & (t < (602 + c[1])), 0, (t >= 609) & (t < (609 + c[1])), 1200, (t >= 616) & (t < (616 + c[1])), 0, (t >= 623) & (t < (623 + c[1])), 0, (t >= 630) & (t < (630 + c[1])), 0, (t >= 637) & (t < (637 + c[1])), 1200, (t >= 644) & (t < (644 + c[1])), 0, (t >= 651) & (t < (651 + c[1])), 0, (t >= 658) & (t < (658 + c[1])), 0, (t >= 665) & (t < (665 + c[1])), 1200, (t >= 672) & (t < (672 + c[1])), 0, (t >= 679) & (t < (679 + c[1])), 0, (t >= 686) & (t < (686 + c[1])), 0, (t >= 693) & (t < (693 + c[1])), 1200, (t >= 700) & (t < (700 + c[1])), 0, (t >= 707) & (t < (707 + c[1])), 0, (t >= 714) & (t < (714 + c[1])), 0, (t >= 721) & (t < (721 + c[1])), 0, (t >= 728) & (t < (728 + c[1])), 0, (t >= 735) & (t < (735 + c[1])), 0, (t >= 742) & (t < (742 + c[1])), 0, (t >= 749) & (t < (749 + c[1])), 0, (t >= 756) & (t < (756 + c[1])), 0, (t >= 763) & (t < (763 + c[1])), 0, (t >= 770) & (t < (770 + c[1])), 0, (t >= 777) & (t < (777 + c[1])), 0, (t >= 784) & (t < (784 + c[1])), 0, (t >= 791) & (t < (791 + c[1])), 0, (t >= 798) & (t < (798 + c[1])), 0, (t >= 805) & (t < (805 + c[1])), 0, (t >= 812) & (t < (812 + c[1])), 0, (t >= 819) & (t < (819 + c[1])), 0, (t >= 826) & (t < (826 + c[1])), 0, (t >= 833) & (t < (833 + c[1])), 0, (t >= 840) & (t < (840 + c[1])), 0))))

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

	def __init__(self, y_indexes={'A': 0, 'C': 1, 'Cp': 2, 'A_beta': 3, 'VWD': 4, 'APP': 5, 'C83': 6, 'C99': 7, 'p3': 8}, w_indexes={'Dsc': 0, 'Div': 1, 'BGTS': 2}, c_indexes={'F': 0, 'D1': 1, 'KA': 2, 'CL': 3, 'Vc': 4, 'Q': 5, 'Vp': 6, 'alpha_removal': 7, 'k_repair': 8, 'A_beta0': 9, 'VWD0': 10, 'BGTS_max': 11, 'EG50': 12, 'pow': 13, 'vr0': 14, 'Vm1': 15, 'Vm2': 16, 'Vm3': 17, 'Vm4': 18, 'Vm5': 19, 'Km1': 20, 'Km2': 21, 'Km3': 22, 'Km4': 23, 'Km5': 24, 'APP0': 25, 'C83_0': 26, 'C99_0': 27, 'p3_0': 28, 'absorption_site': 29, 'central': 30, 'peripheral': 31, 'local_amyloid': 32, 'VWD_compartment': 33}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 3.2, 0.0, 1.0, 1.0, 1.0, 1.0]), w0=jnp.array([450.0, 0.0, 0.0]), c=jnp.array([0.494, 0.0821, 0.22, 0.336, 3.52, 0.869, 6.38, 0.000126, 0.0124, 3.2, 0.0, 60.0, 1.0, 3.72, 0.8, 1.1, 0.153, 14.6, 1.0, 0.0223, 0.0186, 1.64, 28.8, 0.915, 0.0672, 1.0, 1.0, 1.0, 1.0, 1.0, 3.52, 6.38, 1.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

