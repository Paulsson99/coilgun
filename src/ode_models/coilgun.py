import numpy as np
from scipy import integrate
from typing import Callable

def drag_force(v):
	"""Drag force from the air"""
	rho = 1.2 # Air density [kg/m^3]
	C_d = 100 # Drag coefficient, roughly 0.5 for a sphere
	r = 4.6e-3 # Radius of the ball
	A = np.pi * r**2

	return rho * C_d * A * v**2 / 2 * np.sign(-v)


def ode_solver_coilgun(
	C: float, 		# Capacitance of the capaitance bank
	R: float, 		# Resistance of the coil
	m: float, 		# Mass of the projectile
	L: Callable, 	# Function that calculates the inductance of the coil at the position x
	dLdx: Callable, # Function that calculates the derivative of the inductance at the position x
	x0: float,		# Starting position relative to the center of the coil for the projectile
	t_max: float,	# Maximum time for the simulation
	V0: float,		# Starting voltage over the capacitance bank
	v0: float, 		# Starting velocity of the projectile
	x1: float=None, # If present the simulation will end when the projectile passes this point
	t_steps: int=None # Minimum steps the solver should take
):
	"""
	ODE solver for a coilgun.
	This method assumes that the inductance is a known function of the position
	"""

	def dydt(t, y):
		"""
		Calculate the derivative of the state [x, v, I, dIdt, V]
		"""
		x, v, I, dIdt, V = y

		dvdt = I**2 * dLdx(x) / (2*m)
		dIdt2 = -I / (L(x)*C) - R*dIdt / L(x) - dIdt*dLdx(x)*v/L(x)
		dVdt = -I/C


		return [v, dvdt, dIdt, dIdt2, dVdt]

	def end_point_reached(t, y):
		"""Stop the solver if the end point is reached"""
		return y[0] - x1

	def no_reverse_voltage(t, y):
		"""
		Stop the solver when the voltage over the capacitance goes under zero
		This is because the diode that protects the circuit
		"""
		return y[4]

	# Stop at event
	end_point_reached.terminal = True
	no_reverse_voltage.terminal = True

	# Set the events
	events = [no_reverse_voltage]
	if x1 is not None:
		events.append(end_point_reached)

	# Calculate maximum time step
	t_step = t_max / t_steps if t_steps is not None else np.inf

	# Calculate initial conditions
	y0 = [x0, v0, 0.0, V0/L(x0), V0]

	sol = integrate.solve_ivp(
		fun=dydt, 
		y0=y0, 
		t_span=(0, t_max), 
		events=events,
		max_step=t_step
	)

	t = sol.t
	x, v, I, dIdt, V = sol.y

	# Calculate voltage over capacitance bank
	#V = V0 - 1/C*integrate.cumulative_trapezoid(I, t, initial=0)

	# # Power loss
	# P_R = R*I*I
	# P_C = V*I
	# P_L = L(x)*dIdt*I

	# # Force
	# F = I**2 * dLdx(x) / 2
	# a = F / m

	# # Print enery loss by the circuit
	# print(f"Energy lost by the capacitator: {integrate.trapz(P_C, t)} J")
	# print(f"Energy lost by the resistance: {integrate.trapz(P_R, t)} J")
	# print(f"Energy lost by the inductor: {integrate.trapz(P_L, t)} J")
	# print(f"Energy lost by the resistance and inductor: {integrate.trapz(P_R+P_L, t)} J")
	# print(f"Final velocity: {integrate.trapz(a, t)} m/s")
	# print(f"Work on the projectile: {integrate.trapz(F, x)} J")
	# print(f"Energy gained by the projectile: {m/2*(v[-1]**2-v[0]**2)} J")

	return t, x, v, I, V, dIdt

def ode_solver_RL(
	R: float, 		# Resistance of the coil
	m: float, 		# Mass of the projectile
	L: Callable, 	# Function that calculates the inductance of the coil at the position x
	dLdx: Callable, # Function that calculates the derivative of the inductance at the position x
	x0: float,		# Starting position relative to the center of the coil for the projectile
	t_max: float,	# Maximum time for the simulation
	v0: float, 		# Starting velocity of the projectile
	I0: float, 		# Starting currnet through the coil
	x1: float=None, # If present the simulation will end when the projectile passes this point
	t_steps: int=None # Minimum steps the solver should take
):
	"""
	ODE solver for a coilgun after the contact to capacitance bank is closed.
	This method assumes that the inductance is a known function of the position
	"""
	def dydt(t, y):
		"""
		Calculate the derivative of the state [x, v, I]
		"""
		x, v, I = y

		dvdt = I**2 * dLdx(x) / (2*m)
		dIdt = -I*R/L(x)

		return [v, dvdt, dIdt]

	def end_point_reached(t, y):
		"""Stop the solver if the end point is reached"""
		return y[0] - x1

	# Stop at event
	end_point_reached.terminal = True

	# Set the events
	events = None
	if x1 is not None:
		events = end_point_reached

	# Calculate maximum time step
	t_step = t_max / t_steps if t_steps is not None else np.inf

	# Calculate initial conditions
	y0 = [x0, v0, I0]

	sol = integrate.solve_ivp(
		fun=dydt, 
		y0=y0, 
		t_span=(0, t_max), 
		events=events,
		max_step=t_step
	)

	t = sol.t
	x, v, I = sol.y

	return t, x, v, I

def calculate_efficiency(
	v0: float,	# Starting velocity for the projectile
	v1: float, 	# Ending velocity for the projectile
	V0: float, 	# Stating voltage over the capacitance bank
	V1: float,	# Ending voltage over the capacitance bank
	m: float, 	# Mass of the projectile
	C: float 	# Capacitance of the capacitnace bank
) -> float:
	"""
	Calculate the efficiency of a coilgun
	"""
	# Kinetic energy gained of the projectile
	dE_k = m*(v1**2-v0**2)/2
	# Energy lost in the capacitance bank
	dE_C = C*(V0**2-V1**2)/2
	
	return dE_k / dE_C

