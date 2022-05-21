from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from .DNA import DNA
from coilgun.coil import Coil, CoilEnum, GeometryCoil, Solenoid
from coilgun.power_source import ConstantCurrent, ConstantVoltage, PowerSource, PowerSourceEnum
from coilgun.projectile import Projectile1D, MagneticProjectile, FerromageneticProjectile, ProjectileEnum
from simulation.simulate import CoilgunSimulation, SimulationConf

from ode_models.coilgun import ode_solver_coilgun, calculate_efficiency
from ode_models.simulation import CoilgunSimulationODE

def coil_from_DNA(dna: DNA) -> Coil:
	"""Create a coil from the DNA"""
	coil_enum = CoilEnum[dna["CoilType"]]

	if coil_enum == CoilEnum.GeometryCoil:
		return GeometryCoil(
			coils=dna["coils"],
			inner_diameter=dna["inner_diameter"],
			wire_diameter=dna["wire_diameter"],
			resistivity=dna["resistivity"]
		)
	elif coil_enum == CoilEnum.Solenoid:
		return Solenoid(
			L=dna["coil_lenght"],
			N=dna["number_of_turns"],
			inner_diameter=dna["inner_diameter"],
			wire_diameter=dna["wire_diameter"],
			resistivity=dna["resistivity"]
		)
	else:
		raise NoDNAError(f"{dna['CoilType']} is not an implemented coil type.")

def power_source_from_DNA(dna: DNA) -> PowerSource:
	"""Create a power source from the DNA"""
	power_enum = PowerSourceEnum[dna["PowerType"]]

	if power_enum == PowerSourceEnum.ConstantCurrent:
		return ConstantCurrent(
			current=dna["current"]
		)
	elif power_enum == PowerSourceEnum.ConstantVoltage:
		return ConstantVoltage(
			voltage=dna["voltage"]
		)
	else:
		raise NoDNAError(f"{dna['PowerType']} is not an implemented power source.")

def projectile_from_DNA(dna: DNA) -> Projectile1D:
	"""Create a projectile from the DNA"""
	projectile_enum = ProjectileEnum[dna["ProjectileType"]]

	if projectile_enum == ProjectileEnum.Projectile1D:
		return Projectile1D(
			mass=dna["projectile_mass"],
			pos=dna["projectile_position"],
			vel=dna["projectile_velocity"]
		)
	elif projectile_enum == ProjectileEnum.MagneticProjectile:
		return MagneticProjectile(
			mass=dna["projectile_mass"],
			pos=dna["projectile_position"],
			vel=dna["projectile_velocity"],
			m=dna["magnetic_dipole_momnet"]
		)
	elif projectile_enum == ProjectileEnum.FerromageneticProjectile:
		return FerromageneticProjectile(
			mass=dna["projectile_mass"],
			pos=dna["projectile_position"],
			vel=dna["projectile_velocity"],
			mu=dna["???"]
		)
	else:
		raise NoDNAError(f"{dna['ProjectileType']} is not an implemented projectile.")


class FitnessFunction(ABC):
	"""A class that calculates the fitness of a population"""

	@abstractmethod
	def __call__(self, dna: DNA) -> float:
		"""Calculate the fitness of a DNA"""


class CoilFitness(FitnessFunction):
	"""Calculate the fitness of a coil"""

	def __init__(self, simulation_conf: SimulationConf):
		self.sim_conf = simulation_conf

	def __call__(self, dna: DNA) -> float:
		coil = coil_from_DNA(dna)
		power_source = power_source_from_DNA(dna)
		projectile = projectile_from_DNA(dna)

		simulation = CoilgunSimulation(
			coil=coil,
			power_source=power_source,
			projectile=projectile,
			conf=self.sim_conf
		)

		simulation_data = simulation.run()

		# Calculate the efficiency of the coil
		energy_gain = simulation_data.energy_gain(projectile.mass)
		energy_consumed = simulation_data.energy_consumption()

		if energy_consumed == 0:
			n = 0
		else:
			n = energy_gain / energy_consumed
		return n


class ODECoilFitness(FitnessFunction):
	"""Calculate the fitness of a coil modeled as an ODE"""

	def __init__(self, max_time: float, minimum_solver_steps: int=None):
		self.max_time = max_time
		self.minimum_solver_steps = minimum_solver_steps

	def __call__(self, dna: DNA) -> float:
		sim = CoilgunSimulationODE.from_DNA(dna)
		t, x, v, I, V = sim.run(self.max_time, self.minimum_solver_steps)

		v0, v1 = v[0], v[-1]
		V0, V1 = V[0], V[-1]

		score = calculate_efficiency(
			v0=v0,
			v1=v1,
			V0=V0,
			V1=V1,
			m=dna["projectile_mass"],
			C=dna["capacitance"]
		)

		max_Amp = np.max(I)
		amp_threshold = 600

		# Limit ampare to some extent
		if max_Amp > amp_threshold:
			score /= max_Amp / amp_threshold

		return score


class NoDNAError(Exception):
	"""Raise when there is no mathing DNA"""
	pass
