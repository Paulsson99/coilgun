import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

from GA.DNA import DNA
from visualise.coil import draw_coil
from visualise.simulation import draw_simulation, plot_ode_solution
from simulation.simulate import CoilgunSimulation
from utils.path import defaults_path
from .load_objects import get_coil, get_power_source, get_projectile, get_simulation_conf, parse_args


def show_coil(args):
	"""Command line interface for showing a coil"""

	# Create a coil
	coil = get_coil(args)

	# Create a figure
	fig, ax = plt.subplots()
	ax.set_aspect('equal')

	# Draw the coil
	draw_coil(coil, ax)
	plt.show()

def show_simulation(args):
	"""Command line interface for showing a simulation"""
	coil = get_coil(args)
	power_source = get_power_source(args)
	projectile = get_projectile(args)
	conf = get_simulation_conf(args)

	sim = CoilgunSimulation(coil, power_source, projectile, conf)

	animation = draw_simulation(sim)
	
	plt.show()

def show_ode_sim(args):
	"""Command line interface for showing an ode simulation"""
	dna = DNA.read_DNA(Path(args["DNA"]))

	plot_ode_solution(dna, args["max_time"], args["minimum_solver_steps"], output_file=args["outfile"])

def main():
	parser = ArgumentParser(
		description='Visualise a coilgun using matplotlib'
	)
	parser.add_argument(
		"--conf",
		default=f"{defaults_path() / 'conf_ode_template.yaml'}",
		type=str,
		help="Provide a configuration file for all the settings")

	subparsers = parser.add_subparsers(
		help='Differnet visualizations', 
		dest='prog', 
		required=True
	)

	# Visualise a coil
	coil_parser = subparsers.add_parser(
		'coil', 
		help='Visualise a coil'
	)
	coil_parser.add_argument(
		'-c', '--coil',
		type=str,
		help="Template file for the coil. If not provided a default is used"
	)

	# Visualise a simulation
	sim_parser = subparsers.add_parser(
		'simulation', 
		aliases=['sim'], 
		help='Visualise a simulation of a coilgun'
	)
	sim_parser.add_argument(
		'-c', '--coil',
		type=str,
		help="Template file for the coil. If not provided a default is used"
	)
	sim_parser.add_argument(
		'-s', '--power-source',
		type=str,
		help="Template file for the power source. If not provided a default is used"
	)
	sim_parser.add_argument(
		'-p', '--projectile',
		type=str,
		help="Template file for the projectile. If not provided a default is used"
	)

	# Visualise an ode simulation
	ode_parser = subparsers.add_parser(
		'ode',
		help="Show the simulation of an ODE model"
		)
	ode_parser.add_argument(
		'-d', '--DNA',
		type=str,
		help="Template file for the coil. If not provided a default is used"
	)
	ode_parser.add_argument(
		"--outfile", "-o", 
		type=str,
		default=None,
		help="Save the data to the file 'outfile'"
	)

	# Parse arguments and execute the right program
	args = parse_args(parser.parse_args())

	if args["prog"] == 'coil':
		show_coil(args)
	elif args["prog"] in ['simulation', 'sim']:
		show_simulation(args)
	elif args["prog"] == "ode":
		show_ode_sim(args)



if __name__ == '__main__':
	main()