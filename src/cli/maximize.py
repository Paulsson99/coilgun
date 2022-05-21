from argparse import ArgumentParser
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

from GA.fitness import ODECoilFitness
from GA.DNA import DNA, MutationRules

from utils.path import defaults_path, data_path
from .load_objects import read_DNA_from_template, get_simulation_conf, parse_args

def bounds_from_mutation_rules(rules: MutationRules):
	bounds = []
	for rule in rules.all_rules():
		bounds.append((rules.rule(rule).min, rules.rule(rule).max))
	return bounds

def min_func(dna: DNA, rules: MutationRules, fitness_func):
	variables = rules.all_rules()
	def func(x):
		for i, var in enumerate(variables):
			dna.DNA[var] = x[i]
		return -fitness_func(dna)
	return func

def p0(dna: DNA, rules: MutationRules):
	variables = rules.all_rules()
	return [dna[variable] for variable in variables]



def maximize_parser():
	"""Return a parser for the evolution program"""
	parser = ArgumentParser(
		description='Maximize efficiency of a coilgun using a traditional algorithm'
	)
	parser.add_argument(
		'-d', '--DNA',
		type=str,
		help="Template file for the base DNA. If not provided a default is used"
	)
	parser.add_argument(
		'-r', '--rules',
		type=str,
		help="Template file for the base DNA. If not provided a default is used"
	)
	parser.add_argument(
		'-c', '--conf',
		default=f"{defaults_path() / 'conf_ode_template.yaml'}",
		type=str,
		help="Template file for the simulation configuration. If not provided a default is used"
	)
	return parser

def maximize(args: dict):
	"""Evolve"""

	# Setup evoultion object
	default_DNA = read_DNA_from_template(args["DNA"])
	mutation_rules = MutationRules.read_rules(Path(args["rules"]))

	fitness_func = ODECoilFitness(
		max_time=args["max_time"],
		minimum_solver_steps=args["minimum_solver_steps"]
	)

	for i in range(10):
		default_DNA.randomize_DNA(mutation_rules)
		bounds = bounds_from_mutation_rules(mutation_rules)
		minimize_function = min_func(default_DNA, mutation_rules, fitness_func)
		inital_cond = p0(default_DNA, mutation_rules)

		res = minimize(minimize_function, inital_cond, bounds=bounds)

		print(fitness_func(default_DNA))

def main():
	parser = maximize_parser()

	# Parse arguments and execute the program
	args = parse_args(parser.parse_args())

	maximize(args)


if __name__ == '__main__':
	main()