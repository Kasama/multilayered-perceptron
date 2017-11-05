"""
Class that holds a genetic algorithm for evolving a network.

Credit:
	A lot of those code was originally inspired by:
	http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from mlp import MLP

class Optimizer():
	"""Class that implements genetic algorithm for MLP optimization."""

	def __init__(self, nn_param_choices, retain=0.4,
				 random_select=0.1, mutate_chance=0.2):
		"""Create an optimizer.

		Args:
			nn_param_choices (dict): Possible network paremters
			retain (float): Percentage of population to retain after
				each generation
			random_select (float): Probability of a rejected network
				remaining in the population
			mutate_chance (float): Probability a network will be
				randomly mutated

		"""
		self.mutate_chance = mutate_chance
		self.random_select = random_select
		self.retain = retain
		self.nn_param_choices = nn_param_choices

	def create_population(self, count):
		"""Create a population of random networks.

		Args:
			count (int): Number of networks to generate, aka the
				size of the population

		Returns:
			(list): Population of network objects

		"""
		pop = []
		for _ in range(0, count):
			# Create a random network.
			network = MLP(self.nn_param_choices['input'], self.nn_param_choices['hidden'], self.nn_param_choices['output'])
			#network.create_random()

			# Add the network to our population.
			pop.append(network)
		return pop

	@staticmethod
	def fitness(network):
		"""Return the accuracy, which is our fitness function."""
		return 1-network.squared_err

	def grade(self, pop):
		"""Find average fitness for a population.

		Args:
			pop (list): The population of networks

		Returns:
			(float): The average accuracy of the population

		"""
		summed = reduce(add, (self.fitness(network) for network in pop))
		return summed / float((len(pop)))

	def breed(self, mother, father):
		"""Make two children as parts of their parents.

		Args:
			mother (dict): Network parameters
			father (dict): Network parameters

		Returns:
			(list): Two network objects

		"""
		children = []
		for _ in range(2):

			child = MLP(self.nn_param_choices['input'], self.nn_param_choices['hidden'], self.nn_param_choices['output'])

			# Loop through the parameters and pick params for the kid.
			# for param in self.nn_param_choices:
				# child[param] = random.choice(
					# [mother.network[param], father.network[param]]
				# )
			if mother.hidden_layer_weights.shape == father.hidden_layer_weights.shape:
				for h in range(self.nn_param_choices['hidden']):
					for o in range(self.nn_param_choices['output']):
						child.hidden_layer_weights[i] = random.choice(mother.hidden_layer_weights[i][h][o], father.hidden_layer_weights[i][h][o])

			children.append(child)

		return children

	def evolve(self, pop, threads):
		"""Evolve a population of networks.

		Args:
			pop (list): A list of network parameters

		Returns:
			(list): The evolved population of networks

		"""
		for t in threads:
			t.do_run = False
		
		# Get scores for each network.
		graded = [(self.fitness(network), network) for network in pop]

		# Sort on the scores.
		graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

		# Get the number we want to keep for the next gen.
		retain_length = int(len(graded)*self.retain)

		# The parents are every network we want to keep.
		parents = graded[:retain_length]

		# For those we aren't keeping, randomly keep some anyway.
		for individual in graded[retain_length:]:
			if self.random_select > random.random():
				parents.append(individual)

		# Randomly mutate some of the networks we're keeping.
		# for individual in parents:
			# if self.mutate_chance > random.random():
				# individual = self.mutate(individual)

		# Now find out how many spots we have left to fill.
		parents_length = len(parents)
		desired_length = len(pop) - parents_length
		children = []

		# Add children, which are bred from two remaining networks.
		while len(children) < desired_length:

			# Get a random mom and dad.
			male = random.randint(0, parents_length-1)
			female = random.randint(0, parents_length-1)

			# Assuming they aren't the same network...
			if male != female:
				male = parents[male]
				female = parents[female]

				# Breed them.
				babies = self.breed(male, female)

				# Add the children one at a time.
				for baby in babies:
					# Don't grow larger than desired length.
					if len(children) < desired_length:
						children.append(baby)

		parents.extend(children)

		return parents
