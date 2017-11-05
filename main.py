"""Entry point to evolving the neural network. Start here."""
import logging
import numpy as np
import pickle
from optimizer import Optimizer
from tqdm import tqdm
from threading import Thread


# Setup logging.
logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%m/%d/%Y %I:%M:%S %p',
	level=logging.DEBUG,
	filename='log.txt'
)

def train_networks(networks, X, Y):
	"""Train each network.

	Args:
		networks (list): Current population of networks
		dataset (str): Dataset to use for training/evaluating
	"""
	pbar = tqdm(total=len(networks))
	threads = []
	for network in networks:
		#network.learn(X,Y)
		t = Thread(target=network.learn, args=(X,Y))
		threads.append(t)
		t.start()
		pbar.update(1)
	pbar.close()
	return threads

def get_average_accuracy(networks):
	"""Get the average accuracy for a group of networks.

	Args:
		networks (list): List of networks

	Returns:
		float: The average accuracy of a population of networks.

	"""
	total_accuracy = 0
	for network in networks:
		total_accuracy += 1 - network.squared_err
	return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, X, Y):
	"""Generate a network with the genetic algorithm.

	Args:
		generations (int): Number of times to evole the population
		population (int): Number of networks in each generation
		nn_param_choices (dict): Parameter choices for networks
		dataset (str): Dataset to use for training/evaluating

	"""
	optimizer = Optimizer(nn_param_choices)
	networks = optimizer.create_population(population)

	# Evolve the generation.
	for i in range(generations):
		logging.info("***Doing generation %d of %d***" %
					 (i + 1, generations))

		# Train and get accuracy for networks.
		threads = train_networks(networks, X, Y)

		# Get the average accuracy for this generation.
		average_accuracy = get_average_accuracy(networks)

		# Print out the average accuracy each generation.
		logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
		logging.info('-'*80)

		# Evolve, except on the last iteration.
		if i != generations - 1:
			# Do the evolution.
			networks = optimizer.evolve(networks, threads)

	# Sort our final population.
	networks = sorted(networks, key=lambda x: 1-x.squared_err, reverse=True)

	# Print out the top 5 networks.
	print_networks(networks[:5])

def print_networks(networks):
	"""Print a list of networks.

	Args:
		networks (list): The population of networks

	"""
	logging.info('-'*80)
	for network in networks:
		print(network)

def main():
	"""Evolve a network."""
	generations = 10  # Number of times to evole the population.
	population = 20  # Number of networks in each generation.
	nn_param_choices = {
		'input': 22,
		'hidden': 36,
		'output': 3,
	}

	logging.info("***Evolving %d generations with population %d***" % (generations, population))
	observations = pickle.load(open('C:/Users/Nexor/Desktop/SH.STATES', 'rb'))
	results = pickle.load(open('C:/Users/Nexor/Desktop/SH.RESULTS', 'rb'))
	X = np.array([np.array(observation, dtype=np.float) for observation in observations])
	Y = np.array([np.array(result, dtype=np.float) for result in results])
	generate(generations, population, nn_param_choices, X, Y)

if __name__ == '__main__':
	main()
