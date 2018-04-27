#! /usr/bin/env python3

"""
Development Version: Python 3.5.1
Author: Aaron Coyner
Description: Perceptron Learning Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import datagen


class perceptron():
	def __init__(self, X, Y, max_iters=1000):
		self.X = X
		self.Y = Y
		self.weights = np.zeros(len(self.X[0]))
		self.max_iters = max_iters
		self.train()

	def train(self):
		correct = [False] * len(self.weights)

		for i in range(self.max_iters):
			correct = []
			for index in range(len(self.X)):
				self.y_hat = np.matmul(self.weights.transpose(), self.X[index])

				if self.y_hat < 0:
					self.y_hat = -1
				else:
					self.y_hat = 1

				if self.Y[index] != self.y_hat and self.y_hat == -1:
					self.weights += self.X[index]
					correct.append(False)
				elif 	self.Y[index] != self.y_hat and self.y_hat == 1:
					self.weights -= self.X[index]
					correct.append(False)
				else:
					correct.append(True)

			if False not in correct:
				print('Algorithm converged after ' + str(i) + ' iterations.')
				self.plot_train()
				break
			elif i == self.max_iters - 1:
				print('Algorithm did not converge in ' + str(self.max_iters) + ' iterations.')
				print('Data is either not linearly separable or requires more iterations.')
			else:
				pass


	def plot_train(self):
		self.bias = -(self.weights[0]) / self.weights[2]
		self.slope = -(self.weights[1] / self.weights[2])

		for index in range(len(self.X)):
			plt.plot(self.X[index, 1], self.X[index, 2], 'ro' if self.Y[index] == 1 else 'bo')

		self.x_vals = np.array([0, 1])
		self.y_vals = np.array(self.slope * self.x_vals + self.bias)

		plt.plot(self.x_vals, self.y_vals, '--')
		plt.show()




if __name__ == "__main__":
	data = datagen.data_generator(random=True, num_samples=4, print_data=True)
	model = perceptron(data.X, data.Y)
