import numpy as np


class data_generator():	
	def __init__(self, random=False, num_samples=3, print_data=False, num_dims=2):
		assert random == True or random == False
		assert print_data == True or print_data == False
		assert num_dims >= 2
		assert num_samples >= 2

		if random is True:
			self.num_samples = num_samples
			self.num_dims = num_dims			
			self.create_random()
		else:
			self.use_predefined()

		if print_data is True:
			self.print_dataset()


	def create_random(self):
		X_temp = np.random.rand(self.num_samples, self.num_dims)
		self.X = np.ones((self.num_samples, self.num_dims + 1))
		self.X[:, 1:] = X_temp
		self.Y = np.random.randint(2, size=self.num_samples)
		
		while 1 not in self.Y or 0 not in self.Y:
			self.Y = np.random.randint(2, size=self.num_samples)

		self.Y[self.Y == 0] = -1


	def use_predefined(self):
		X_temp = np.array([[-1, -1], [2, 0], [2, 1], [0, 1], [0.5, 1.5], [3.5, 2.5], [3, 4], [4, 2], [5.5, 3]])
		X_temp /= 5.5
		self.X = np.ones((len(X_temp), len(X_temp[0]) + 1))
		self.X[:, 1:] = X_temp
		self.Y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1])


	def print_dataset(self):
		print('X: \n', self.X)
		print('Y: \n', self.Y)
