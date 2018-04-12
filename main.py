import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math
import sys

# takes in a matrix representation of the edges of an image
# edges are marked with 1s, 0s everywhere else
# returns a list of the pixels where the edges are
def preprocessing():
	# Il = sio.loadmat('figure2-1.mat')
	Il = sio.loadmat('edges.mat')
	I = Il['bw05']
	# print(I)

	# plt.matshow(I)
	plt.imshow(I, cmap='Greys')
	# plt.show()

	I = np.array(I)

	# turn everything into points
	points = []
	it = np.nditer(I, flags=['multi_index'])
	while not it.finished:
		if it[0].item(0) is not 0:
			points.append(Point(it.multi_index[0],it.multi_index[1]))
		it.iternext()
	return points

# represents a pixel on the edge
class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __str__(self):
		return "("+str(self.x)+" ,"+str(self.y)+")"
	__repr__ = __str__

# class for assigning the points to the lines
class P1:
	@staticmethod
	def assign_points_to_lines(points, lines, buckets):
		for p in points:
			buckets[P1.closest_distance(p, lines)].append(p)
	@staticmethod
	def closest_distance(p, lines):
		min_dist = sys.maxsize
		min_ind = 0
		# print("\nnew")
		for i in range(len(lines)):
			dis = P1.distance(p, lines[i])
			# print(dis)
			if dis < min_dist:
				# print("decrease")
				min_dist = dis
				min_ind = i
		# print("this is always 2?? ", i)
		return min_ind
	@staticmethod
	def distance(p, line):
		x = math.fabs(p.x*line[0] + p.y*line[1] + line[2]) / math.sqrt(line[0]**2 + line[1]**2)
		# print("distance is", x, "   ", p.x, line[0], p.y, line[1], line[2])
		return x 

# class for changing the lines to fit the points
class P2:
	@staticmethod
	def update_lines(lines, buckets):
		# for line,bucket in lines,buckets:
		for i in range(3):
			line = lines[i]
			bucket = buckets[i]
			if len(bucket) is 0:
				continue
			x_mean = P2.mean(bucket,0)
			y_mean = P2.mean(bucket,1)
			A = P2.mean_two(bucket, 0,0) - x_mean**2
			B = P2.mean_two(bucket, 0,1) - x_mean*y_mean
			C = P2.mean_two(bucket, 1,1) - y_mean**2
			w,v = LA.eig(np.array([[A,B],[B,C]]))
			if w[0]<=0 or w[1]<=0:
				print("SOMETHING WRONG\n", A, B, C)
				import sys
				sys.exit(0)
			# print("Eigenvector:", v)
			if w[0]<w[1]:
				# print(str(v[:,0][0]) + "  "+str(v[:,0][1]))
				line[0] = v[:,0][0]
				line[1] = v[:,0][1]
				line[2] = - line[0]*x_mean - line[1]*y_mean
			else:
				line[0] = v[:,1][0]
				line[1] = v[:,1][1]
				line[2] = - line[0]*x_mean - line[1]*y_mean	
	
	@staticmethod
	def mean(bucket, index):
		ans = 0.0
		length = 0
		for point in bucket:
			p = (point.x,point.y)
			ans += p[index]
			length += 1
		return ans/length
		
	def mean_two(bucket, i1, i2):
		ans = 0.0
		length = 0
		for point in bucket:
			p = (point.x,point.y)
			ans += p[i1]*p[i2]
			length += 1
		return ans/length
		
def delete_buckets(buckets):
	del buckets[0][:]
	del buckets[1][:]
	del buckets[2][:]



	
points = preprocessing()
# print(points)

# initialize
lines = [[0 for x in range(3)] for y in range(3)] 

# these lines arent really necessary, but just helps show that the equations of the lines are indeed moving
for i in range(3):
	lines[i][0] = np.random.random()
	lines[i][1] = np.random.random()
	lines[i][2] = np.random.random()
	
buckets = [[] for i in range(3)]

# randomly assign points
for p in points:
	x = np.random.randint(3)
	buckets[x].append(p)

prev = [0]*3
converge_times = 20
iterations = 0


print("\nInitial Lines Configuration")
print(lines[0])
print(lines[1])
print(lines[2])


# initialize the 3 lines
while True:
	for i in range(100):
		# update the lines to the points
		P2.update_lines(lines, buckets)
		# check if the lines didnt change that much / the lines the points are assigned to didnt change at all	
		# delete buckets
		delete_buckets(buckets)
		# assign the points to the lines
		P1.assign_points_to_lines(points, lines, buckets)
		
		if prev[0] is len(buckets[0]) and prev[1] is len(buckets[1]) and prev[2] is len(buckets[2]):
			# print("Converge???")
			converge_times-=1
			if converge_times is 0:
				print("FINAL CONVERGE")
				break
		# print("length of buckets", len(buckets[0]), len(buckets[1]), len(buckets[2]))
		prev[0] = len(buckets[0])
		prev[1] = len(buckets[1])
		prev[2] = len(buckets[2])
		iterations += 1
	
	break

print("\nTook", iterations, "Iterations")
print(lines[0])
print(lines[1])
print(lines[2])




# # for graphing 
# def graph(function):
# 	x = np.array(range(0,80))  
# 	y = eval(function)
# 	plt.plot(x, y) 

# graph('-3.75/0.45 + (0.89/0.45)*x')
# graph('3.735/0.894 + (0.45/0.894)*x')
# graph('83.11/0.705 + (-	0.71/0.705)*x')


# graph('-3.75/0.45 + (0.88/0.45)*x')
# graph('6/0.91 + (0.41/0.91)*x')
# graph('87/0.61 - (0.73/0.61)*x')


# plt.show()