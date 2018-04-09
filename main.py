import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	



# Il = sio.loadmat('figure2-1.mat')
Il = sio.loadmat('edges.mat')
I = Il['bw05']
print(I)

# plt.matshow(I)
plt.imshow(I, cmap='Greys')
plt.show()

# turn everything into points
points = []
for ix,iy in np.ndindex(a.shape):
	points.append(Point(ix,iy))


lines = [[0 for x in range(3)] for y in range(3)] 
for i in range(3):
	t = 2*Math.pi*np.random()
	a = Math.cos(t)
	b = Math.sin(t)
	r = (177+114)*np.random() - 114
	d = Math.sqrt(a**2 + b**2)
	lines[i][0] = a / d
	lines[i][1] = b / d
	lines[i][2] = r / d
	
buckets = []
buckets.append([])
buckets.append([])
buckets.append([])

# initialize the 3 lines
# while True:
	# assign the points to the lines
	P1.assign_points_to_lines(points, lines, buckets)
	# update the lines to the points
	
	# check if the lines didnt change that much / the lines the points are assigned to didnt change at all	
	
	# delete buckets
	delete_buckets(buckets)
class P1:
	@staticmethod
	def assign_points_to_lines(self, points, lines, buckets):
		for p in points:
			buckets[closest_distance(p, lines)].append(p)
	@staticmethod
	def closest_distance(self, p, lines):
		min_dist = sys.maxsize
		min_ind = 0
		for i in range(len(lines)):
			dis = distance(p, lines[i])
			if dis < min_dist:
				min_dist = dis
				min_ind = i
		return i
	@staticmethod
	def distance(self, p, line):
		return Math.abs(p[0]*line[0] + p[1]*line[1] + line[2])
	
def delete_buckets(buckets):
	del buckets[0][:]
	del buckets[1][:]
	del buckets[2][:]