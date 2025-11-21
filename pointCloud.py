import laspy
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree
from scipy.interpolate import BSpline, make_interp_spline, splprep, PPoly, splev
import time
import math
import utm
import sys
import heapq
import os
import csv
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax


def convertToLatLon(inputFile,outputFile): #Converts all UTM las points to Lat/Lon las points
	las = laspy.read(inputFile)
	xUtm = np.array(las.x)
	yUtm = np.array(las.y)
	zAlt = np.array(las.z) - np.min(las.z)
	lat_lon = utm.to_latlon(xUtm,yUtm, 18, northern = True)
		
	xLon = np.array(lat_lon[1]) / 0.00001
	yLat = np.array(lat_lon[0]) / 0.00001
		
	new_file = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
	new_file.x = xLon
	new_file.y = yLat
	new_file.z = zAlt

	new_file.write(outputFile)

	las2 = laspy.read(outputFile)
	coordsBefore = np.vstack((xLon,yLat,las.z)).transpose()
	coordsAfter = np.vstack((las2.x,las2.y,las2.z)).transpose()
	print("Testing data loss:")
	print("First coords: ",coordsBefore[0]," Second coords: ",coordsAfter[0])

def makeEmptySpaceCloud(files, fileName, directory = "", scale = 1.0, distance = 1.0):
	print("Making ESC")
	if distance < 0:
		distance = 1.0
	if directory != "" and not os.path.isdir(directory):
		os.mkdir(directory)
	print("Established distance and directory")
	
	# Read header
	las0 = laspy.read(files[0])
	base_header = las0.header
	print("Header cloned")
	
	# Buffers
	points_buffer = []
	buffer_limit = 50_000_000  # adjustable
	
	fullStart = time.time()
	found_points = 0	

	# Output file path
	out_path = directory + fileName
	first_chunk = True


	def flush_buffer():
		nonlocal first_chunk
		if len(points_buffer) == 0:
			return
	
		arr = np.array(points_buffer, dtype=float)
	
		# Build correct-size header
		hdr = base_header
		hdr.point_count = len(arr)
	
		las = laspy.LasData(hdr)
		las.x = arr[:,0]
		las.y = arr[:,1]
		las.z = arr[:,2]

		mode = "w" if first_chunk else "a"
		with laspy.open(out_path, mode=mode, header=hdr) as writer:
			writer.write_points(las.points)

		points_buffer.clear()
		first_chunk = False


	def load_border_slice(filepath, xMin, xMax, yMin, yMax, zMin, zMax, distance):
		las = laspy.read(filepath)
		mask = (
			(las.x >= xMin - distance) & (las.x <= xMax + distance) &
			(las.y >= yMin - distance) & (las.y <= yMax + distance) &
			(las.z >= zMin - distance) & (las.z <= zMax + distance)
		)
		if not np.any(mask):
			return None
		return np.vstack((las.x[mask], las.y[mask], las.z[mask])).T


	print("Starting files")
	for fNum in range(len(files)):
		print(f"Working on file {files[fNum]}")
	
		las = laspy.read(files[fNum])

		# Bounding box
		xMin = np.floor(np.min(las.x))
		xMax = np.ceil(np.max(las.x))
		yMin = np.floor(np.min(las.y))
		yMax = np.ceil(np.max(las.y))
		zMin = np.floor(np.min(las.z))
		zMax = np.ceil(np.max(las.z))
	
		# Build KD-tree base tile
		coords_main = np.vstack((las.x, las.y, las.z)).T
	
		next_slice = None
		prev_slice = None

		if fNum + 1 < len(files):
			next_slice = load_border_slice(files[fNum+1], xMin, xMax, yMin, yMax, zMin, zMax, 1.5*distance)
		if next_slice is not None:
			coords_main = np.concatenate((coords_main, next_slice), axis=0)
	
		if fNum - 1 >= 0:
			prev_slice = load_border_slice(files[fNum-1], xMin, xMax, yMin, yMax, zMin, zMax, 1.5*distance)
		if prev_slice is not None:
			coords_main = np.concatenate((coords_main, prev_slice), axis=0)
	
		tree_main = cKDTree(coords_main)
		del coords_main

		start = time.time()
	
		for x in np.arange(xMin, xMax, scale):
			for y in np.arange(yMin, yMax, scale):
				for z in np.arange(zMin, zMax, scale):
					d, _ = tree_main.query([x, y, z], k=1, distance_upper_bound=distance)
					if math.isinf(d):
						found_points += 1
						points_buffer.append([x, y, z])
	
						# flush when full
						if len(points_buffer) >= buffer_limit:
							flush_buffer()
	
		end = time.time()
		print("Time taken to search: ", end - start)

	# Final write
	flush_buffer()

	print(f"Found {found_points} possible points")
	print("Writing to " + out_path)
	print("Total time: ", time.time() - fullStart)


def aStarSearch(start, end, coords, threshold = 3, k = 0.5):
	assert np.any(np.all(np.array(start) == coords, axis=1))
	assert np.any(np.all(np.array(end) == coords, axis=1))
	
	#Helper functions and node class
	def manhattanDistance(p1,p2):
		return abs((p1[0]-p2[0]))+abs((p1[1]-p2[1]))+abs((p1[2]-p2[2]))
		
	def euclideanDistance(p1,p2):
		return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]))

	def pointDistance(P,A,B):
		P, A, B = np.array(P), np.array(A), np.array(B)
		AB = B-A
		AP = P-A
		t = np.dot(AP, AB) / np.dot(AB, AB)
		t = max(0, min(1, t))
		closest_point = A + t * AB
		distance = np.linalg.norm(P-closest_point)
		return distance
	
	class searchPt:
		parent = None
		g = None
		h = None
		f = None
		def __init__(self, crds, parent):
			self.crds = crds
			self.parent = parent
		def __lt__(self, other):
			if self.crds[0] != other.crds[0]:
				return self.crds[0] < other.crds[0]
			if self.crds[1] != other.crds[1]:
				return self.crds[1] < other.crds[1]
			if self.crds[2] != other.crds[2]:
				return self.crds[2] < other.crds[2]
	
	
	print("Distance from start to end: ", euclideanDistance(start,end))
	tree = cKDTree(coords)
	path = [start]
	startPt = searchPt(start, None)
	endPt = searchPt(end, None)
	startPt.g = 0.0
	startPt.h = 0.0
	startPt.f = 0.0
	endPt.g = 0.0
	endPt.h = 0.0
	endPt.f = 0.0
	unsearched = []
	heapq.heappush(unsearched, (0.0, startPt))
	unsearchedEnd = []
	heapq.heappush(unsearchedEnd, (0.0, endPt))
	visited = {tuple(start):startPt}
	visitedEnd = {tuple(end):endPt}
	endPt = None
	startPt = None
	
	if np.array_equal(start,end):
		return path
	dist1 = 0
	dist2 = 0
	finished = False
	while len(unsearched) > 0 and len(unsearchedEnd) > 0 and not finished:
		#Search through points from start
		curr = (heapq.heappop(unsearched))[1]
		dist1 = euclideanDistance(curr.crds,end)
		distances, indices = tree.query(curr.crds, k=27, distance_upper_bound = 2.0)
		for i in indices:
			if i >= coords.shape[0] or np.array_equal(curr.crds,coords[i]):
				continue
			newPt = searchPt(coords[i], curr)
			if np.array_equal(newPt.crds,end) or visitedEnd.get(tuple(newPt.crds),None) != None:
				unsearched = []
				unsearchedEnd = []
				if visitedEnd.get(tuple(newPt.crds),None) != None:
					startPt = newPt
					endPt = visitedEnd.get(tuple(newPt.crds),None)
				elif np.array_equal(newPt.crds,end):
					endPt = newPt
					startPt = None
				finished = True
				break
			newPt.g = curr.g + euclideanDistance(curr.crds,newPt.crds)
			newPt.h = manhattanDistance(newPt.crds,end)
			distance = pointDistance(coords[i], start, end)
			penalty = k * (distance - threshold)**2 if (distance > threshold) else 0
			newPt.f = newPt.g + newPt.h + penalty
			check = visited.get(tuple(newPt.crds),None)
			if check == None or newPt.f < check.f:
				heapq.heappush(unsearched, (newPt.f, newPt))
				visited[tuple(newPt.crds)] = newPt
				euDist = euclideanDistance(newPt.crds,end)
		if finished:
			continue
		#Search through points from end
		curr = (heapq.heappop(unsearchedEnd))[1]
		dist2 = euclideanDistance(curr.crds,start)
		distances, indices = tree.query(curr.crds, k=27, distance_upper_bound = 2.0)
		for i in indices:
			if i >= coords.shape[0] or np.array_equal(curr.crds,coords[i]):
				continue
			newPt = searchPt(coords[i], curr)
			if np.array_equal(newPt.crds,start) or visited.get(tuple(newPt.crds),None) != None:
				unsearched = []
				unsearchedEnd = []
				if visited.get(tuple(newPt.crds),None) != None:
					startPt = visited.get(tuple(newPt.crds),None)
					endPt = newPt
				elif np.array_equal(newPt.crds,start):
					endPt = None
					startPt = newPt
				finished = True
				break
			newPt.g = curr.g + euclideanDistance(curr.crds,newPt.crds)
			newPt.h = manhattanDistance(newPt.crds,start)
			penalty = k * (distance - threshold)**2 if (distance > threshold) else 0
			newPt.f = newPt.g + newPt.h + penalty
			check = visitedEnd.get(tuple(newPt.crds),None)
			if check == None or newPt.f < check.f:
				heapq.heappush(unsearchedEnd, (newPt.f, newPt))
				visitedEnd[tuple(newPt.crds)] = newPt
				euDist = euclideanDistance(newPt.crds,end)
				'''exploredPoints[0].append(newPt.crds[0])
				exploredPoints[1].append(newPt.crds[1])
				exploredPoints[2].append(newPt.crds[2])'''
		if finished:
			continue
		print(dist1,dist2)
	if endPt == None and startPt == None:
		print("No path found")
		return []
	
	
	#Find and return path
	path = []
	if startPt != None:
		pt = startPt
		while pt.parent != None:
			path.append(pt.crds)
			pt = pt.parent
		path.append(pt.crds)
		path.reverse()
	if endPt != None:
		pt = endPt
		while pt.parent != None:
			path.append(pt.crds)
			pt = pt.parent
		path.append(pt.crds)
	remove_duplicates = []
	for x in range(0,len(path)-1):
		if np.all(path[x] == path[x+1]):
			remove_duplicates.append(x)
	while remove_duplicates:
		del path[ remove_duplicates.pop() ]
	return path
	
def lasToCsv(infile,outfile):
	las = laspy.read(infile)
	x = np.array(las.x)
	y = np.array(las.y)
	z = np.array(las.z)
	data = {"x": x,
		"y": y,
		"z": z}
	size = range(0,x.size)
	df = pd.DataFrame(data, index = size)
	df.to_csv(outfile,index=False,header=False)
				
def flightToPathDistance(estimateCoords,path):
    las = laspy.read(path)
    coords = np.vstack((las.x,las.y,las.z)).transpose()
    tree = cKDTree(coords)
    distances = []
    for point in estimateCoords:
        neighbors_distance, neighbors_indices = tree.query(point, k = 1, workers = -1)
        distances.append(neighbors_distance)
    return distances
    
def trajectoryToLas(infile,outfile,header): #For visualizing trajectories made with genTrajectory
	class polynomial:
		p = None
		
		def __init__(self, p):
			assert(len(p) == 8)
			self.p = p
		
		# evaluate a polynomial using horner's rule
		def eval(self, t):
			assert t >= 0
			x = 0.0
			for i in range(0, len(self.p)):
				x = x * t + self.p[len(self.p) - 1 - i]
			return x

	output = laspy.create(point_format=header.point_format, file_version=header.version)
	csvReader = csv.reader(open(infile,'r'),delimiter=',')
	next(csvReader)
	traj = np.array(list(csvReader)).astype("float")
	x = []
	y = []
	z = []
	for segment in traj:
		duration = segment[0]
		xSegment = polynomial(segment[1:9])
		ySegment = polynomial(segment[9:17])
		zSegment = polynomial(segment[17:25])
		ts = np.arange(0.0, duration, 0.001)
	for ms in ts:
		tempX = xSegment.eval(ms)
		tempY = ySegment.eval(ms)
		tempZ = zSegment.eval(ms)
		
		x.append(tempX)
		y.append(tempY)
		z.append(tempZ)
	try:
		output.x = x
	except OverflowError:
		print("X values do not fit after scale and offset. Max value: ", max(x))
		return
	try:
		output.y = y
	except OverflowError:
		print("Y values do not fit after scale and offset. Max value: ", max(y))
		return
	try:
		output.z = z
	except OverflowError:
		print("Z values do not fit after scale and offset. Max value: ", max(z))
		return
	output.write(outfile)
    
def pointDistances(infile):
	if infile[-3:] == "las":
		las = laspy.read(infile)
		points = np.vstack((las.x,las.y,las.z)).transpose()
	elif infile[-3:] == "csv":
		points = np.array(list(csv.reader(open(infile,'r'),delimiter=','))).astype("float")
	else:
		print("Input needs to be las or csv")
		return []
	distances = []
	
	for x in range(len(points)-1):
		distances.append( math.sqrt( (points[x][0]-points[x+1][0])**2 + (points[x][1]-points[x+1][1])**2 + (points[x][2]-points[x+1][2])**2 ) )
	
	return distances
	
def extendWaypoints(infile, outfile, maxDistance=0.5):
	if infile[-3:] == "las":
		las = laspy.read(infile)
		points = np.vstack((las.x,las.y,las.z)).transpose()
	elif infile[-3:] == "csv":
		points = np.array(list(csv.reader(open(infile,'r'),delimiter=','))).astype("float")
	else:
		print("Input needs to be las or csv")
		return
	distances = pointDistances(infile)
	while min(distances) == 0:
		minIndex = np.argmin(distances)
		distances.pop(minIndex)
		points = np.delete(points,minIndex, axis=0)
	addPoints = []
	if maxDistance == 0:
		maxDistance = 0.5
	for index in range(0,len(points)-1):
		numTimes = int(distances[index]/maxDistance)
		if numTimes != 0:
			basechange = 1/numTimes
		for nPIndex in range(1,numTimes):
			totalChange = nPIndex*basechange
			newPoint = [ points[index][0] + totalChange*(points[index+1][0]-points[index][0]), points[index][1] + totalChange*(points[index+1][1]-points[index][1]), points[index][2] + totalChange*(points[index+1][2]-points[index][2])]
			addPoints.append([index+nPIndex,newPoint])
	for i in range(len(addPoints)-1,-1,-1):
		points = np.insert(points, addPoints[i][0], addPoints[i][1], axis=0)
	points = points.transpose()
	if infile[-3:] == "las":
		output = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
		output.x = points[0]
		output.y = points[1]
		output.z = points[2]
		output.write(outfile)
	else:
		data = {"x": points[0],
			"y": points[1],
			"z": points[2]}
		size = range(0,points.size)
		df = pd.DataFrame(data, index = size)
		df.to_csv(outfile,index=False,header=False)
	
def visualizeLinesToLas(lines,outfile,header):
	lines = np.array(lines)
	assert(lines.shape[1]==2)
	las = laspy.create(point_format=header.point_format, file_version=header.version)
	x = []
	y = []
	z = []
	for line in lines:
		difference = np.array(line[1]) - np.array(line[0])
		percentages = np.arange(0, 1, 0.01)
		for percent in percentages:
			tempX = line[0][0] + percent*difference[0]
			tempY = line[0][1] + percent*difference[1]
			tempZ = line[0][2] + percent*difference[2]
			x.append(tempX)
			y.append(tempY)
			z.append(tempZ)
	las.x = x
	las.y = y
	las.z = z
	las.write(outfile)
	
def stateEstimationToLas(infile,outfile,header,x_offset=0,y_offset=0,z_offset=0):
	f = open(infile,'r')
	pattern = "[a-df-zA-Z{}':,]"
	points = []
	for line in f:
		temp = re.sub(pattern,'',line)
		temp = temp.replace("ee. ",'')
		temp = temp.split(" ")
		points.append([float(temp[0]),float(temp[1]),float(temp[2])])
	points = np.array(points).T
	las = laspy.create(point_format=header.point_format, file_version=header.version)
	las.x = points[0] + x_offset
	las.y = points[1] + y_offset
	las.z = points[2] + z_offset
	las.write(outfile)
		
		
def genTrajectory(infile, outfile, max_velocity = 1.0, max_acceleration = 1.0, drift_max = 1.0, verbose=False):
	#Validate input file and get points
	if infile[-3:] == "csv":
		points = np.array(list(csv.reader(open(infile,'r'),delimiter=','))).astype("float")
	else:
		print("Input should be a csv file of x,y,z coordinates")
		return
	
	#Compute longest straight lines to minimize waypoints
	lines = []
	start = points[0]
	v1 = np.array(points[1]) - np.array(points[0])
	for index in range(2,len(points)):
		v2 = np.array(points[index]) - np.array(points[index-1])
		crossProduct = np.cross(v1,v2)
		if np.all(crossProduct == 0):
			continue
		else:
			end = points[index-1]
			lines.append([start,end])
			start = points[index-1]
			v1 = v2
	lines.append([start,points[-1]])
	
	#Get waypoints
	waypoints = []
	for x in range(0,len(lines)):
		waypoints.append(lines[x][0])
	waypoints.append(lines[-1][1])
	waypoints = np.array(waypoints)
	
	#Start the optimization loop to make sure drift doesn't go above drift max while going as fast as possible
	l, r = [(1, (0, 0, 0))], [(1, (0, 0, 0))] #Bounds, sets first derivative (velocity) to 0 at start and end
	x_offset = waypoints[0][0]
	y_offset = waypoints[0][1]
	z_offset = waypoints[0][2]
	waypoints[:,0] -= x_offset
	waypoints[:,1] -= y_offset
	waypoints[:,2] -= z_offset
	
	init_waypoints = waypoints
	time_segments = (np.linalg.norm(np.diff(waypoints, axis=0), axis=1)/max_velocity)
	test_params = np.append([0],np.cumsum(time_segments)) #Initial guess is everything can be done at max speed (obviously impossible)
	test_params_size = len(test_params)
	
	def magnitude(arr):
		return np.linalg.norm(arr,axis=1)
		
	while True: #First, optimize time within the max vel/acc constraints
		num_changed = 0
		try:
			clamped_spline = make_interp_spline(test_params, init_waypoints, bc_type=(l, r))
		except ValueError as e:
			print(e)
			print("If the ValueError is referring to the x parameter, that means that the times were not strictly increasing. Should be impossible, however it could be solved by making the init_params longer or reducing max_vel parameter.")
			return
		time_segments = np.diff(test_params,axis=0)
		for x in range(1,test_params_size):
			test_linspace = np.linspace(test_params[x-1],test_params[x],max(int(time_segments[x-1]*100),100))
			max_vel_test = np.max(magnitude(clamped_spline.derivative(1)(test_linspace)))
			max_acc_test = np.max(magnitude(clamped_spline.derivative(2)(test_linspace)))
			if max_vel_test > max_velocity or max_acc_test > max_acceleration:
				time_segments[x-1] *= 1.1
				num_changed += 1
		test_params = np.append([0],np.cumsum(time_segments))
		if num_changed == 0:
			break

	def distance(arr,start,end):
		line = end - start
		return np.linalg.norm(np.cross(line, arr - start), axis=1) / np.linalg.norm(line)
		
	def closest_point(start, end, point): #For finding the closest point on a line from a point
		if np.all(start == end): raise Exception("Start cannot be the same as end")
		line = end - start
		t = ( line @ ( start - point ) ) / ( line @ line )
		new_point = start - ( t * line )
		percent = np.linalg.norm( new_point - start ) / np.linalg.norm( line )
		if percent > 1: new_point = np.copy(end)
		if percent < 0: new_point = np.copy(start)
		return new_point, percent
			
	
	while True: #Second, optimize for max drift
		test_params_size = test_params.shape[0]
		try:
			clamped_spline = make_interp_spline(test_params, waypoints, bc_type=(l, r))
		except ValueError as e:
			print(e)
			print("If the ValueError is referring to the x parameter, that means that the times were not strictly increasing. Should be impossible, however it could be solved by making the init_params longer or reducing max_vel parameter.")
			print(test_params)
			return
		time_segments = np.diff(test_params,axis=0)
		add_points = []
		show_prob = []
		for x in range(1,test_params_size): #Find the farthest point away from the line between consecutive waypoints, if it's more than dift max then add it to the list to change waypoints
			test_linspace = np.linspace(test_params[x-1],test_params[x],max(int(time_segments[x-1]*100),100))
			test_points = clamped_spline(test_linspace)
			distances = distance(test_points,waypoints[x-1],waypoints[x])
			max_distance_ind = np.argmax(distances)
			if distances[max_distance_ind] > drift_max:
				add_points.append((x,test_points[max_distance_ind]))
				show_prob.append(test_points[max_distance_ind])
		
		if not add_points:
			break	
			
			
		#For each point to add, find the closest spot on the line between waypoints and add that, as well as break up the time segments into new even chunks
		new_points = len(add_points)
		new_size = waypoints.shape[0]+new_points
		new_waypoints = np.zeros((new_size,3))
		new_time_segments = np.zeros((new_size-1,))
		new_index = 0
		waypoint_index = 0
		add_points_index = 0
		while new_index < new_size: #Linear time resizing
			if add_points_index < new_points and waypoint_index == add_points[add_points_index][0]:
				new_point, percentage = closest_point(waypoints[waypoint_index - 1], waypoints[waypoint_index], add_points[add_points_index][1])
				if np.linalg.norm( new_point - waypoints[waypoint_index] ) < 0.01 or np.linalg.norm( new_point - waypoints[waypoint_index - 1] ) < 0.01:
					new_waypoints[new_index] = waypoints[waypoint_index]
					new_time_segments[new_index-1] = time_segments[waypoint_index - 1] * 1.1
					waypoint_index += 1
					new_size -= 1
				else:
					new_waypoints[new_index] = new_point
					new_time = percentage * time_segments[waypoint_index - 1]
					new_time_segments[new_index - 1] = new_time
					time_segments[waypoint_index - 1] = time_segments[waypoint_index - 1] - new_time
				add_points_index += 1
			else:
				new_waypoints[new_index] = waypoints[waypoint_index]
				if waypoint_index > 0: new_time_segments[new_index - 1] = time_segments[waypoint_index - 1]
				waypoint_index += 1
			new_index += 1
			
		if new_size < waypoints.shape[0] + new_points:
			new_waypoints = new_waypoints[:new_size]
			new_time_segments = new_time_segments[:new_size-1]
		waypoints = new_waypoints
		test_params = np.append([0], np.cumsum(new_time_segments))
		
	time_segments = np.diff(test_params,axis=0)
	while True: #Third, reoptimize time within the max vel/acc constraints
		num_changed = 0
		try:
			clamped_spline = make_interp_spline(test_params, waypoints, bc_type=(l, r))
		except ValueError as e:
			print(e)
			print("If the ValueError is referring to the x parameter, that means that the times were not strictly increasing. Should be impossible, however it could be solved by making the init_params longer or reducing max_vel parameter.")
			return
		time_segments = np.diff(test_params,axis=0)
		for x in range(1,test_params_size):
			test_linspace = np.linspace(test_params[x-1],test_params[x],max(int(time_segments[x-1]*100),100))
			max_vel_test = np.max(magnitude(clamped_spline.derivative(1)(test_linspace)))
			max_acc_test = np.max(magnitude(clamped_spline.derivative(2)(test_linspace)))
			if max_vel_test > max_velocity or max_acc_test > max_acceleration:
				time_segments[x-1] *= 1.1
				num_changed += 1
		test_params = np.append([0],np.cumsum(time_segments))
		if num_changed == 0:
			break


	print("Total time: ", test_params[-1], "s, ", test_params[-1]/60,"m")	


	if verbose: #Visualize trajectory and velocity/acceleration/jerk
		test_linspace = np.linspace(test_params[0],test_params[-1],int(test_params[-1]*100))
		newpnts = clamped_spline(test_linspace)
		xnew = newpnts[:,0]
		ynew = newpnts[:,1]
		znew = newpnts[:,2]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(xnew+x_offset, ynew+y_offset, znew+z_offset, label='first iteration')
		ax.plot(points[:,0], points[:,1], points[:,2], color='green', label='given')
		plt.show()
		vel_spline = clamped_spline.derivative(1)
		x_vel, y_vel, z_vel = vel_spline(test_linspace).T
		acc_spline = vel_spline.derivative(1)
		x_acc, y_acc, z_acc = acc_spline(test_linspace).T
		jerk_spline = acc_spline.derivative(1)
		x_jerk, y_jerk, z_jerk = jerk_spline(test_linspace).T
		
		def magnitude(x,y,z):
			return np.sqrt(x**2 + y**2 + z**2)
		
		vel_mag = magnitude(x_vel,y_vel,z_vel)
		acc_mag = magnitude(x_acc,y_acc,z_acc)
		jerk_mag = magnitude(x_jerk,y_jerk,z_jerk)
		print("Max vel:",np.max(vel_mag),"Max acc:",np.max(acc_mag),"Max jerk:",np.max(jerk_mag))
		figure, axis = plt.subplots(3, 1)
		axis[0].plot(test_linspace, vel_mag)
		axis[0].set_title("Velocity")
		axis[1].plot(test_linspace, acc_mag)
		axis[1].set_title("Acceleration")
		axis[2].plot(test_linspace, jerk_mag)
		axis[2].set_title("Jerk")
		plt.show()
	
	#Extract coefficients
	l, r = [(1, 0)], [(1, 0)]
	x_coeff = PPoly.from_spline(make_interp_spline(test_params, waypoints[:,0], bc_type=(l, r)).tck)
	y_coeff = PPoly.from_spline(make_interp_spline(test_params, waypoints[:,1], bc_type=(l, r)).tck)
	z_coeff = PPoly.from_spline(make_interp_spline(test_params, waypoints[:,2], bc_type=(l, r)).tck)
	
	assert(np.all(x_coeff.x == y_coeff.x) and np.all(y_coeff.x == z_coeff.x)) #If this fails, something is wrong with the knots
	x_basis_c = x_coeff.c.T
	y_basis_c = y_coeff.c.T
	z_basis_c = z_coeff.c.T
	prevKnot = 0.0
	durations = []
	x = []
	y = []
	z = []
	for ind in range(len(x_coeff.x)):
		if x_coeff.x[ind] == prevKnot:
			continue
		x_temp_c = np.append(np.flip(x_basis_c[ind-1]),[0]*4) #Gets the coefficients for the range [x-1,x], reverses it (so x^0 is first) and expands for trajectory
		x.append(x_temp_c)
		y_temp_c = np.append(np.flip(y_basis_c[ind-1]),[0]*4)
		y.append(y_temp_c)
		z_temp_c = np.append(np.flip(z_basis_c[ind-1]),[0]*4)
		z.append(z_temp_c)
		durations.append(x_coeff.x[ind]-prevKnot)
		prevKnot = x_coeff.x[ind]
	x = np.array(x).T
	x[0] += x_offset
	y = np.array(y).T
	y[0] += y_offset
	z = np.array(z).T
	z[0] += z_offset
	
	assert( len(durations) == x.shape[1] and x.shape[1] == y.shape[1] and y.shape[1] == z.shape[1]) #The lengths of the segments must match to print out correctly
	assert( 8 == x.shape[0] and x.shape[0] == y.shape[0] and y.shape[0] == z.shape[0]) #The length of the coefficient arrays must match to 8 for power 7 polynomial
	data = {
		"duration": durations,
		"x0": x[0],
		"x1": x[1],
		"x2": x[2],
		"x3": x[3],
		"x4": x[4],
		"x5": x[5],
		"x6": x[6],
		"x7": x[7],
		"y0": y[0],
		"y1": y[1],
		"y2": y[2],
		"y3": y[3],
		"y4": y[4],
		"y5": y[5],
		"y6": y[6],
		"y7": y[7],
		"z0": z[0],
		"z1": z[1],
		"z2": z[2],
		"z3": z[3],
		"z4": z[4],
		"z5": z[5],
		"z6": z[6],
		"z7": z[7],
		"yaw0": [0]*len(durations),
		"yaw1": [0]*len(durations),
		"yaw2": [0]*len(durations),
		"yaw3": [0]*len(durations),
		"yaw4": [0]*len(durations),
		"yaw5": [0]*len(durations),
		"yaw6": [0]*len(durations),
		"yaw7": [0]*len(durations)
	}
	
	size = range(0,len(durations))
	df = pd.DataFrame(data, index = size)
	df.to_csv(outfile,index=False,header=True)
	
	return
	

