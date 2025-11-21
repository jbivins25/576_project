import pointCloud as pc
import time
import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
import argparse
import json

def pipeline(lasfile, output, start, end, drift_max, optional_points=[]):
	optional_points.insert(0,start)
	optional_points.append(end)
	print(optional_points)
	las = laspy.read(lasfile)
	points = np.vstack((las.x,las.y,las.z)).T
	tree = cKDTree(points)
	point_ind = []
	for x in range(0,len(optional_points)):
		_, ind = tree.query(optional_points[x],k=1,distance_upper_bound=0.1)
		point_ind.append(ind)
	total_time = 0
	path1 = np.empty((0,3))
	print(path1)
	for x in range(0,len(point_ind)-1):
		start = time.time() 
		path = pc.aStarSearch(points[point_ind[x]],points[point_ind[x+1]],points)
		end = time.time()
		total_time += end - start
		path1 = np.append(path1,np.array(path),axis=0)
		print(path1)
	print(f'{total_time} seconds to find path')
	path = np.array(path1)
	unique_points = [path[0]]
	for p in path[1:]:
		if not np.allclose(p, unique_points[-1], atol=1e-9):
			unique_points.append(p)
	path = np.array(unique_points)
	path = path.T
	data = {'x':path[0],'y':path[1],'z':path[2]}
	df = pd.DataFrame(data, range(0,path.shape[1]))
	df.to_csv(f'{output}Flight.csv',index=False,header=False)
	start = time.time()
	pc.genTrajectory(f'{output}Flight.csv',f'{output}Traj.csv', drift_max=drift_max)
	end = time.time()
	print(f'{end-start} seconds to gen traj')


if __name__ == '__main__':
	parser = argparse.ArgumentParser( "Pipeline", description="Pipeline to find shortest path and generate trajectory")
	parser.add_argument("lasfile")
	parser.add_argument("output")
	parser.add_argument("start", type=json.loads)
	parser.add_argument("end", type=json.loads)
	parser.add_argument("drift_max", type=float)
	parser.add_argument("optional_points", nargs='?', type=json.loads)
	args = parser.parse_args()
	if args.optional_points == None:
		pipeline(args.lasfile, args.output, args.start, args.end, args.drift_max)
	else:
		pipeline(args.lasfile, args.output, args.start, args.end, args.drift_max, args.optional_points)
