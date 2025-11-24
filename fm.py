import cv2
import numpy as np

IMG_PATHS = [
	"heart_image0.jpg",
	"heart_image1.jpg",
	"heart_image2.jpg",
	"heart_image3.jpg"
]

MAX_FEATURES = 500
MAX_TRACK_DIST = 2.0

fx = 1444.0
fy = 2587.0
cx = 160.0
cy = 160.0
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
K_inv = np.linalg.inv(K)

poses = [
	(np.eye(3), np.array([-0.1,0.0,0.26])),
	(np.eye(3), np.array([0.0,0.0,0.26])),
	(np.eye(3), np.array([0.1,0.0,0.26])),
	(np.eye(3), np.array([0.0,0.1,0.26]))
]

def pixel_to_ray(u,v):
	p = np.array([u,v,1.0])
	ray = K_inv @ p
	return ray / np.linalg.norm(ray)

def triangulate_two_views(ray1,R1,t1,ray2,R2,t2):
	d1 = R1 @ ray1
	d2 = R2 @ ray2
	d1 /= np.linalg.norm(d1)
	d2 /= np.linalg.norm(d2)
	A = np.stack([d1,-d2], axis=1)
	b = t2 - t1
	sol, _, _, _ = np.linalg.lstsq(A,b,rcond=None)
	s,t = sol
	p1 = t1 + s*d1
	p2 = t2 + t*d2
	return 0.5*(p1+p2)

def detect_features(img):
	orb = cv2.ORB_create(MAX_FEATURES)
	kp, des = orb.detectAndCompute(img,None)
	return kp, des

def match_features(des1, des2):
	if des1 is None or des2 is None:
		return []
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	return matches

class Track:
	def __init__(self,id,frame,pt):
		self.id = id
		self.obs = [(frame, pt)]
		self.last_frame = frame

	def add_obs(self,frame,pt):
		self.obs.append((frame,pt))
		self.last_frame = frame

def build_multiframe_tracks(images):
	tracks = {}
	next_id = 0
	kp_list, des_list = [], []
	for img in images:
		kp, des = detect_features(img)
		kp_list.append(kp)
		des_list.append(des)
	
	for i in range(len(images)-1):
		matches = match_features(des_list[i], des_list[i+1])
		# map kp idx in frame i -> track id
		idx_to_tid = {}
		for tid,tr in tracks.items():
			if tr.last_frame == i:
				# find kp idx in frame i closest to last observation
				last_pt = tr.obs[-1][1]
				dists = [np.linalg.norm(np.array(last_pt)-np.array(kp_list[i][j].pt)) for j in range(len(kp_list[i]))]
				if len(dists) > 0:
					best_idx = np.argmin(dists)
					if dists[best_idx] < 5.0:  # pixel threshold
						idx_to_tid[best_idx] = tid
		for m in matches:
			if m.queryIdx in idx_to_tid:
				tracks[idx_to_tid[m.queryIdx]].add_obs(i+1, kp_list[i+1][m.trainIdx].pt)
			else:
				tracks[next_id] = Track(next_id, i, kp_list[i][m.queryIdx].pt)
				tracks[next_id].add_obs(i+1, kp_list[i+1][m.trainIdx].pt)
				next_id += 1
	return tracks

def triangulate_tracks(tracks):
	points = []
	for tid,tr in tracks.items():
		if len(tr.obs) < 2:
			points.append(None)
			continue
		best_pair = max([(tr.obs[i], tr.obs[j]) for i in range(len(tr.obs)) for j in range(i+1,len(tr.obs))],
						key=lambda x: abs(x[0][0]-x[1][0]))
		(fr1, pt1), (fr2, pt2) = best_pair
		ray1 = pixel_to_ray(pt1[0], pt1[1])
		ray2 = pixel_to_ray(pt2[0], pt2[1])
		R1, t1 = poses[fr1]
		R2, t2 = poses[fr2]
		P = triangulate_two_views(ray1,R1,t1,ray2,R2,t2)
		for (f,pt) in tr.obs:
			if f == fr1 or f == fr2:
				continue
			ray = pixel_to_ray(pt[0], pt[1])
			R, t = poses[f]
			d = R @ ray
			d /= np.linalg.norm(d)
			v = P - t
			s = np.dot(v,d)
			P_new = t + s*d
			P = 0.5*(P+P_new)
		points.append(P)
	return points

def visualize_tracks(images,tracks,points):
	vis = [cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR) for im in images]
	for tid,tr in tracks.items():
		P = points[tid]
		if P is None:
			continue
		if np.linalg.norm(P - poses[0][1]) > MAX_TRACK_DIST:
			continue
		for (f,pt) in tr.obs:
			cv2.circle(vis[f], (int(pt[0]), int(pt[1])), 3, (0,255,255), 1)
	combined = np.hstack(vis)
	cv2.imshow("Tracks", combined)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def run_pipeline(image_paths):
	clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(16,16))
	images = [clahe.apply(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in image_paths]
	tracks = build_multiframe_tracks(images)
	print(f"Built {len(tracks)} tracks.")
	points = triangulate_tracks(tracks)
	for i,P in enumerate(points):
		if P is None:
			continue
		dist = np.linalg.norm(P - poses[0][1])
		if dist <= MAX_TRACK_DIST:
			dx,dy,dz = P - poses[0][1]
			print(f"Track {i}: P={P}, dist={dist:.2f}, rel=({dx:.2f},{dy:.2f},{dz:.2f})")
	visualize_tracks(images,tracks,points)
	return tracks, points

if __name__=="__main__":
	tracks, points = run_pipeline(IMG_PATHS)
