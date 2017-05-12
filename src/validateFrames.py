import os
from os.path import exists, join, basename
import sys
import csv
import glob

if len(sys.argv) < 3:
	print '''validateFrames.py <csv_file> <dataset_path>'''

csv_path = sys.argv[1]
dataset_path = sys.argv[2]

csv_rdr = csv.reader(open(csv_path))

non_found_dirs = set()
for vidid, start_frame, end_frame, _, _ in csv_rdr:
	start_frame = int(start_frame)
	end_frame = int(end_frame)
	dir_path = join(dataset_path, vidid)
	if not os.path.exists(dir_path):
		non_found_dirs.add(dir_path)
		print dir_path
		continue
	for i in range(start_frame+1, end_frame):
		vid_frame_filename = '%s-%06d_pose.json'%(vidid,i)
		frame_path = join(dir_path, vid_frame_filename)
		if not os.path.exists(frame_path):
			print frame_path

print len(non_found_dirs)
