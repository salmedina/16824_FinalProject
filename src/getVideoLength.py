import os
import sys
import glob
import re
from os.path import join

if len(sys.argv) < 2:
	print '''Usage: getVideoLength.py <dataset_path>'''

dataset_path = sys.argv[1]

for item in os.listdir(dataset_path):
	item_path = join(dataset_path, item)
	if os.path.isdir(item_path):
		json_files = glob.glob(join(item_path,'*.json'))
		json_files.sort(reverse=True)	
		last_name = json_files[0]
		vid_len = int(re.findall(r'-(\d+)_pose.json', last_name)[0])
		print '{},{}'.format(item, vid_len)
