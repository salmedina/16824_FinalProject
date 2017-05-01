#!/user/bin/env python

from __future__ import division
import os
import sys
import json
import glob
import json
import numpy as np
from math import ceil
from random import randint
from os.path import join
import seaborn as sns

sns.palplot(sns.color_palette("Set2", 10))

#################################
# AUXILIARY FUNCTIONS

def which_dirs_in_path(path, dirs):
    subdirs = os.listdir(path)
    return [d for d in dirs if d in subdirs and os.path.isdir(join(path, d))]


def extract_data(data_path, sel_classes, num_frames, actions_dict):
    '''
    Reads the data from the folders and loads into numpy
    @returns: a tuple with the data samples and datalabels
    '''

    # Verify that all sel_classes are in path and notify missing classes
    found_classes = which_dirs_in_path(data_path, sel_classes)
    if set(found_classes) != set(sel_classes):
        print 'Not all the classes were found. The missing classes are:'
        print ', '.join(list(set(sel_classes) - set(found_classes)))
        print ''

    REQ_FRAMES = num_frames
    data_samples = []
    data_labels = []
    # For each selected class
    for sel_class in found_classes:
        print '>>> Processing class {} <<<'.format(sel_class)
        # Get its full path and the videos which are dirs
        sel_class_path = join(data_path, sel_class)
        sel_class_videos = get_dirs(sel_class_path)
        print 'Found {} videos for the class {}'.format(len(sel_class_videos), sel_class)

        # For each video for the selected class
        for video_name in sel_class_videos:
            # Each of the json files correspond to each frame
            vf_jsons = glob.glob(join(sel_class_path, video_name, '*.json'))
            print 'Found {} frames for {}'.format(len(vf_jsons), video_name)

            # Collect all the frames for  the video

            last_pose = np.zeros(54)
            video_poses = np.array([])
            missing_count = 0
            for vf_json_path in vf_jsons:
                vf_json = json.load(open(vf_json_path))
                if 'people' in vf_json and len(vf_json['people']) > 0 and 'body_parts' in vf_json['people'][0]:
                    frame_pose = vf_json['people'][0]['body_parts']
                    if len(video_poses) < 1:
                        video_poses = np.array([frame_pose], dtype=np.float32)
                    else:
                        video_poses = np.concatenate((video_poses, [frame_pose]))
                    last_pose = frame_pose
                else:
                    if len(video_poses) < 1:  # When the first frame did not find any pose
                        continue
                    missing_count += 1
                    video_poses = np.concatenate((video_poses, [last_pose]))
            if missing_count > 0:
                print 'Missing frames: {}/{}'.format(missing_count, len(video_poses))

            if len(video_poses) < REQ_FRAMES:
                # Loop through the video until we get the required length
                loops = int(ceil(REQ_FRAMES / len(video_poses)))
                video_poses = np.tile(video_poses, (loops, 1))[:REQ_FRAMES]
            elif len(video_poses) > REQ_FRAMES:
                # Select a segment at random from the video
                start_idx = randint(0, len(video_poses) - REQ_FRAMES)  # randint is inclusive
                video_poses = video_poses[start_idx:start_idx + REQ_FRAMES]

            # Append pose matrix to data tensor
            if len(data_samples) < 1:
                data_samples = np.array([video_poses], dtype=np.float32)
            else:
                data_samples = np.concatenate((data_samples, [video_poses]))

            # Append label to label tensor
            data_labels.append(actions_dict[sel_class])

        assert len(data_samples) == len(data_labels)

    data_labels = np.array(data_labels)

    return data_samples, data_labels


def store_npdata(save_path, data, labels):
    '''
    Saves data into a compressed numpy binary file
    @returns: None
    '''
    np.savez_compressed(save_path, data=data, labels=labels)


def load_npdata(save_path):
    '''
    Loades the stored binary data
    @returns: tuple with data and labels as numpy arrays
    '''
    loaded_data = np.load(save_path)
    data = loaded_data['data']
    labels = loaded_data['labels']
    return data, labels


def get_dirs(path):
    if not os.path.exists(path):
        return []

    return [x for x in os.listdir(path) if os.path.isdir(join(path, x)) and not x.startswith('.')]


def get_list_inv_dict(in_list):
    inv_dict = {}
    for i, x in enumerate(in_list):
        inv_dict[x] = i
    return inv_dict

def load_classes(classes_path):
    '''
    Loads the list of classes found in a text file which are newline separated
    :param classes_path: path to a text file with the selected classes to load 
    :return: list of classes found in file
    '''
    if not os.path.exists(classes_path):
        return []
    return [l.strip() for l in open(classes_path).readlines()]

def main(data_path, data_save_path, sel_classes_path):
    data_path = '../Data/mockup/'
    data_save_path = '../Data/mockup.npz'
    sel_classes_path = './body_motion_labels.txt'
    sel_classes = load_classes(sel_classes_path)
    req_frames = 180

    print len(sel_classes), 'human motion classes'
    actions_dict = get_list_inv_dict(sel_classes)

    data, labels = extract_data(data_path, sel_classes, req_frames, actions_dict)
    store_npdata(data_save_path, data, labels)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python data_proc.py <config_json>'

    config = json.load(open(sys.argv[1]))
    data_path = config['data_path']
    data_save_path = config['data_save_path']
    sel_classes_path = config['sel_classes_path']
    main(data_path, data_save_path, sel_classes_path)

