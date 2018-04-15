# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf

from tensorflow.python.platform import app
from delf import feature_io

cmd_args = None

_DISTANCE_THRESHOLD = 0.8

def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths

def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)
  


  query_image_paths =_ReadImageList(cmd_args.query_list_images_path)
  train_image_paths = _ReadImageList(cmd_args.train_list_images_path)

  output_fh = open(cmd_args.output_file, 'w')


  for query in query_image_paths:
    # Read features.
    locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
        query)
    num_features_1 = locations_1.shape[0]
    #tf.logging.info("Loaded image 1's %d features" % num_features_1)

    best_f = ""
    best_inliers = 0
    for train in train_image_paths:

      locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
          train)
      num_features_2 = locations_2.shape[0]
      #tf.logging.info("Loaded image 2's %d features" % num_features_2)

      # Find nearest-neighbor matches using a KD tree.
      d1_tree = cKDTree(descriptors_1)
      _, indices = d1_tree.query(
          descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

      # Select feature locations for putative matches.
      locations_2_to_use = np.array([
          locations_2[i,]
          for i in range(num_features_2)
          if indices[i] != num_features_1
      ])
      locations_1_to_use = np.array([
          locations_1[indices[i],]
          for i in range(num_features_2)
          if indices[i] != num_features_1
      ])

      # Perform geometric verification using RANSAC.
      try:
        _, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=3,
            residual_threshold=20,
            max_trials=100)

        sum_inliers =  sum(inliers)
        if(sum_inliers > best_inliers):
          best_inliers = sum_inliers
          best_label = train

        #print(best_label, train, query)
      except:
        continue

      #print(best_label, train, query)

      #tf.logging.info('Found %d inliers' % sum(inliers))
    if(best_inliers>20):
        output_fh.write(query.split('data/query_features/')[1]\
          + ","+best_label.split('data/train_features/')[1]+"\n")
    else:
        output_fh.write(query+"\n")

  output_fh.close()


      # Visualize correspondences, and save to file.
      # _, ax = plt.subplots()
      # img_1 = mpimg.imread(cmd_args.image_1_path)
      # img_2 = mpimg.imread(cmd_args.image_2_path)
      # inlier_idxs = np.nonzero(inliers)[0]
      # plot_matches(
      #     ax,
      #     img_1,
      #     img_2,
      #     locations_1_to_use,
      #     locations_2_to_use,
      #     np.column_stack((inlier_idxs, inlier_idxs)),
      #     matches_color='b')
      # ax.axis('off')
      # ax.set_title('DELF correspondences')
      # plt.savefig(cmd_args.output_image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--query_list_images_path',
      type=str,
      default='query_features_list.txt',
      help="""
      Path to list of query images.
      """)
  parser.add_argument(
      '--train_list_images_path',
      type=str,
      default='train_features_list.txt',
      help="""
      Path to list of training images.
      """)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='data/train_features/',
      help="""
      Path of DIR where train features are saved.
      """)
  parser.add_argument(
      '--query_dir',
      type=str,
      default='data/query_features/',
      help="""
      Path to DIR where query features are saved.
      """)
  parser.add_argument(
      '--output_file',
      type=str,
      default='submission.csv',
      help="""
      Path where output csv file will be save.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
