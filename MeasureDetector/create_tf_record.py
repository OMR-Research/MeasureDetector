import hashlib
import io
import json
import os
from glob import glob

import PIL.Image
import tensorflow as tf
from typing import List, Dict, Generator

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data', 'Root directory to raw dataset.')
flags.DEFINE_string('output_path', './training.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'mapping.txt', 'Path to label map proto')
FLAGS = flags.FLAGS


def annotations_to_tf_example_list(all_image_paths: List[str],
                                   all_annotation_paths: List[str],
                                   label_map_dict: Dict[str, int]) -> Generator[tf.train.Example, None, None]:
    """Convert json files and images to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    total_number_of_images = len(all_image_paths)
    for index in tqdm(range(total_number_of_images), desc="Serializing annotations", total=total_number_of_images):
        path_to_image, path_to_annotations = all_image_paths[index], all_annotation_paths[index]

        try:
            with tf.gfile.GFile(path_to_image, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                print(f"Skipping image, that probably does not belong to the project {path_to_image}.")
                continue
            key = hashlib.sha256(encoded_jpg).hexdigest()

            width = image.width
            height = image.height

            xmin = []
            ymin = []
            xmax = []
            ymax = []
            classes = []
            classes_text = []
            truncated = []
            poses = []
            difficult_obj = []

            with open(path_to_annotations, 'r') as gt_file:
                data = json.load(gt_file)

            for bar in data["bars"]:
                left, top, bottom, right = bar["left"], bar["top"], bar["bottom"], bar["right"]

                xmin.append(float(left) / width)
                ymin.append(float(top) / height)
                xmax.append(float(right) / width)
                ymax.append(float(bottom) / height)
                classes.append(label_map_dict["system_measure"])
                classes_text.append("system_measure".encode('utf8'))

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    path_to_image.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                    path_to_image.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
                'image/object/truncated': dataset_util.int64_list_feature(truncated),
                'image/object/view': dataset_util.bytes_list_feature(poses),
            }))

            yield example

        except Exception as ex:
            print(f"Skipping image, that caused an error {path_to_image}.\n{ex}")
            pass


def main(_):
    dataset_directory = FLAGS.data_dir
    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    all_image_paths = glob(f"{dataset_directory}/**/*.jpg", recursive=True)
    all_annotation_paths = glob(f"{dataset_directory}/**/*.json", recursive=True)

    assert (len(all_image_paths) == len(all_annotation_paths))

    for tf_example in annotations_to_tf_example_list(all_image_paths, all_annotation_paths, label_map_dict):
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
