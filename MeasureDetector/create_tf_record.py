import argparse
import hashlib
import io
import json
import os
import random
from glob import glob

import PIL.Image
import contextlib2
import tensorflow as tf
from typing import List, Dict, Generator

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm


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

        assert (os.path.splitext(os.path.basename(path_to_image))[0] ==
                os.path.splitext(os.path.basename(path_to_annotations))[0])

        try:
            with tf.gfile.GFile(path_to_image, 'rb') as fid:
                encoded_image = fid.read()
            encoded_image_io = io.BytesIO(encoded_image)
            image = PIL.Image.open(encoded_image_io)
            if image.format != 'JPEG' and image.format != 'PNG':
                print(
                    f"Skipping image, that is neither jpeg nor png and probably does not belong to the project {path_to_image}.")
                continue
            if image.width < 600 or image.height < 600:
                print(f"Skipping image, that is smaller than 600x600 and might cause issues {path_to_image}.")
                continue
            key = hashlib.sha256(encoded_image).hexdigest()

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

            object_classes = [("system_measures", "system_measure"), ("stave_measures", "stave_measure"),
                              ("staves", "stave")]
            for class_name, instance_name in object_classes:
                for bounding_box in data[class_name]:
                    left, top, right, bottom = bounding_box["left"], bounding_box["top"], bounding_box["right"], \
                                               bounding_box["bottom"]

                    xmin.append(float(left) / width)
                    ymin.append(float(top) / height)
                    xmax.append(float(right) / width)
                    ymax.append(float(bottom) / height)
                    classes.append(label_map_dict[instance_name])
                    classes_text.append(instance_name.encode('utf8'))

                    assert (left < right)
                    assert (top < bottom)
                    assert (right <= width)
                    assert (bottom <= height)
                    assert (left > 0 and right > 0 and top > 0 and bottom > 0)

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    path_to_image.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                    path_to_image.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_image),
                'image/format': dataset_util.bytes_feature(image.format.lower().encode('utf8')),
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


def get_training_validation_test_indices(all_image_paths):
    seed = 0
    validation_fraction = 0.1
    test_fraction = 0.1
    random.seed(seed)
    dataset_size = len(all_image_paths)
    all_indices = list(range(0, dataset_size))
    validation_sample_size = int(dataset_size * validation_fraction)
    test_sample_size = int(dataset_size * test_fraction)
    validation_sample_indices = random.sample(all_indices, validation_sample_size)
    test_sample_indices = random.sample((set(all_indices) - set(validation_sample_indices)), test_sample_size)
    training_sample_indices = list(set(all_indices) - set(validation_sample_indices) - set(test_sample_indices))

    return training_sample_indices, validation_sample_indices, test_sample_indices


def main(image_directory: str, annotation_directory: str, output_path_training_split: str,
         output_path_validation_split: str, output_path_test_split: str, label_map_path: str, number_of_shards: int):
    os.makedirs(os.path.dirname(output_path_training_split), exist_ok=True)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    all_jpg_image_paths = glob(f"{image_directory}/**/*.jpg", recursive=True)
    all_png_image_paths = glob(f"{image_directory}/**/*.png", recursive=True)
    all_image_paths = all_jpg_image_paths + all_png_image_paths
    all_annotation_paths = glob(f"{annotation_directory}/**/*.json", recursive=True)

    training_sample_indices, validation_sample_indices, test_sample_indices = get_training_validation_test_indices(
        all_image_paths)

    assert (len(all_image_paths) == len(all_annotation_paths))

    with contextlib2.ExitStack() as tf_record_close_stack:
        training_tf_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path_training_split, number_of_shards)
        validation_tf_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path_validation_split, number_of_shards)
        test_tf_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path_test_split, number_of_shards)
        index = 0
        for tf_example in annotations_to_tf_example_list(all_image_paths, all_annotation_paths, label_map_dict):
            shard_index = index % number_of_shards
            index += 1

            if index in training_sample_indices:
                training_tf_records[shard_index].write(tf_example.SerializeToString())
            elif index in validation_sample_indices:
                validation_tf_records[shard_index].write(tf_example.SerializeToString())
            elif index in test_sample_indices:
                test_tf_records[shard_index].write(tf_example.SerializeToString())

    print(f"Exported into\n"
          f"- {len(training_sample_indices)} training samples\n"
          f"- {len(validation_sample_indices)} validation samples\n"
          f"- {len(test_sample_indices)} test samples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a tensorflow record from an existing dataset')
    parser.add_argument('-image_directory', type=str, default="data/muscima_pp/v1.0/data/images",
                        help='Directory, where the images are stored')
    parser.add_argument('-annotation_directory', type=str, default="data/muscima_pp/v1.0/data/json",
                        help='Directory, where the annotations are stored')
    parser.add_argument('-output_path_training_split', type=str, default="data/muscima_pp/training.record",
                        help='Path to output TFRecord')
    parser.add_argument('-output_path_validation_split', type=str, default="data/muscima_pp/validation.record",
                        help='Path to output TFRecord')
    parser.add_argument('-output_path_test_split', type=str, default="data/muscima_pp/test.record",
                        help='Path to output TFRecord')
    parser.add_argument('-label_map_path', type=str, default='mapping.txt',
                        help='Path to label map proto.txt')
    parser.add_argument('-num_shards', type=int, default=1, help='Number of TFRecord shards')

    flags = parser.parse_args()
    image_directory = flags.image_directory
    annotations_directory = flags.annotation_directory
    output_path_training_split = flags.output_path_training_split
    output_path_validation_split = flags.output_path_validation_split
    output_path_test_split = flags.output_path_test_split
    label_map_path = flags.label_map_path
    number_of_shards = flags.num_shards

    main(image_directory, annotations_directory, output_path_training_split, output_path_validation_split,
         output_path_test_split,
         label_map_path, number_of_shards)
