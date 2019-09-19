import argparse
import hashlib
import io
import json
import os
import random
from glob import glob
from typing import List, Dict, Generator

import PIL.Image
import contextlib2
import tensorflow as tf
from MeasureDetector.errors import InvalidImageFormatError, InvalidImageError
from PIL.Image import Image
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util, label_map_util
from tqdm import tqdm


def encode_sample_into_tensorflow_sample(path_to_image: str, annotations: Dict, mask_paths: List[str],
                                         label_map_dict: Dict[str, int], scale_factor: float):
    image = PIL.Image.open(path_to_image)  # type: Image
    image_format = image.format

    if image_format != 'JPEG' and image_format != 'PNG':
        raise InvalidImageFormatError(
            f"Skipped image {path_to_image} that is neither jpeg nor png and probably does not belong to the project.")
    if image.width < 600 or image.height < 600:
        raise InvalidImageError(f"Skipped image {path_to_image} that is smaller than 600x600 and might cause issues.")

    image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), PIL.Image.LANCZOS)
    encoded_image_io = io.BytesIO()
    image.save(encoded_image_io, format='PNG')
    encoded_image = encoded_image_io.getvalue()

    key = hashlib.sha256(encoded_image).hexdigest()

    image_width = image.width
    image_height = image.height

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    object_classes = [("staves", "stave")]
    encoded_mask_png_list = []
    for mask_path in mask_paths:
        mask = PIL.Image.open(mask_path)  # type: Image
        mask = mask.resize((int(mask.width * scale_factor), int(mask.height * scale_factor)), PIL.Image.NEAREST)
        output = io.BytesIO()
        mask.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())

    for class_name, instance_name in object_classes:
        for bounding_box in annotations[class_name]:
            left, top, right, bottom = bounding_box["left"], bounding_box["top"], bounding_box["right"], \
                                       bounding_box["bottom"]

            if left >= right:
                continue
            if top >= bottom:
                continue
            if right > image_width:
                right = image_width
            if bottom > image_height:
                bottom = image_height
            if left < 0 or right < 0 or top < 0 or bottom < 0:
                continue

            xmin.append(float(left) / image_width * scale_factor)
            ymin.append(float(top) / image_height * scale_factor)
            xmax.append(float(right) / image_width * scale_factor)
            ymax.append(float(bottom) / image_height * scale_factor)
            classes.append(label_map_dict[instance_name])
            classes_text.append(instance_name.encode('utf8'))

    assert (len(xmin) == len(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(
            path_to_image.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            path_to_image.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format.lower().encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
        'image/object/mask': dataset_util.bytes_list_feature(encoded_mask_png_list),
    }))
    return example


def annotations_to_tf_example_list(all_image_paths: List[str],
                                   all_annotation_paths: List[str],
                                   all_mask_paths: List[str],
                                   label_map_dict: Dict[str, int],
                                   scale_factor: float) -> Generator[tf.train.Example, None, None]:
    """Convert json files and images to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    total_number_of_images = len(all_image_paths)
    number_of_skipped_or_errored_samples = 0
    error_messages = []
    for index in tqdm(range(total_number_of_images), desc="Serializing annotations", total=total_number_of_images):
        path_to_image, path_to_annotations = all_image_paths[index], all_annotation_paths[index]

        basename = os.path.splitext(os.path.basename(path_to_image))[0]
        assert (basename == os.path.splitext(os.path.basename(path_to_annotations))[0])

        masks = [path for path in all_mask_paths if basename in path]

        try:
            with open(path_to_annotations, 'r') as gt_file:
                annotations = json.load(gt_file)

            example = encode_sample_into_tensorflow_sample(path_to_image, annotations, masks, label_map_dict,
                                                           scale_factor)
            yield example

        except Exception as ex:
            error_messages.append(f"Skipped image {path_to_image} that caused an error: {ex}")
            number_of_skipped_or_errored_samples += 1

    print("Skipped {0} samples".format(number_of_skipped_or_errored_samples))
    for sample in error_messages:
        print(sample)


def get_training_validation_test_indices(all_image_paths):
    seed = 0
    validation_fraction = 0.1
    test_fraction = 0.0
    random.seed(seed)
    dataset_size = len(all_image_paths)
    all_indices = list(range(0, dataset_size))
    validation_sample_size = int(dataset_size * validation_fraction)
    test_sample_size = int(dataset_size * test_fraction)
    validation_sample_indices = random.sample(all_indices, validation_sample_size)
    test_sample_indices = random.sample((set(all_indices) - set(validation_sample_indices)), test_sample_size)
    training_sample_indices = list(set(all_indices) - set(validation_sample_indices) - set(test_sample_indices))

    return training_sample_indices, validation_sample_indices, test_sample_indices


def main(image_directory: str, annotation_directory: str, mask_directory: str, output_path_training_split: str,
         output_path_validation_split: str, output_path_test_split: str, label_map_path: str, number_of_shards: int,
         scale_factor: float):
    os.makedirs(os.path.dirname(output_path_training_split), exist_ok=True)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    all_jpg_image_paths = glob(f"{image_directory}/**/*.jpg", recursive=True)
    all_png_image_paths = glob(f"{image_directory}/**/*.png", recursive=True)
    all_image_paths = all_jpg_image_paths + all_png_image_paths
    all_annotation_paths = glob(f"{annotation_directory}/**/*.json", recursive=True)
    all_mask_paths = glob(f"{mask_directory}/**/*.png", recursive=True)

    # Filter out the dataset.json files, which are complete dataset annotations
    all_annotation_paths = [a for a in all_annotation_paths if "dataset.json" not in a]

    training_sample_indices, validation_sample_indices, test_sample_indices = get_training_validation_test_indices(
        all_image_paths)

    all_annotation_paths = sorted(all_annotation_paths)
    all_image_paths = sorted(all_image_paths)

    if len(all_image_paths) != len(all_annotation_paths):
        print("Not every image has annotations")

    for annotation_path, image_path in zip(all_annotation_paths, all_image_paths):
        if os.path.splitext(os.path.basename(image_path))[0] not in annotation_path:
            print("Invalid annotations detected: {0}, {1}".format(image_path, annotation_path))

    print(f"Exporting\n"
          f"- {len(training_sample_indices)} training samples\n"
          f"- {len(validation_sample_indices)} validation samples\n"
          f"- {len(test_sample_indices)} test samples")

    with contextlib2.ExitStack() as tf_record_close_stack:
        training_tf_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path_training_split, number_of_shards)
        validation_tf_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path_validation_split, number_of_shards)
        test_tf_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path_test_split, number_of_shards)
        index = 0
        for tf_example in annotations_to_tf_example_list(all_image_paths, all_annotation_paths, all_mask_paths,
                                                         label_map_dict, scale_factor):
            shard_index = index % number_of_shards
            index += 1

            if index in training_sample_indices:
                training_tf_records[shard_index].write(tf_example.SerializeToString())
            elif index in validation_sample_indices:
                validation_tf_records[shard_index].write(tf_example.SerializeToString())
            elif index in test_sample_indices:
                test_tf_records[shard_index].write(tf_example.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a tensorflow record from an existing dataset.'
                                                 'Recursively searchers the image- and annotation-directories'
                                                 'for png/jpg files and json files respectively. One json-file'
                                                 'per image has to be stored in the folders, where the filenames'
                                                 'must match, except for the file-ending.')
    parser.add_argument('--image_directory', type=str, default="data/muscima_pp/v2.0/data/images",
                        help='Directory, where the images are stored')
    parser.add_argument('--mask_directory', type=str, default="data/muscima_pp_masks",
                        help='Directory, where the images are stored')
    parser.add_argument('--annotation_directory', type=str, default="data/muscima_pp/v2.0/data/json",
                        help='Directory, where the annotations are stored')
    parser.add_argument('--output_path_training_split', type=str, default="data/training.record",
                        help='Path to output TFRecord')
    parser.add_argument('--output_path_validation_split', type=str, default="data/validation.record",
                        help='Path to output TFRecord')
    parser.add_argument('--output_path_test_split', type=str, default="data/test.record",
                        help='Path to output TFRecord')
    parser.add_argument('--label_map_path', type=str, default='mapping.txt',
                        help='Path to label map proto.txt')
    parser.add_argument('--num_shards', type=int, default=4, help='Number of TFRecord shards')
    parser.add_argument('--scale_factor', type=float, default=1, help='Scale factor of the images and annotations')

    flags = parser.parse_args()
    image_directory = flags.image_directory
    annotations_directory = flags.annotation_directory
    mask_directory = flags.mask_directory
    output_path_training_split = flags.output_path_training_split
    output_path_validation_split = flags.output_path_validation_split
    output_path_test_split = flags.output_path_test_split
    label_map_path = flags.label_map_path
    number_of_shards = flags.num_shards
    scale_factor = flags.scale_factor

    main(image_directory, annotations_directory, mask_directory, output_path_training_split,
         output_path_validation_split,
         output_path_test_split, label_map_path, number_of_shards, scale_factor)
