import argparse
import json
import os
from glob import glob
from xml.dom import minidom
from xml.etree.ElementTree import ElementTree

import numpy
from PIL import Image
from mung.node import Node
from omrdatasettools.converters.ImageColorInverter import ImageColorInverter
from omrdatasettools.converters.ImageConverter import ImageConverter
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader
from tqdm import tqdm
from typing import List, Tuple

from StaveDetector.SingleChannelImageMaskGenerator import SingleChannelImageMaskGenerator


def render_general_masks(annotation_directory: str, image_directory: str, mask_directory: str):
    print("Extracting Masks from arbitrary Dataset...")
    json_files = [y for x in os.walk(annotation_directory) for y in glob(os.path.join(x[0], '*.json'))]
    png_files = [y for x in os.walk(image_directory) for y in glob(os.path.join(x[0], '*.png'))]
    jpg_files = [y for x in os.walk(image_directory) for y in glob(os.path.join(x[0], '*.jpg'))]
    file_paths = list(zip(json_files, png_files + jpg_files))

    os.makedirs(mask_directory, exist_ok=True)

    for json_file, image_file in tqdm(file_paths, desc="Generating mask images"):
        original_image = Image.open(image_file)  # type: Image.Image
        with open(json_file, 'r') as gt_file:
            annotations = json.load(gt_file)

        if "png" in image_file:
            destination_filename = os.path.basename(json_file).replace(".json", ".png")
        else:
            destination_filename = os.path.basename(json_file).replace(".json", ".jpg")

        __render_masks_of_staff_blob_for_instance_segmentation(annotations["staves"], mask_directory,
                                                               destination_filename,
                                                               original_image.width, original_image.height)


def __render_masks_of_staff_blob_for_instance_segmentation(stave_annotations, mask_directory: str,
                                                           destination_filename: str,
                                                           width: int, height: int):
    image_index = 1

    for stave in stave_annotations:

        try:
            image_array = numpy.zeros((height, width), dtype=numpy.uint8)
            for i in range(int(stave["top"]), int(stave["bottom"])):
                for j in range(int(stave["left"]), int(stave["right"])):
                    image_array[i-1, j-1] = 1
            image = Image.fromarray(image_array, mode="L")
            if "png" in destination_filename:
                destination_filename_with_id = destination_filename.replace(".png", "_" + str(image_index) + ".png")
            else:
                destination_filename_with_id = destination_filename.replace(".jpg", "_" + str(image_index) + ".jpg")
            image.save(os.path.join(mask_directory, destination_filename_with_id))
            image_index += 1
        except Exception as ex:
            print("Error drawing node. " + str(ex))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates masks if you already have json annotations of the staves')
    parser.add_argument("--annotation_directory", type=str, default="data")
    parser.add_argument("--image_directory", type=str, default="data")
    parser.add_argument("--mask_directory", type=str, default="data")

    flags = parser.parse_args()

    render_general_masks(flags.annotation_directory, flags.image_directory, flags.mask_directory)
