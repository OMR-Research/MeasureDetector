import argparse
import json
import os
from collections import defaultdict
from glob import glob
from typing import List, Tuple

import sys

import numpy
from mung.io import read_nodes_from_file, parse_node_classes

from PIL import Image
from mung.node import Node
from tqdm import tqdm
from enum import Enum


class SingleChannelImageMaskGenerator:
    def __init__(self) -> None:
        super().__init__()
        self.path_of_this_file = os.path.dirname(os.path.realpath(__file__))

    def render_node_masks(self, raw_data_directory: str, destination_directory: str):
        """
        Extracts all symbols from the raw XML documents and generates individual symbols from the masks

        :param raw_data_directory: The directory, that contains the xml-files and matching images
        :param destination_directory: The directory, in which the symbols should be generated into.
                                      Per file, one mask will be generated.
        """
        print("Extracting Masks from Muscima++ Dataset...")

        file_paths = self.__get_all_file_paths(raw_data_directory)
        for xml_file, png_file in tqdm(file_paths, desc="Generating mask images"):
            original_image = Image.open(png_file)  # type: Image.Image
            nodes = read_nodes_from_file(xml_file)
            destination_filename = os.path.basename(xml_file).replace(".xml", ".png")
            self.__render_masks_of_staff_blob_for_instance_segmentation(nodes, destination_directory,
                                                                        destination_filename,
                                                                        original_image.width, original_image.height)

    def __get_all_file_paths(self, raw_data_directory: str) -> List[Tuple[str, str]]:
        """ Loads all XML-files that are located in the folder.
        :param raw_data_directory: Path to the raw directory, where the MUSCIMA++ dataset was extracted to
        """
        annotations_directory = os.path.join(raw_data_directory, "v2.0", "data", "annotations")
        xml_files = [y for x in os.walk(annotations_directory) for y in glob(os.path.join(x[0], '*.xml'))]
        images_directory = os.path.join(raw_data_directory, "v2.0", "data", "images")
        png_files = [y for x in os.walk(images_directory) for y in glob(os.path.join(x[0], '*.png'))]
        return list(zip(xml_files, png_files))

    def __render_masks_of_staff_blob_for_instance_segmentation(self, nodes: List[Node], destination_directory: str,
                                                               destination_filename: str,
                                                               width: int, height: int):
        included_classes = ["staffLine"]
        staff_line_index = 0
        os.makedirs(destination_directory, exist_ok=True)
        image_index = 1
        for node in nodes:
            if node.class_name not in included_classes:
                continue

            if staff_line_index == 4:
                try:
                    image_array = numpy.zeros((height, width), dtype=numpy.uint8)
                    for i in range(first_staff_line_of_staff.top, node.bottom):
                        for j in range(node.left, node.right):
                            image_array[i, j] = 1
                    image = Image.fromarray(image_array, mode="L")
                    destination_filename_with_id = destination_filename.replace(".png", "_" + str(image_index) + ".png")
                    image.save(os.path.join(destination_directory, destination_filename_with_id))
                    image_index += 1
                except:
                    print("Error drawing node {0}".format(node.unique_id))

            if staff_line_index == 5:
                staff_line_index = 0

            staff_line_index += 1
            if staff_line_index == 1:
                first_staff_line_of_staff = node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_directory",
        type=str,
        default="data/muscima_pp",
        help="The directory, where the raw Muscima++ dataset can be found")
    parser.add_argument(
        "--image_dataset_directory",
        type=str,
        default="data/muscima_pp_masks",
        help="The directory, where the generated bitmaps will be created")

    flags, unparsed = parser.parse_known_args()

    mask_image_generator = SingleChannelImageMaskGenerator()
    mask_image_generator.render_node_masks(flags.raw_dataset_directory, flags.image_dataset_directory)
