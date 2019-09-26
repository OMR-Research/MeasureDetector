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


def render_node_masks(raw_data_directory: str, destination_directory: str, stave_annotation_directory: str):
    print("Extracting Masks from DeepScores Dataset...")

    file_paths = __get_all_file_paths(raw_data_directory)
    os.makedirs(stave_annotation_directory, exist_ok=True)
    for xml_file, png_file in tqdm(file_paths, desc="Generating mask images"):
        original_image = Image.open(png_file)  # type: Image.Image
        nodes = __read_objects(xml_file)
        destination_filename = os.path.basename(xml_file).replace(".xml", ".png")
        stave_annotation_filename = os.path.join(stave_annotation_directory,
                                                 os.path.basename(xml_file).replace(".xml", ".json"))
        __render_masks_of_staff_blob_for_instance_segmentation(nodes, destination_directory,
                                                               destination_filename,
                                                               original_image.width, original_image.height,
                                                               stave_annotation_filename)


def __read_objects(xml_file):
    document = minidom.parse(xml_file)
    width = int(document.getElementsByTagName("width")[0].firstChild.nodeValue)
    height = int(document.getElementsByTagName("height")[0].firstChild.nodeValue)

    nodes = []
    for node in document.getElementsByTagName('object'):
        left = float(node.getElementsByTagName("xmin")[0].firstChild.nodeValue) * width
        right = float(node.getElementsByTagName("xmax")[0].firstChild.nodeValue) * width
        top = float(node.getElementsByTagName("ymin")[0].firstChild.nodeValue) * height
        bottom = float(node.getElementsByTagName("ymax")[0].firstChild.nodeValue) * height
        new_node = Node(0, node.getElementsByTagName("name")[0].firstChild.nodeValue, round(top), round(left),
                        round(right - left),
                        round(bottom - top))
        if new_node.class_name == "staffLine":
            nodes.append(new_node)
    nodes = sorted(nodes, key=lambda node: node.top)
    return nodes


def __get_all_file_paths(raw_data_directory: str) -> List[Tuple[str, str]]:
    """ Loads all XML-files that are located in the folder.
    :param raw_data_directory: Path to the raw directory, where the MUSCIMA++ dataset was extracted to
    """
    annotations_directory = os.path.join(raw_data_directory, "xml_annotations")
    xml_files = [y for x in os.walk(annotations_directory) for y in glob(os.path.join(x[0], '*.xml'))]
    images_directory = os.path.join(raw_data_directory, "images_png")
    png_files = [y for x in os.walk(images_directory) for y in glob(os.path.join(x[0], '*.png'))]
    return list(zip(xml_files, png_files))


def __render_masks_of_staff_blob_for_instance_segmentation(nodes: List[Node], destination_directory: str,
                                                           destination_filename: str,
                                                           width: int, height: int,
                                                           stave_annotation_filename):
    included_classes = ["staffLine"]
    staff_line_index = 0
    os.makedirs(destination_directory, exist_ok=True)
    image_index = 1
    stave_annotations = {}
    stave_annotations["width"] = width
    stave_annotations["height"] = width
    stave_annotations["system_measures"] = {}
    stave_annotations["stave_measures"] = {}
    staves = []

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
                staves.append({"left": node.left, "top": first_staff_line_of_staff.top, "right": node.right,
                               "bottom": node.bottom})
                image_index += 1
            except:
                print("Error drawing node {0}".format(node.unique_id))

        if staff_line_index == 5:
            staff_line_index = 0

        staff_line_index += 1
        if staff_line_index == 1:
            first_staff_line_of_staff = node

    stave_annotations["staves"] = staves
    with open(stave_annotation_filename, mode="w") as file:
        json.dump(stave_annotations, file, indent="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloads and prepares the DeepScores dataset')
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, where the extracted dataset will be copied to")

    flags = parser.parse_args()
    dataset_directory = os.path.join(flags.dataset_directory, "deep_scores")
    mask_directory = os.path.join(flags.dataset_directory, "deep_scores_masks")
    stave_annotations_directory = os.path.join(flags.dataset_directory, "deep_scores_staves")

    # TODO: Download the dataset
    # dataset_downloader = MuscimaPlusPlusDatasetDownloader(dataset_version="2.0")
    # dataset_downloader.download_and_extract_dataset(dataset_directory)
    # dataset_downloader.download_and_extract_measure_annotations(dataset_directory)

    render_node_masks(dataset_directory, mask_directory, stave_annotations_directory)
