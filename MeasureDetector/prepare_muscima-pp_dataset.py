import argparse
import os

from omrdatasettools.converters.ImageColorInverter import ImageColorInverter
from omrdatasettools.converters.ImageConverter import ImageConverter
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloads the MUSCIMA++ dataset')
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, where the extracted dataset will be copied to")

    flags = parser.parse_args()
    dataset_directory = os.path.join(flags.dataset_directory, "muscima_pp")

    dataset_downloader = MuscimaPlusPlusDatasetDownloader()
    dataset_downloader.download_and_extract_dataset(dataset_directory)
    dataset_downloader.download_and_extract_measure_annotations(dataset_directory)

    image_inverter = ImageColorInverter()
    image_inverter.invert_images(dataset_directory, "*.png")

    image_converter = ImageConverter()
    image_converter.convert_grayscale_images_to_rgb_images(dataset_directory)
