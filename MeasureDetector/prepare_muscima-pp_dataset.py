import os

from omrdatasettools.converters.ImageColorInverter import ImageColorInverter
from omrdatasettools.converters.ImageConverter import ImageConverter
from omrdatasettools.converters.MuscimaPlusPlusAnnotationConverter import MuscimaPlusPlusAnnotationConverter
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == "__main__":
    dataset_directory = os.path.join("data", "muscima_pp")

    dataset_downloader = MuscimaPlusPlusDatasetDownloader()
    dataset_downloader.download_and_extract_dataset(dataset_directory)

    image_inverter = ImageColorInverter()
    image_inverter.invert_images(dataset_directory, "*.png")

    image_converter = ImageConverter()
    image_converter.convert_grayscale_images_to_rgb_images(dataset_directory)

    annotation_converter = MuscimaPlusPlusAnnotationConverter()
    annotation_converter.convert_measure_annotations_to_one_json_file_per_image(dataset_directory)
