import argparse
import os

from omrdatasettools.converters.EdiromAnnotationConverter import EdiromAnnotationConverter
from omrdatasettools.downloaders.EdiromDatasetDownloader import EdiromDatasetDownloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloads a dataset from the Edirom system')
    parser.add_argument('-dataset', dest='dataset', type=str, required=True,
                        help='Must be either "Bargheer" or "FreischuetzDigital"')
    parser.add_argument('-url', dest='url', type=str, required=True,
                        help='URL where to download the dataset from. Must be provided manual to prevent automatic '
                             'crawling. Please contact the authors if you want to know the URLs.')
    parser.add_argument("--dataset_directory", type=str, default="../data",
                        help="The directory, where the extracted dataset will be copied to")

    flags, unparsed = parser.parse_args()

    downloader = EdiromDatasetDownloader(flags.dataset)
    downloader.download_and_extract_dataset(os.path.join(flags.dataset_directory, flags.dataset))
    downloader.download_images_from_mei_annotation(flags.dataset_directory, flags.url)

    annotation_converter = EdiromAnnotationConverter()
    annotation_converter.convert_annotations_to_one_json_file_per_image(flags.dataset_directory, flags.dataset)

