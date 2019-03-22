import argparse
import json
import os
from glob import glob

from tqdm import tqdm
from typing import Dict


def computer_overlap(previous_top, previous_bottom, current_top, current_bottom):
    if current_top > previous_bottom or previous_top > current_bottom:
        return 0

    overlap_start = max(previous_top, current_top)
    overlap_end = max(previous_bottom, current_bottom)
    return overlap_end - overlap_start


def in_the_same_system(previous_top, previous_bottom, current_top, current_bottom):
    overlapping_range = computer_overlap(previous_top, previous_bottom, current_top, current_bottom)
    if overlapping_range > (current_bottom - current_top) * 0.5:
        return True
    return False


def create_engraving_statistics(dataset_directory: str):
    if len(glob(f'{dataset_directory}/**/dataset.json', recursive=True)) == 0:
        print(f"Could not find json files in {dataset_directory} directory.")

    engraving_types = dict()  # type: Dict[str, int]
    number_of_systems = [0] * 15  # type: Dict[int, int]
    individual_results = []
    for dataset in tqdm(glob(f'{dataset_directory}/**/dataset.json', recursive=True), desc="Converting annotations"):
        with open(dataset, "r") as file:
            dataset_annotations = json.load(file)

        for name, source in dataset_annotations["sources"].items():
            engraving_type = source["type"]

            current_number_of_pages_of_this_type = 0
            if engraving_type in engraving_types:
                current_number_of_pages_of_this_type = engraving_types[engraving_type]

            current_number_of_pages_of_this_type += len(source["pages"])
            engraving_types[engraving_type] = current_number_of_pages_of_this_type

            for page in source["pages"]:
                system_measures = []

                for measure in page["annotations"]["measures"]:
                    left = int(measure["bbox"]["x"])
                    top = int(measure["bbox"]["y"])
                    right = int(measure["bbox"]["x"] + measure["bbox"]["width"])
                    bottom = int(measure["bbox"]["y"] + measure["bbox"]["height"])

                    data = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
                    system_measures.append(data)

                number_of_systems_on_this_page = compute_number_of_system_on_page(system_measures)
                number_of_systems[number_of_systems_on_this_page] += 1
                individual_results.append((engraving_type, number_of_systems_on_this_page))

    for engraving_type, number_of_pages in engraving_types.items():
        print("Found {0} pages of engraving {1}.".format(number_of_pages, engraving_type))

    for index in range(len(number_of_systems)):
        print("Pages with {1} systems: {0}.".format(number_of_systems[index], index))

    with open("dataset_statistics.csv", "w") as statistics_file:
        statistics_file.write("Engraving Type,Number of Systems\n")
        for item in individual_results:
            statistics_file.write(item[0] + "," + str(item[1]) + "\n")


def compute_number_of_system_on_page(system_measures) -> int:
    number_of_systems_on_this_page = 0
    previous_top = 0
    previous_bottom = 0
    for system_measure in system_measures:
        current_measure_is_in_the_same_system_as_last_measure = in_the_same_system(previous_top,
                                                                                   previous_bottom,
                                                                                   system_measure["top"],
                                                                                   system_measure["bottom"])
        previous_top = system_measure["top"]
        previous_bottom = system_measure["bottom"]

        if current_measure_is_in_the_same_system_as_last_measure:
            continue
        else:
            number_of_systems_on_this_page += 1
    return number_of_systems_on_this_page


def create_engraving_statistics_from_individual_files(dataset_directory, engraving_type):
    if len(glob(f'{dataset_directory}/**/*.json', recursive=True)) == 0:
        print(f"Could not find json files in {dataset_directory} directory.")

    number_of_systems = [0] * 15  # type: Dict[int, int]
    individual_results = []
    for dataset in tqdm(glob(f'{dataset_directory}/**/*.json', recursive=True), desc="Computing from individual files"):
        with open(dataset, "r") as file:
            image_annotations = json.load(file)
        system_measures = image_annotations["system_measures"]
        number_of_systems_on_this_page = compute_number_of_system_on_page(system_measures)
        number_of_systems[number_of_systems_on_this_page] += 1
        individual_results.append((engraving_type, number_of_systems_on_this_page))

    with open("dataset_statistics_muscima_pp.csv", "w") as statistics_file:
        statistics_file.write("Engraving Type,Number of Systems\n")
        for item in individual_results:
            statistics_file.write(item[0] + "," + str(item[1]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes dataset statistics for the given dataset')
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, where the extracted dataset resides")

    flags = parser.parse_args()
    create_engraving_statistics(flags.dataset_directory)
    create_engraving_statistics_from_individual_files(flags.dataset_directory + "/muscima_pp", "lined")
