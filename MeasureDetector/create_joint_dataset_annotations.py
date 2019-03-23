import argparse
import json
import os
import random
from glob import glob

from tqdm import tqdm
from typing import Dict, List

from MeasureDetector.create_dataset_statistics import compute_number_of_system_on_page

engraving_type_mapping = {"print": "typeset",
                          "printed": "typeset",
                          "exported": "typeset",
                          "handwritten": "handwritten",
                          "lined": "handwritten",
                          "mixed": "handwritten"}

joint_dataset = {
    "handwritten": {"0": [], "1": [], "2": [], "3": [], "more": []},
    "typeset": {"0": [], "1": [], "2": [], "3": [], "more": []}
}

training_dataset = {
    "handwritten": {"0": [], "1": [], "2": [], "3": [], "more": []},
    "typeset": {"0": [], "1": [], "2": [], "3": [], "more": []}
}

validation_dataset = {
    "handwritten": {"0": [], "1": [], "2": [], "3": [], "more": []},
    "typeset": {"0": [], "1": [], "2": [], "3": [], "more": []}
}

test_dataset = {
    "handwritten": {"0": [], "1": [], "2": [], "3": [], "more": []},
    "typeset": {"0": [], "1": [], "2": [], "3": [], "more": []}
}


def load_from_dataset_json_files(dataset_directory: str):
    if len(glob(f'{dataset_directory}/**/dataset.json', recursive=True)) == 0:
        print(f"Could not find json files in {dataset_directory} directory.")

    for dataset in tqdm(glob(f'{dataset_directory}/**/dataset.json', recursive=True), desc="Converting annotations"):
        with open(dataset, "r") as file:
            dataset_annotations = json.load(file)

        root_directory = dataset_annotations["root_dir"]

        for name, source in dataset_annotations["sources"].items():
            source_directory = source["root_dir"]
            engraving_type = source["type"]
            mapped_engraving_type = engraving_type_mapping[engraving_type]

            for page in source["pages"]:
                system_measures = []
                width = page["width"]
                height = page["height"]

                for measure in page["annotations"]["measures"]:
                    left = int(measure["bbox"]["x"])
                    top = int(measure["bbox"]["y"])
                    right = int(measure["bbox"]["x"] + measure["bbox"]["width"])
                    bottom = int(measure["bbox"]["y"] + measure["bbox"]["height"])

                    data = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
                    system_measures.append(data)

                page_path = os.path.join(dataset_directory, root_directory, source_directory, page["image"])
                number_of_systems_on_this_page = compute_number_of_system_on_page(system_measures)
                mapped_number_of_systems_on_this_page = map_number_of_systems(number_of_systems_on_this_page)

                page_annotations = {"path": page_path,
                                    "width": width,
                                    "height": height,
                                    "system_measures": system_measures}

                joint_dataset[mapped_engraving_type][mapped_number_of_systems_on_this_page].append(page_annotations)


def map_number_of_systems(number_of_system_on_this_page) -> str:
    if number_of_system_on_this_page > 3:
        return "more"
    else:
        return str(number_of_system_on_this_page)


def append_muscima_pp_dataset(dataset_directory):
    if len(glob(f'{dataset_directory}/**/*.json', recursive=True)) == 0:
        print(f"Could not find json files in {dataset_directory} directory.")

    engraving_type = "lined"
    mapped_engraving_type = engraving_type_mapping[engraving_type]

    for dataset in tqdm(glob(f'{dataset_directory}/**/*.json', recursive=True), desc="Computing from individual files"):
        with open(dataset, "r") as file:
            image_annotations = json.load(file)
        system_measures = image_annotations["system_measures"]
        number_of_systems_on_this_page = compute_number_of_system_on_page(system_measures)
        mapped_number_of_systems_on_this_page = map_number_of_systems(number_of_systems_on_this_page)

        page_annotations = {"path": dataset.replace("json", "png"),
                            "width": image_annotations["width"],
                            "height": image_annotations["height"],
                            "system_measures": system_measures}

        joint_dataset[mapped_engraving_type][mapped_number_of_systems_on_this_page].append(page_annotations)


def get_random_sample_indices(dataset_size: int,
                              validation_percentage: float,
                              test_percentage: float) -> (
        List[int], List[int], List[int]):
    """
    Returns a set of random sample indices from the entire dataset population
    :param dataset_size: The population size
    :param validation_percentage: the percentage of the entire population size that should be used for validation
    :param test_percentage: the percentage of the entire population size that should be used for testing
    :param seed: An arbitrary seed that can be used to obtain repeatable pseudo-random indices
    :return: A triple of three list, containing indices of the training, validation and test sets
    """
    all_indices = range(0, dataset_size)
    validation_sample_size = int(dataset_size * validation_percentage)
    test_sample_size = int(dataset_size * test_percentage)
    validation_sample_indices = random.sample(all_indices, validation_sample_size)
    test_sample_indices = random.sample((set(all_indices) - set(validation_sample_indices)), test_sample_size)
    training_sample_indices = list(set(all_indices) - set(validation_sample_indices) - set(test_sample_indices))
    return training_sample_indices, validation_sample_indices, test_sample_indices


def split_dataset_annotations_into_train_validation_test(validation_percentage: float = 0.1,
                                                         test_percentage: float = 0.1):
    random.seed(0)  # To make sampling reproducible

    for engraving_type, values in joint_dataset.items():
        for number_of_staves, samples in values.items():
            number_of_images_in_class = len(samples)
            training_sample_indices, validation_sample_indices, test_sample_indices = \
                get_random_sample_indices(number_of_images_in_class, validation_percentage, test_percentage)

            training_samples = [samples[i] for i in training_sample_indices]
            validation_samples = [samples[i] for i in validation_sample_indices]
            test_samples = [samples[i] for i in test_sample_indices]

            training_dataset[engraving_type][number_of_staves].extend(training_samples)
            validation_dataset[engraving_type][number_of_staves].extend(validation_samples)
            test_dataset[engraving_type][number_of_staves].extend(test_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates one large json-file for the entire available dataset')
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, where the extracted dataset resides. Directories will be crawled for "
                             "files called dataset.json. The directory for MUSCIMA++ is assumed 'muscima_pp' and "
                             "annotations will be loaded on a per-file base instead of from a ")

    flags = parser.parse_args()
    dataset_directory = flags.dataset_directory
    load_from_dataset_json_files(dataset_directory)
    append_muscima_pp_dataset(os.path.join(dataset_directory, "muscima_pp"))

    for engraving_type, values in joint_dataset.items():
        for number_of_staves, samples in values.items():
            print("Collected {0} samples of {1} engraving with {2} staves".format(len(samples), engraving_type,
                                                                                  number_of_staves))

    split_dataset_annotations_into_train_validation_test()

    json_path = os.path.join(dataset_directory, "joint_dataset_annotations.json")
    with open(json_path, 'w') as file:
        json.dump(joint_dataset, file, indent=4)

    json_path = os.path.join(dataset_directory, "training_joint_dataset.json")
    with open(json_path, 'w') as file:
        json.dump(training_dataset, file, indent=4)

    json_path = os.path.join(dataset_directory, "validation_joint_dataset.json")
    with open(json_path, 'w') as file:
        json.dump(validation_dataset, file, indent=4)

    json_path = os.path.join(dataset_directory, "test_joint_dataset.json")
    with open(json_path, 'w') as file:
        json.dump(test_dataset, file, indent=4)
