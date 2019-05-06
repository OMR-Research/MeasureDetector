import argparse
import json
import os
import pickle
from glob import glob
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import inference_over_image
from object_detection.utils import visualization_utils as vis_util


def run_inference_for_image_batch(images_np, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    image_batch = np.concatenate(images_np, axis=0)
    output_dicts = sess.run(tensor_dict, feed_dict={image_tensor: image_batch})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dicts['num_detections'] = output_dicts['num_detections'].astype(np.uint8)
    output_dicts['detection_classes'] = output_dicts['detection_classes'].astype(np.uint8)

    return output_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--inference_graph', dest='inference_graph', type=str, required=True,
                        help='Path to the frozen inference graph.')
    parser.add_argument('--label_map', dest='label_map', type=str, default="mapping.txt",
                        help='Path to the label map, which is json-file that maps each category name '
                             'to a unique number.')
    parser.add_argument('--input_directory', dest='input_directory', type=str, required=True,
                        help='Path to the directory that contains the images for which object detection should be performed')
    parser.add_argument('--output_directory', dest='output_directory', type=str, default='detection_output',
                        help='Path to the output directory, that will contain the results.')
    parser.add_argument('--show_scores', dest='show_scores', type=bool, default=True)
    parser.add_argument('--show_labels', dest='show_labels', type=bool, default=True)
    parser.add_argument('--max_batch_size', dest='max_batch_size', type=int, default=15,
                        help='Maximum number of images per batch. Only images of the same dimensions will be batched.'
                             'If images have different dimensions, they will be processed individually.')
    parser.add_argument('--score_threshold', dest='score_threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Uncomment the next line on Windows to run the evaluation on the CPU, despite a present GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_frozen_inference_graph = args.inference_graph
    path_to_labels = args.label_map
    number_of_classes = 999999
    input_image_directory = args.input_directory
    output_directory = args.output_directory
    show_scores = args.show_scores
    show_labels = args.show_labels
    score_threshold = args.score_threshold
    max_batch_size = args.max_batch_size

    start_time = time()

    # Read frozen graph
    detection_graph = inference_over_image.load_detection_graph(path_to_frozen_inference_graph)
    category_index = inference_over_image.load_category_index(path_to_labels, number_of_classes)

    jpg_files = glob(input_image_directory + "/**/*.jpg", recursive=True)
    png_files = glob(input_image_directory + "/**/*.png", recursive=True)
    input_files = jpg_files + png_files
    os.makedirs(output_directory, exist_ok=True)

    detection_start_time = time()

    with detection_graph.as_default():
        # Uncomment the following line to restrict GPU memory
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        gpu_options = tf.GPUOptions()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            index_for_batch = 0
            images_for_batch = []
            image_dimensions_for_batch = []
            input_files_for_batch = []
            detections_list = []
            images_have_same_dimensions_for_batching = True
            for input_file_index in tqdm(range(len(input_files)), desc="Detecting objects"):
                input_file = input_files[input_file_index]
                try:
                    image = Image.open(input_file).convert("RGB")
                    image_width, image_height = image.size
                    image_dimensions_for_batch.append(image.size)
                except:
                    # print("Can not read {0} as image. Skipping file".format(input_file))
                    continue

                image_np = np.array(image)
                images_for_batch.append(np.expand_dims(image_np, 0))
                input_files_for_batch.append(input_file)

                if input_file_index < len(input_files) - 1:
                    try:
                        image_dimensions_of_next_image = Image.open(input_files[input_file_index + 1]).size
                        if image.size != image_dimensions_of_next_image:
                            images_have_same_dimensions_for_batching = False
                    except:
                        images_have_same_dimensions_for_batching = False

                index_for_batch += 1
                if index_for_batch < max_batch_size and images_have_same_dimensions_for_batching:
                    continue
                number_of_images_in_batch = index_for_batch
                index_for_batch = 0
                images_have_same_dimensions_for_batching = True

                outputs_dict = run_inference_for_image_batch(images_for_batch, sess, tensor_dict)

                for index in range(0, number_of_images_in_batch):
                    per_file_detections_list = []
                    image_np = np.squeeze(images_for_batch[index], axis=0)
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        outputs_dict['detection_boxes'][index],
                        outputs_dict['detection_classes'][index],
                        outputs_dict['detection_scores'][index],
                        category_index,
                        instance_masks=outputs_dict.get('detection_masks')[index] if outputs_dict.get(
                            'detection_masks') is not None else None,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                        skip_scores=not show_scores,
                        skip_labels=not show_labels,
                        min_score_thresh=score_threshold)

                    input_file_name, extension = os.path.splitext(os.path.basename(input_files_for_batch[index]))
                    output_file = os.path.join(output_directory, "{0}_detection{1}".format(input_file_name, extension))
                    pil_image = Image.fromarray(image_np)
                    width = pil_image.width
                    height = pil_image.height
                    pil_image.save(output_file)

                    boxes = outputs_dict['detection_boxes'][index]
                    classes = outputs_dict['detection_classes'][index]
                    scores = outputs_dict['detection_scores'][index]

                    # Currently, the dataset only has system measure annotation, so we leave the other two types empty
                    system_measures = []
                    stave_measures = []
                    staves = []
                    json_filename = input_file_name + ".json"
                    json_path = os.path.join(output_directory, json_filename)

                    for i in range(len(boxes)):
                        top, left, bottom, right = tuple(list(boxes[i]))
                        class_name = category_index[classes[i]]['name']
                        score = scores[i]
                        top *= image_height
                        left *= image_width
                        bottom *= image_height
                        right *= image_width
                        if score >= score_threshold:
                            detections_list.append(
                                [input_file_name + extension, top, left, bottom, right, class_name, score])
                            per_file_detections_list.append(
                                [input_file_name + extension, top, left, bottom, right, class_name, score])
                            if class_name == "system_measure":
                                system_measures.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
                            elif class_name == "stave_measure":
                                stave_measures.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
                            elif class_name == "stave":
                                staves.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})

                    detections = pd.DataFrame(data=per_file_detections_list,
                                              columns=["image_name", "top", "left", "bottom", "right", "class_name",
                                                       "confidence"])
                    output_csv_file = os.path.join(output_directory, "{0}_detection.csv".format(input_file_name))
                    detections.to_csv(output_csv_file, index=False, float_format="%.2f")
                    with open(json_path, 'w') as file:
                        json.dump({'width': width, 'height': height, 'system_measures': system_measures,
                                   'stave_measures': stave_measures, 'staves': staves}, file, indent=4)

                images_for_batch = []
                image_dimensions_for_batch = []
                input_files_for_batch = []

    detections = pd.DataFrame(data=detections_list,
                              columns=["image_name", "top", "left", "bottom", "right", "class_name",
                                       "confidence"])
    detections.to_csv(os.path.join(output_directory, "detections.csv"), index=False, float_format="%.2f")

    end_time = time()

    print("Total execution time: {0:.0f} seconds".format(end_time - start_time))
    print("Execution time without initialization: {0:.0f} seconds".format(end_time - detection_start_time))
