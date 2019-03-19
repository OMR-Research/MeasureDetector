import argparse
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


def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


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
    args = parser.parse_args()

    # Uncomment the next line on Windows to run the evaluation on the CPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_frozen_inference_graph = args.inference_graph
    path_to_labels = args.label_map
    number_of_classes = 999999
    input_image_directory = args.input_directory
    output_directory = args.output_directory
    show_scores = args.show_scores
    show_labels = args.show_labels

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
        with tf.Session() as sess:
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

            detections_list = []
            for input_file in tqdm(input_files, desc="Detecting objects"):
                per_file_detections_list = []
                try:
                    image = Image.open(input_file).convert("RGB")
                    image_width, image_height = image.size
                except:
                    # print("Can not read {0} as image. Skipping file".format(input_file))
                    continue

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = inference_over_image.load_image_into_numpy_array(image)

                output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    skip_scores=not show_scores,
                    skip_labels=not show_labels)

                input_file_name, extension = os.path.splitext(os.path.basename(input_file))
                output_file = os.path.join(output_directory, "{0}_detection{1}".format(input_file_name, extension))
                Image.fromarray(image_np).save(output_file)

                boxes = output_dict['detection_boxes']
                classes = output_dict['detection_classes']
                scores = output_dict['detection_scores']

                for i in range(len(boxes)):
                    top, left, bottom, right = tuple(list(boxes[i]))
                    class_name = category_index[classes[i]]['name']
                    score = scores[i]
                    top *= image_height
                    left *= image_width
                    bottom *= image_height
                    right *= image_width
                    if score >= 0.5:
                        detections_list.append(
                            [input_file_name + extension, top, left, bottom, right, class_name, score])
                        per_file_detections_list.append(
                            [input_file_name + extension, top, left, bottom, right, class_name, score])

                detections = pd.DataFrame(data=per_file_detections_list,
                                          columns=["image_name", "top", "left", "bottom", "right", "class_name",
                                                   "confidence"])
                output_csv_file = os.path.join(output_directory, "{0}_detection.csv".format(input_file_name))
                detections.to_csv(output_csv_file, index=False, float_format="%.2f")

            detections = pd.DataFrame(data=detections_list,
                                      columns=["image_name", "top", "left", "bottom", "right", "class_name",
                                               "confidence"])
            detections.to_csv(os.path.join(output_directory, "detections.csv"), index=False, float_format="%.2f")

            end_time = time()

            print("Total execution time: {0:.0f}s".format(end_time - start_time))
            print("Execution time without initialization: {0:.0f}s".format(end_time - detection_start_time))
