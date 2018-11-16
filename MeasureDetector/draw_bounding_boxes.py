import json
import cv2
import argparse


def draw_bounding_boxes_into_image(image_path: str, ground_truth_annotations_path: str, destination_path: str):
    img = cv2.imread(image_path, True)

    with open(ground_truth_annotations_path, 'r') as gt_file:
        data = json.load(gt_file)

    draw_rectangles(data["system_measures"], img, (255, 0, 0), -1, 0.4)
    draw_rectangles(data["stave_measures"], img, (255, 0, 255), -1, 0.4)
    draw_rectangles(data["staves"], img, (0, 255, 255), -1, 0.4)

    cv2.imwrite(destination_path, img)


def draw_rectangles(rectangles, image, color, line_thickness, alpha):
    for rectangle in rectangles:
        left, top, bottom, right = rectangle["left"], rectangle["top"], rectangle["bottom"], \
                                   rectangle["right"]

        # String to float, float to int
        left = int(float(left))
        top = int(float(top))
        bottom = int(float(bottom))
        right = int(float(right))

        overlay = image.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, line_thickness)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding boxes from the ground-truth data.')
    parser.add_argument('-img', dest='img_path', type=str, required=True, help='Path to the image.')
    parser.add_argument('-gt', dest='gt_path', type=str, required=True, help='Path to the ground truth.')
    parser.add_argument('-save', dest='save_img', type=str, required=True, help='Path to save the processed image.')
    args = parser.parse_args()

    draw_bounding_boxes_into_image(args.img_path, args.gt_path, args.save_img)
