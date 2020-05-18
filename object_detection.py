from look_back import *
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box
from visualization import *
from keras_retinanet.utils.image import preprocess_image, resize_image
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from numpy import expand_dims
from numpy import zeros


def run_yolbo(frame, boxes, scores, labels, labels_to_names, step, previous_matrix, num_labels, frame_height,
              frame_width):
    num_boxes = 0
    detection_matrix = zeros((num_labels, frame_height, frame_width))
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        center_x, center_y, _, _ = bb_center(box)
        detection_matrix[label, center_y, center_x] = score

        if score < 0.5 and step == 1:
            break
        if score < 0.3:
            break
        if (score >= 0.3) and (score < 0.5):
            best_previous_score = look_back(box, label, previous_matrix)
            if best_previous_score >= 0.5:
                detection_matrix[label, center_y, center_x] = best_previous_score
            else:
                break
        color = label_color(label)
        num_boxes += 1
        b = box.astype(int)
        draw_box(frame, b, color=color)
        label_name = labels_to_names[label]
        draw_label(frame, label_name, box, color)

    return frame, detection_matrix, num_boxes


def run_retinanet(model, frame, step, frame_height, frame_width, labels_to_names, yolbo, previous_matrix):

    if yolbo is True:
        # copy to draw on
        draw = frame.copy()
        draw = cvtColor(draw, COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(frame)
        image, scale = resize_image(image)

        # process image
        boxes, scores, labels = model.predict_on_batch(expand_dims(image, axis=0))

        # correct for image scale
        boxes /= scale
        num_labels = len(labels_to_names)
        draw, detection_matrix, num_boxes = run_yolbo(draw, boxes, scores, labels, labels_to_names, step,
                                                      previous_matrix, num_labels, frame_height, frame_width)
        annotate_frame(draw, step, frame_height, num_boxes)

        return draw, detection_matrix

    else:
        # copy to draw on
        draw = frame.copy()
        draw = cvtColor(draw, COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(frame)
        image, scale = resize_image(image)

        # process image
        boxes, scores, labels = model.predict_on_batch(expand_dims(image, axis=0))
        num_boxes = 0
        # correct for image scale
        boxes /= scale
        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break

            if score < 0.5:
                break
            color = label_color(label)
            num_boxes += 1
            b = box.astype(int)
            draw_box(draw, b, color=color)
            label_name = labels_to_names[label]
            draw_label(draw, label_name, box, color)

        annotate_frame(draw, step, frame_height, num_boxes)

        return draw
