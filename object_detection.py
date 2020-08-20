"""
Performs object detection on single frame
"""
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
    """
    Runs the YOLBO algorithm
    """
    num_boxes = 0
    # initialize detection matrix
    detection_matrix = zeros((num_labels, frame_height, frame_width))

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        # insert score into detection matrix
        center_x, center_y, _, _ = bb_center(box)
        detection_matrix[label, center_y, center_x] = score

        if score < 0.5 and step == 1:
            # it is not possible to look at the previous frame on the first frame
            break
        if score < 0.3:
            # a score less than 0.3 is too low and means the model is not confident in the detection at all
            break
        if (score >= 0.3) and (score < 0.5):
            # the model is "unsure" about this detection, not confident enough, so look back at the previous frame
            # to check if there are similar detections that it was confident on
            best_previous_score = look_back(box, label, previous_matrix)
            if best_previous_score >= 0.5:
                # the model was confident on a similar detection in the prior frame, so we can assume this detection
                # is actually valid
                detection_matrix[label, center_y, center_x] = best_previous_score  # update detection matrix
            else:
                # there were no similar detections of high confidence
                break
        color = label_color(label)  # get color for specific label
        num_boxes += 1  # count number of detections
        b = box.astype(int)
        draw_box(frame, b, color=color)  # draw bounding box on frame
        label_name = labels_to_names[label]  # get name associated with integer label
        draw_label(frame, label_name, box, color)  # draw label on bounding box

    return frame, detection_matrix, num_boxes


def run_retinanet(model, frame, step, frame_height, frame_width, labels_to_names, yolbo, previous_matrix):
    """
    Performs object detection on frame using RetinaNet
    """
    if yolbo is True:
        # run with yolbo
        # copy to draw on
        draw = frame.copy()
        draw = cvtColor(draw, COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(frame)
        image, scale = resize_image(image)

        # process image, make predictions
        boxes, scores, labels = model.predict_on_batch(expand_dims(image, axis=0))

        # correct for image scale
        boxes /= scale

        # run yolbo algo, get detection matrix and frame with annotated detections
        num_labels = len(labels_to_names)
        draw, detection_matrix, num_boxes = run_yolbo(draw, boxes, scores, labels, labels_to_names, step,
                                                      previous_matrix, num_labels, frame_height, frame_width)
        # additional annotations on frame including frame # and number of visualized detections in frame
        annotate_frame(draw, step, frame_height, num_boxes, yolbo=True)

        return draw, detection_matrix

    else:
        # run without yolbo
        # copy to draw on
        draw = frame.copy()
        draw = cvtColor(draw, COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(frame)
        image, scale = resize_image(image)

        # process image, make predictions
        boxes, scores, labels = model.predict_on_batch(expand_dims(image, axis=0))
        num_boxes = 0

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                # crossed the threshold, detections no longer confident
                break
            color = label_color(label)
            num_boxes += 1
            b = box.astype(int)
            draw_box(draw, b, color=color)
            label_name = labels_to_names[label]
            draw_label(draw, label_name, box, color)

        annotate_frame(draw, step, frame_height, num_boxes)

        return draw
