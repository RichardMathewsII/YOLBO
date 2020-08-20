"""
When constructing the detection matrix of the current timestep, YOLBO utilizes a Look Back function to
scan the detection matrix of the previous timestep for similar detections. This file is the implementation
of the Look Back function.
"""

from numpy import zeros


def bb_center(box):
    """
    Computes the pixel coordinates of the center of a bounding box
    :param box: list of pixel coordinates describing the corners of a bounding box [x1, y1, x2, y2]
    """
    b = box.astype(int)

    # compute width and height of bounding box in pixel units
    width = b[2] - b[0]
    height = b[3] - b[1]

    # compute center of the bounding box in pixel units
    center_y = b[1] + height//2
    center_x = b[0] + width//2

    return center_x, center_y, width, height


def centerBox(box, cb_ratio=(1/3)):
    """
    Computes a box surrounding the center of a bounding box that has side lengths proportional to the bounding box
    :param box: list of pixel coordinates describing the corners of a bounding box [x1, y1, x2, y2]
    :param cb_ratio: the ratio of the center box size to the bounding box size
    :return: list of pixel coordinates describing the corners of a center box
    """
    # retrieve the center and dimensions of the bounding box
    center_x, center_y, width, height = bb_center(box)

    # compute the dimensions of the center box according to the center box ratio parameter
    centerBoxHeight = int(height * cb_ratio)
    centerBoxWidth = int(width * cb_ratio)

    # represent the center box as a list of the lower-left corner coordinates and upper-right corner coordinates
    cb = zeros(4)
    cb[0] = center_x - centerBoxWidth//2
    cb[1] = center_y - centerBoxHeight//2
    cb[2] = center_x + centerBoxWidth//2
    cb[3] = center_y + centerBoxHeight//2

    return cb


def look_back(box, label, previous_matrix):
    """
    Looks back at the previous frame for similar detections. Uses the center box as the scanning region.
    :param box: list of pixel coordinates describing the corners of a bounding box [x1, y1, x2, y2]
    :param label: the class of the detection as predicted by RetinaNet
    :param previous_matrix: the detection matrix from the previous frame (t-1)
    :return: the max score for a detection of class=label with a bounding box center within the scanning region
    """
    # compute the scanning region for the detection
    scanning_region = centerBox(box)
    scanning_region = scanning_region.astype(int)

    # generate a list of all the scores for class=label inside the scanning region of the previous detection matrix
    previous_scores = []
    for x in range(scanning_region[0], scanning_region[2] + 1):
        for y in range(scanning_region[1], scanning_region[3] + 1):
            previous_scores.append(previous_matrix[label, y, x])

    # return the highest score
    best_previous_score = max(previous_scores)
    return best_previous_score

