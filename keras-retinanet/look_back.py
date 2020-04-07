import numpy as np

def bb_center(box):

    b = box.astype(int)
    width = b[2] - b[0]
    height = b[3] - b[1]
    center_y = b[1] + height//2
    center_x = b[0] + width//2
    return center_x, center_y, width, height


def centerBox(box):

    center_x, center_y, width, height = bb_center(box)
    centerBoxHeight = height//5
    centerBoxWidth = width//5
    cb = np.zeros(4)
    cb[0] = center_x - centerBoxWidth//2
    cb[1] = center_y - centerBoxHeight//2
    cb[2] = center_x + centerBoxWidth//2
    cb[3] = center_y + centerBoxHeight//2
    return cb


def probability_matrix(boxes, scores, labels, frame_width, frame_height, num_labels):

    matrix = np.zeros((num_labels, frame_height, frame_width))
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        center_x, center_y, _, _ = bb_center(box)
        matrix[label, center_y, center_x] = score
    return matrix


def look_back(box, label, previous_probability_matrix):

    cb = centerBox(box)
    cb = cb.astype(int)
    previous_probabilities = []
    for x in range(cb[0], cb[2] + 1):
        for y in range(cb[1], cb[3] + 1):
            previous_probabilities.append(previous_probability_matrix[label, y, x])
    best_previous_score = max(previous_probabilities)
    return best_previous_score

