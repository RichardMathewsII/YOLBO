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
    centerBoxHeight = height//3
    centerBoxWidth = width//3
    cb = np.zeros(4)
    cb[0] = center_x - centerBoxWidth//2
    cb[1] = center_y - centerBoxHeight//2
    cb[2] = center_x + centerBoxWidth//2
    cb[3] = center_y + centerBoxHeight//2
    return cb


def look_back(box, label, previous_matrix):

    cb = centerBox(box)
    cb = cb.astype(int)
    previous_scores = []
    for x in range(cb[0], cb[2] + 1):
        for y in range(cb[1], cb[3] + 1):
            previous_scores.append(previous_matrix[label, y, x])
    best_previous_score = max(previous_scores)
    return best_previous_score

