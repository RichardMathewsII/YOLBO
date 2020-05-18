from numpy import array
from cv2 import FONT_HERSHEY_PLAIN
from cv2 import getTextSize
from cv2 import rectangle
from cv2 import LINE_AA
from cv2 import putText


def draw_label(frame, label, box, color):

    b = array(box).astype(int)
    font = FONT_HERSHEY_PLAIN
    (width, height), baseline = getTextSize(label, font, 1, 1)
    rectangle(frame, (b[0], b[1] - height - baseline - 5), (b[0] + width + 5, b[1]), color, -1, LINE_AA)
    putText(frame, label, (b[0] + 3, b[1] - baseline), font, 1, (0, 0, 0), 1)


def annotate_frame(frame, frame_number, frame_height, num_boxes, yolbo=False):

    putText(frame, 'Frame: ' + str(frame_number), (20, 40), FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
    putText(frame, 'Frame: ' + str(frame_number), (20, 40), FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    putText(frame, str(num_boxes) + ' Boxes', (20, 90), FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
    putText(frame, str(num_boxes) + ' Boxes', (20, 90), FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    if yolbo:
        putText(frame, 'Y.O.L.B.O.', (20, frame_height - 20), FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
        putText(frame, 'Y.O.L.B.O.', (20, frame_height - 20), FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    else:
        putText(frame, 'RetinaNet', (20, frame_height - 20), FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
        putText(frame, 'RetinaNet', (20, frame_height - 20), FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
