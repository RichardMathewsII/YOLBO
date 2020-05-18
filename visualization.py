
def draw_label(frame, label, box, color):

    b = np.array(box).astype(int)
    font = cv2.FONT_HERSHEY_PLAIN
    (width, height), baseline = cv2.getTextSize(label, font, 1, 1)
    cv2.rectangle(frame, (b[0], b[1] - height - baseline - 5), (b[0] + width + 5, b[1]), color, -1,
                  cv2.LINE_AA)
    cv2.putText(frame, label, (b[0] + 3, b[1] - baseline), font, 1, (0, 0, 0), 1)


def annotate_frame(frame, frame_number, frame_height, num_boxes, yolbo=False):

    cv2.putText(frame, 'Frame: ' + str(frame_number), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
    cv2.putText(frame, 'Frame: ' + str(frame_number), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(frame, str(num_boxes) + ' Boxes', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
    cv2.putText(frame, str(num_boxes) + ' Boxes', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    if yolbo:
        cv2.putText(frame, 'Y.O.L.B.O.', (20, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
        cv2.putText(frame, 'Y.O.L.B.O.', (20, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    else:
        cv2.putText(frame, 'RetinaNet', (20, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
        cv2.putText(frame, 'RetinaNet', (20, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
