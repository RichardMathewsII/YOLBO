import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from look_back import *


def run_yolbo(video_path, model, labels_to_names, video_output_name, output="video", fps=30, frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return print("Error opening video file")
    if output == "frames":
        assert frames is not None, 'Provide frame numbers to return'
    assert type(labels_to_names) == dict, 'labels_to_names parameter should be of type: dict'

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(video_output_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (frame_width, frame_height))
    step = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # copy to draw on
            draw = frame.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            image = preprocess_image(frame)
            image, scale = resize_image(image)

            # process image
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            num_boxes = 0
            # correct for image scale
            boxes /= scale
            num_labels = len(labels_to_names)
            matrix = np.zeros((num_labels, frame_height, frame_width))
            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                center_x, center_y, _, _ = bb_center(box)
                matrix[label, center_y, center_x] = score

                if score < 0.5 and step == 1:
                    break
                if score < 0.3:
                    break
                if (score >= 0.3) and (score < 0.5):
                    best_previous_score = look_back(box, label, previous_matrix)
                    if best_previous_score >= 0.5:
                        matrix[label, center_y, center_x] = best_previous_score
                    else:
                        break
                color = label_color(label)
                num_boxes += 1
                b = box.astype(int)
                draw_box(draw, b, color=color)

                b = np.array(box).astype(int)
                font = cv2.FONT_HERSHEY_PLAIN
                label_name = labels_to_names[label]
                (width, height), baseline = cv2.getTextSize(label_name, font, 1, 1)
                cv2.rectangle(draw, (b[0], b[1] - height - baseline - 5), (b[0] + width + 5, b[1]), color, -1,
                              cv2.LINE_AA)
                cv2.putText(draw, label_name, (b[0] + 3, b[1] - baseline), font, 1, (0, 0, 0), 1)
            cv2.putText(draw, 'Frame: ' + str(step), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
            cv2.putText(draw, 'Frame: ' + str(step), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
            cv2.putText(draw, str(num_boxes) + ' Boxes', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
            cv2.putText(draw, str(num_boxes) + ' Boxes', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
            cv2.putText(draw, 'Y.O.L.B.O.', (20, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
            cv2.putText(draw, 'Y.O.L.B.O.', (20, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            previous_matrix = matrix
            if output == 'video':
                out.write(draw)
            if output == 'frames':
                if step in frames:
                    cv2.imwrite('frame'+str(step)+'.jpg', draw)
            step += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
