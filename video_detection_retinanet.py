import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from visualization import *


def run_retinanet(video_path, model, labels_to_names, video_output_name, output="video", fps=30, frames=None):
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
