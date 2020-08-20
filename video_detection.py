"""
Perform object detection using RetinaNet on video with or without YOLBO
"""
from cv2 import VideoCapture
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
from cv2 import imwrite
from cv2 import waitKey
from cv2 import destroyAllWindows
from object_detection import *


def detect_objects_in_video(video_path, model, labels_to_names, video_output_name, output="video", fps=30, frames=None,
                            yolbo=False):
    """
    Perform object detection on video data
    :param video_path: file path to video (str)
    :param model: pretrained RetinaNet model
    :param labels_to_names: dictionary mapping integer labels to string names
    :param video_output_name: output video file (str) to be written
    :param output: either 'video' for annotated video or 'frames' for specific annotated frames
    :param fps: frames per second of video
    :param frames: list of integers representing the annotated frames to write (only if output='frames')
    :param yolbo: True to run YOLBO (defaults to False)
    :return: annotated video file (output='video'), or annotated image files (output='frames')
    """
    cap = VideoCapture(video_path)
    if not cap.isOpened():
        return print("Error opening video file")
    if output == "frames":
        assert frames is not None, 'Provide frame numbers to return'
    assert type(labels_to_names) == dict, 'labels_to_names parameter should be of type: dict'

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = VideoWriter(video_output_name, VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    step = 1
    detection_matrix = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if yolbo:
                draw, detection_matrix = run_retinanet(model, frame, step, frame_height, frame_width, labels_to_names,
                                                       yolbo, detection_matrix)
            else:
                draw = run_retinanet(model, frame, step, frame_height, frame_width, labels_to_names, yolbo, None)
            if output == 'video':
                out.write(draw)
            if output == 'frames':
                if step in frames:
                    imwrite('frame'+str(step)+'.jpg', draw)
            step += 1
            if waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    destroyAllWindows()
