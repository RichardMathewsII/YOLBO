def detect_objects_in_video(video_path, model, labels_to_names, video_output_name, output="video", fps=30, frames=None, yolbo=False):
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
                    cv2.imwrite('frame'+str(step)+'.jpg', draw)
            step += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
