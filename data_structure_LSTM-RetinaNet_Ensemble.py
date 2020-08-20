"""
This file is the implementation of the data structure proposed in the paper, section 5, for training an LSTM-RetinaNet
Ensemble. This file is not related to YOLBO and should be ignored to anyone working with the YOLBO-RetinaNet model
for object detection in video data.
"""


def preprocess_data(video_file, num_labels):
    """
    transforms video into data to feed into deep RNN
    """
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    i = 0
    row_length = int((frame_height / 10) * (frame_width / 10))
    data_structure = np.zeros((1, 3, row_length))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and (i%10==0):
            i += 1
            # preprocess image for network
            image = preprocess_image(frame)
            image, scale = resize_image(image)

            # process image
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            # print("processing time: ", time.time() - start)
            num_boxes = 0
            # correct for image scale
            boxes /= scale
            encoded_matrix = np.zeros((num_labels, frame_height, frame_width, 1))
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                center_x, center_y, width, height = bb_center(box)
                if score > 0.4:
                    encoded_matrix[label, center_y, center_x] = width * height
            matrix = np.zeros((3, frame_height, frame_width, 1))
            # people movement
            matrix[0] = encoded_matrix[0] + encoded_matrix[24] + encoded_matrix[26] + encoded_matrix[28]
            # vehicle movement
            matrix[1] = encoded_matrix[2] + encoded_matrix[3] + encoded_matrix[5] + encoded_matrix[7]
            # stationary object movement
            matrix[2] = encoded_matrix[9] + encoded_matrix[10] + encoded_matrix[11] + encoded_matrix[13]
            pooled_matrix = tf.nn.max_pool2d(matrix, [1, 10, 10, 1], [1, 10, 10, 1], padding='SAME', data_format='NHWC')
            row_length = int((frame_height / 10) * (frame_width / 10))
            data_structure_at_t = tf.reshape(pooled_matrix, [3, row_length])
            ds = np.array([tf.Session().run(data_structure_at_t)])
            data_structure = np.append(data_structure, ds, axis=0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif ret:
            i += 1
            continue
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return data_structure


def generate_training_data(data_structure, movement):
    movement_options = ['PEOPLE', 'VEHICLE', 'STATIONARY']
    if movement not in movement_options:
        print('Movement parameter must be one of the following: \'PEOPLE\', \'VEHICLE\', \'STATIONARY\'')
        return None

    timesteps, movements, locations = data_structure.shape
    # ensures a label can be generated for last sequence in batch
    # labels are 30 timesteps after last timestep in sequence
    total_steps = int(timesteps)
    timesteps -= 30
    # 2 seconds per batch (30 frames per second)
    batch_size = (timesteps // 60) + ((timesteps - 30) // 60)
    stop = (timesteps // 60) * 60
    # second sequence generator starting at t = 30
    stop2 = 30 + ((timesteps - 30) // 60) * 60
    counter = 0  # counter for label vector
    counter2 = (timesteps // 60) * 4

    if movement == 'PEOPLE':
        people_movement = np.zeros((batch_size, 60, locations))
        plabels = np.zeros((batch_size * 4, locations))
        for t in range(total_steps):
            seq = t // 60
            # sequences starting at t=30
            seq2 = ((t - 30) // 60) + (timesteps // 60)
            if t < stop:
                idx1 = t - (60 * seq)
                people_movement[seq, idx1] = data_structure[t, 0]
            if t < stop2 and t >= 30:
                idx2 = t - 60 * ((t - 30) // 60) - 30
                people_movement[seq2, idx2] = data_structure[t, 0]
            if (t - 29) % 60 == 0:
                plabels[counter] = data_structure[t, 0]
                counter += 1
                plabels[counter] = data_structure[t + 1, 0]
                counter += 1
                plabels[counter] = data_structure[t + 2, 0]
                counter += 1
                plabels[counter] = data_structure[t + 3, 0]
                counter += 1
            if (t + 1) % 60 == 0 and t > 60:
                plabels[counter2] = data_structure[t, 0]
                counter2 += 1
                plabels[counter2] = data_structure[t + 1, 0]
                counter2 += 1
                plabels[counter2] = data_structure[t + 2, 0]
                counter2 += 1
                plabels[counter2] = data_structure[t + 3, 0]
                counter2 += 1
        plabels = np.reshape(plabels, (1, batch_size * 4, locations, 1))
        pooled_plabels = tf.nn.max_pool2d(plabels, [1, 4, 1, 1], [1, 4, 1, 1], padding='SAME', data_format='NHWC')
        return people_movement, pooled_plabels, batch_size

    if movement == 'VEHICLE':
        vehicle_movement = np.zeros((batch_size, 60, locations))
        vlabels = np.zeros((batch_size * 4, locations))
        for t in range(total_steps):
            seq = t // 60
            # sequences starting at t=30
            seq2 = ((t - 30) // 60) + (timesteps // 60)
            if t < stop:
                idx1 = t - (60 * seq)
                vehicle_movement[seq, idx1] = data_structure[t, 1]
            if t < stop2 and t >= 30:
                idx2 = t - 60 * ((t - 30) // 60) - 30
                vehicle_movement[seq2, idx2] = data_structure[t, 1]
            if (t - 29) % 60 == 0:
                vlabels[counter] = data_structure[t, 1]
                counter += 1
                vlabels[counter] = data_structure[t + 1, 1]
                counter += 1
                vlabels[counter] = data_structure[t + 2, 1]
                counter += 1
                vlabels[counter] = data_structure[t + 3, 1]
                counter += 1
            if (t + 1) % 60 == 0 and t > 60:
                vlabels[counter2] = data_structure[t, 1]
                counter2 += 1
                vlabels[counter2] = data_structure[t + 1, 1]
                counter2 += 1
                vlabels[counter2] = data_structure[t + 2, 1]
                counter2 += 1
                vlabels[counter2] = data_structure[t + 3, 1]
                counter2 += 1
        vlabels = np.reshape(vlabels, (1, batch_size * 4, locations, 1))
        pooled_vlabels = tf.nn.max_pool2d(vlabels, [1, 4, 1, 1], [1, 4, 1, 1], padding='SAME', data_format='NHWC')
        return vehicle_movement, pooled_vlabels, batch_size

    if movement == 'STATIONARY':
        stationary_movement = np.zeros((batch_size, 60, locations))
        slabels = np.zeros((batch_size * 4, locations))
        for t in range(total_steps):
            seq = t // 60
            # sequences starting at t=30
            seq2 = ((t - 30) // 60) + (timesteps // 60)
            if t < stop:
                idx1 = t - (60 * seq)
                stationary_movement[seq, idx1] = data_structure[t, 2]
            if t < stop2 and t >= 30:
                idx2 = t - 60 * ((t - 30) // 60) - 30
                stationary_movement[seq2, idx2] = data_structure[t, 2]
            if (t - 29) % 60 == 0:
                slabels[counter] = data_structure[t, 2]
                counter += 1
                slabels[counter] = data_structure[t + 1, 2]
                counter += 1
                slabels[counter] = data_structure[t + 2, 2]
                counter += 1
                slabels[counter] = data_structure[t + 3, 2]
                counter += 1
            if (t + 1) % 60 == 0 and t > 60:
                slabels[counter2] = data_structure[t, 2]
                counter2 += 1
                slabels[counter2] = data_structure[t + 1, 2]
                counter2 += 1
                slabels[counter2] = data_structure[t + 2, 2]
                counter2 += 1
                slabels[counter2] = data_structure[t + 3, 2]
                counter2 += 1
        slabels = np.reshape(slabels, (1, batch_size * 4, locations, 1))
        pooled_slabels = tf.nn.max_pool2d(slabels, [1, 4, 1, 1], [1, 4, 1, 1], padding='SAME', data_format='NHWC')
        return stationary_movement, pooled_slabels, batch_size