import numpy as np


def dataloader(vehicle_objects, window_size, shift):
    # print("dataset length: {}".format(vehicle_objects.__len__()))
    num_of_parameters = 3
    tensor_idx = 0
    total_size = 572
    N = int(window_size / shift)

    lane_change_tensor = np.zeros((total_size, num_of_parameters, window_size))
    lane_keeping_tensor = np.zeros((total_size, num_of_parameters, window_size))

    features = np.zeros((num_of_parameters, window_size))
    tt = np.zeros((num_of_parameters, window_size))

    left_seq = []
    right_seq = []
    keep_seq = []
    label_sequences = []
    data = []
    left = [1., 0., 0.]
    right = [0., 0., 1.]
    keep = [0., 1., 0.]
    # print(lane_change_tensor.shape)
    # print(lane_keeping_tensor.shape)
    # print(features.shape)
    # print(tt.shape)

    for vehicle in vehicle_objects:
        # print("Vehicle: {}, size: {}".format(vehicle.id, vehicle.size))
        lane_change_idx, label = lane_change_to_idx(vehicle)

        if (lane_change_idx - 1) > 2 * window_size:
            batch = []
            for k in range(N):

                features[0] = vehicle.x[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]\
                              - vehicle.x[lane_change_idx - window_size - k * shift: lane_change_idx - k * shift]

                features[1] = vehicle.v[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]
                features[2] = vehicle.a[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]

                batch.append(features)

            if label == -1:
                left_seq.append(batch)
            else:
                right_seq.append(batch)

        elif lane_change_idx == 0:
            batch = []
            for k in range(N):
                features[0] = vehicle.x[lane_change_idx + 1 + k * shift: lane_change_idx + 1 + k * shift + window_size]\
                              - vehicle.x[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[1] = vehicle.v[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[2] = vehicle.a[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]

                batch.append(features)

            keep_seq.append(batch)

    lab = []
    for i in range(N):
        lab.append(left)
    for i in range(N):
        lab.append(right)
    for i in range(N):
        lab.append(keep)

    for l, r, k in zip(left_seq, right_seq, keep_seq):
        batch = np.concatenate((l, r, k), axis=0)
        data.append(batch)
        label_sequences.append(lab)

    data = np.array(data).transpose((0, 1, 3, 2))
    label_sequences = np.array(label_sequences)
    return data, label_sequences


def lane_change_to_idx(vehicle):
    j = 0
    labels = 0
    lane_change_idx = 0

    while (j < vehicle.size - 1) & (lane_change_idx == 0):
        delta = vehicle.lane_id[j + 1] - vehicle.lane_id[j]
        if delta != 0:
            lane_change_idx = j
            labels = delta
            # print("Lane change idx: {}".format(lane_change_idx))
        j = j + 1

    return lane_change_idx, labels