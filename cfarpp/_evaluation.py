import numpy as np


def gt_eval(self, scenario_name: str, frame_id: int):
    """Evaluation of the ground truth. Writes lines in the det/gt CSV files.

    Args:
        scenario_name (str): name of the scenario.
        frame_id (int): Frame ID to evaluate.
    """

    dist_matrix = np.zeros((self.num_gt, self.num_detections))

    # print(f"Ground truth: {self.gt[:self.num_gt, 2:4]}")
    # print(f"Detections: {self.detections[:self.num_detections, 2:4]}")

    if self.num_detections != 0 and self.num_gt != 0:
        diff_matrix = self.gt[:self.num_gt, np.newaxis, 2:4] - \
            self.detections[np.newaxis, :self.num_detections, 2:4]
        # determine norm
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        min_dist_gt = np.min(dist_matrix, axis=1)
        detected = (min_dist_gt <= np.take(
            self.map_intlabel_to_dist_thr, int(self.gt[:, 5][0]))).astype(int)

        # loop over all ground truth objects
        for i in range(self.num_gt):

            # create csv line
            csv_gt_row = [scenario_name, frame_id, i, self.gt[i, 5],
                          self.gt[i, 2], self.gt[i, 3], detected[i], 0,
                          min_dist_gt[i], self.gt[i, 4]]
    else:
        # document gt object as blank object
        for i in range(self.num_gt):
            csv_gt_row = [scenario_name, frame_id, i, self.gt[i, 5],
                          self.gt[i, 2], self.gt[i, 3], 0, 0,
                          1000, self.gt[i, 4]]

    if self.num_gt > 0:
        # write the data row
        self.gt_writer.writerow(csv_gt_row)
        if self.num_detections != 0:
            min_dist_det = np.min(dist_matrix, axis=0)
            label_of_gt = int(self.gt[np.argmin(dist_matrix, axis=0), 5][0])
            matching_gt = (min_dist_det < np.take(
                self.map_intlabel_to_dist_thr, label_of_gt)).astype(int)
            for j in range(self.num_detections):
                # write csv detection [name, frame, detection_num, x, y, bool_matching, distance]
                csv_det_row = [scenario_name, frame_id, j,
                               self.detections[j, 2], self.detections[j, 3], matching_gt[j], min_dist_det[j], self.detections[j, 5], self.detections[j, 4]]
            # write the data row
            self.det_writer.writerow(csv_det_row)

    # rid, aid = ra2idx(self.range_gt, self.az_gt,
    #                       self.range_grid, self.angle_grid)
    # print(
    #     f"Ground truth range: {self.range_gt} m, ground truth azimuth: {self.az_gt}°, which is this cell: {rid, aid}.")

    # print(
    #     f"Ground truth (x, y) cartesian tuple: ({self.x_gt},{self.y_gt}).")


def fuse_standard_and_priori(self, det_standard, num_det_standard):
    deleted_rows = []
    for i in range(self.num_detections):
        # vote not sufficient -> detection cannot be confirmed
        if self.detections[i, 5] <= 2:
            deleted_rows.append(i)
        else:
            # check if detection is already member of 'standard detection'
            for j in range(num_det_standard):
                dist = np.linalg.norm(
                    self.detections[i, 2:4] - det_standard[j, 2:4])
                if dist < 0.5:
                    deleted_rows.append(i)
                    break

    # delete detections without confirmation
    self.detections = np.delete(self.detections, deleted_rows, 0)
    self.num_detections = self.num_detections - \
        len(deleted_rows) + num_det_standard

    # print(
    #     f"num dets: {self.num_detections}, num dets std: {num_det_standard}.")

    # add standard detections
    self.detections = np.concatenate((self.detections, det_standard), axis=0)


def gt_visu_oneframe(self, frame_id: int):
    """Visualize the ground truth objects of one frame in the respective camera frame.
    Work in progress!!

    Args:
        frame_id (int): Frame ID to visualize.
    """
    # read ground truth range/angle from annotations
    a = self.annotations[self.annotations["no"] == frame_id]
    print(f"Length of ground truth annotations in this frame: {len(a)}.")
    # loop over all ground truth objects
    for i in range(len(a)):
        pass
