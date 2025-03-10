import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.ndimage import label

import ops

from time import time


def compute_ca_cfar(self, age: int = -1):
    """Function maps class function with class variables on numba function.

    Args:
        age (int, optional): The age of the power spectrum, which should be used. Defaults to -1.

    Returns:
        _type_: Function call of numba optimized method.
    """
    return _ca_cfar_both(self.r_l_out, self.r_r_out,
                         self.a_l_out, self.a_r_out,
                         self.r_l_in, self.r_r_in,
                         self.a_l_in, self.a_r_in,
                         self.kernel_size,
                         self.pow_spec_abs_hist[:, :, :, age])


@staticmethod
@njit
def _ca_cfar_both(r_l_out: np.array, r_r_out: np.array, a_l_out: np.array, a_r_out: np.array,
                  r_l_in: np.array, r_r_in: np.array, a_l_in: np.array, a_r_in: np.array,
                  kernel_size: np.array, pow_spec_abs: np.array) -> np.array:
    """Determines a 2D CFAR threshold function for the complete spectrum.

    Args:
        r_l_out (np.array): (128,) Outter range border on the left hand side of the array.
        r_r_out (np.array): (128,) Outter range border on the right hand side of the array.
        a_l_out (np.array): (128,) Outter angle border on the left hand side of the array.
        a_r_out (np.array): (128,) Outter angle border on the right hand side of the array.
        r_l_in (np.array):  (128,) Inner range border on the left hand side of the array.
        r_r_in (np.array):  (128,) Inner range border on the right hand side of the array.
        a_l_in (np.array):  (128,) Inner angle border on the left hand side of the array.
        a_r_in (np.array):  (128,) Inner angle border on the right hand side of the array.
        kernel_size (np.array): (128, 128) Size of each cells kernel (number of averaged cells)
        pow_spec_abs (np.array): (128, 128, 4) The absolute power spectrum.

    Returns:
        np.array: (128, 128) The 2D threshold function.
    """

    # cfar threshold per chirp for all cells
    cfar_th = np.zeros((128, 128, 4))

    # loop over all chirps
    for chirp_id in range(4):
        # loop over all range cells
        for rcell in range(128):
            # loop over all angle cells
            for acell in range(128):
                # cut mask from current power spectrum
                out_mask = pow_spec_abs[r_l_out[rcell]:r_r_out[rcell],
                                        a_l_out[acell]:a_r_out[acell],
                                        chirp_id]
                in_mask = pow_spec_abs[r_l_in[rcell]:r_r_in[rcell],
                                       a_l_in[acell]:a_r_in[acell],
                                       chirp_id]

                # averaging to determine the threshold
                cfar_th[rcell, acell, chirp_id] = (
                    np.sum(out_mask) - np.sum(in_mask)) / kernel_size[rcell, acell]

    return cfar_th


def cfar_peak_detection(self, age: int, thresh_factor: float = 5.0, reduced: bool = False):
    """Apply peak detection for peaks in self.pow_spec_abs_hist above self.cfar_th, respecting the correct "age".

    Args:
        age (int): Frame age. If current frame is 100 and peak detection shall be applied on frame 98, the frame age is -2.
        thresh_factor (float, optional): Factor for the threshold to exceed the cell average. Defaults to 5.0.
        reduced (bool, optional): Selection if reduced threshold or standard threshold shall be applied. Defaults to False.
    """

    # reduced threshold mode --> map correct threshold and power spectrum according to age
    if reduced:
        thresh = self.cfar_th_red * thresh_factor
    # non-reduced --> take calculated threshold and power spectrum of current frame
    else:
        thresh = self.cfar_th[:, :, :, age] * thresh_factor

    pow_spec = self.pow_spec_abs_hist[:, :, :, age]

    # print(
    #     f"Threshold cell: {thresh[50, 50, :]}, power spectrum: {pow_spec[50, 50, :]}.")

    # detection (vote factor per cell = 2 chirps)
    # positives.shape = (n_pos, 3) range cell, angle_cell, chirp_id
    positives = np.argwhere((pow_spec - thresh) >= 0)

    # find unique values of the cell tuples [rcell, acell]
    cellcombi_candidates, cnt = np.unique(
        positives[:, :2], axis=0, return_counts=True)

    # a mask to detect (first, set everything to zero)
    self.det_bin_mask = np.zeros((128, 128))
    self.det_bin_mask[cellcombi_candidates[:, 0],
                      cellcombi_candidates[:, 1]] = 1

    # Connected Component Labeling (CCL) reduces the CFAR positive areas to points
    labeled, num_blobs = label(self.det_bin_mask, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    # array with centroids of all blobs
    self.blob_centroids = np.zeros((num_blobs, 2))
    # loop over all blobs
    for label_id in range(1, num_blobs + 1):
        # all points which belong to this blob
        coordinates = np.argwhere(labeled == label_id)

        # METHOD 1: spatial center of all detections
        # print(f"Max, method 1: {coordinates.mean(axis=0)}")
        # self.blob_centroids[label_id - 1, :] = coordinates.mean(axis=0)

        # METHOD 2: find the cell which had the maximum power (mean of all 4 chirps)
        # mean power of each cell candidate over all four chirps
        coord_mean_power = pow_spec[coordinates[:, 0],
                                    coordinates[:, 1], :].mean(axis=1)
        # cell index with the maximum power
        coord_max_power = np.argmax(coord_mean_power)
        # coordinate
        # print(f"Max, method 2: {coordinates[coord_max_power]}")
        self.blob_centroids[label_id - 1, :] = coordinates[coord_max_power]


def fill_detections_from_blobs(self):
    """
    Take the current "self.blob_centroids" and writes range, angle, x, y data inside.
    """
    self.num_detections = self.blob_centroids.shape[0]
    self.detections = np.zeros((self.num_detections, 6))
    self.detections[:, :2] = self.blob_centroids
    r_det, ang_det = ops.idx2ra(self.detections[:, 0].astype(int),
                                self.detections[:, 1].astype(int),
                                self.range_grid,
                                self.angle_grid)
    det_y, det_x = ops.pol2cart_ramap(r_det, ang_det)
    self.detections[:, 2] = det_x
    self.detections[:, 3] = -det_y


def associate_detections(self):
    """Associates the detections of the current measurement (self.detections) to the ones of a previous measurement ("self.blob_centroid").
    """
    # loop over all priori point clouds, not the current one
    for i in range(self.num_detections):

        # get number of detections in previous point cloud (P_t-x)
        num_priori_dets = self.blob_centroids.shape[0]

        # if detection is critical
        if self.detections[i, 4] > 0.0:
            det_x = self.detections[i, 2]
            det_y = self.detections[i, 3]

            # loop over all priori detections
            for j in range(num_priori_dets):
                priori_det = self.blob_centroids[j, :]
                # determine cartesian coordinates of current cell
                r_det, ang_det = ops.idx2ra(int(priori_det[0]),
                                            int(priori_det[1]),
                                            self.range_grid,
                                            self.angle_grid)
                priori_det_y, priori_det_x = ops.pol2cart_ramap(
                    r_det, -ang_det)
                # determine distance between candidate and current cell
                dist = np.linalg.norm(
                    [priori_det_x - det_x, priori_det_y - det_y])
                # associate if distance undercuts threshold
                if dist < 0.5:
                    # increment vote and break for-loop
                    self.detections[i, 5] += 1
                    break
