import ops
import vis

# import internal functions
import _cfar
import _criticality
import _evaluation
import _threshold
import _yolotools

# standard libraries
from copy import copy
import csv
import cv2
from datetime import datetime
from matplotlib import gridspec, use
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join, isfile
from pathlib import Path
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import yaml

# timing
from time import time


"""
    This is the script which defines the main class of a ROD_2021 object.
    To observe a specific frame:
    (1) Load a scenario with self.scenario_loader()

    Author: Tim Bruehl
    Date: 18/11/2024

    """

use('tkagg')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
})


class ROD_2021:

    def __init__(self, av_range: int, av_angle: int,
                 guard_range: int, guard_angle: int,
                 obs_history: int = 5, votings: int = 3,
                 dimple_depth: float = 0.4) -> None:
        """Initializes the boundary look up tabels to accelerate the CFAR calculation.

        Args:
            av_range (int): Number of averaging range cells around the CuT.
            av_angle (int): Number of averaging angle cells around the CuT.
            guard_range (int): Number of guard range cells around the CuT.
            guard_angle (int): Number of guard angle cells around the CuT.
            obs_history (int, optional): The number of frames which are considered for the observation history. Defaults to 5.
            votings (int, optional): The number of required votings from previous spectra to confirm a detection. Defaults to 3.
            dimple_depth (float, optional): The maximum depth of the dimple (as an absolute signal value). Defaults to 0.4.
        """
        # root directory of cfarplusplus
        self.dir_cfarplusplus = Path(__file__).resolve().parents[1]

        self.observ_history = obs_history
        self.no_votings = votings

        # read parameters from YAML file
        with open(join(self.dir_cfarplusplus, "cfg", "config.yaml"), encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

            # data set directory
            self.dir_rod = cfg['rod_dataset_directory']
            self.dir_export = cfg['export_directory']
            self.dir_yolo_model = cfg['yolo_model_directory']

            # frequency / 1/s
            self.fps = cfg['camera_frames_per_second']

            # dimension of dimples
            vru_speed = cfg['vru_velocity_meterspersecond']
            self.width_ang = cfg['dimple_angular_cell_width']
            self.depth = np.linspace(
                dimple_depth, 0,
                num=self.observ_history, endpoint=False
            )[::-1]
            # mapping from integer label to distance threshold accepting a detection belonging to a gt object
            # pedestrian, cyclist, car
            self.map_intlabel_to_dist_thr = np.array(
                cfg["detection_to_groundtruth_acceptance_distance"])

        # image directory
        self.dir_image = join(self.dir_rod, "TRAIN_CAM_0", "TRAIN_CAM_0")
        # annotation directory
        self.dir_annotations = join(
            self.dir_rod, "TRAIN_RAD_H_ANNO", "TRAIN_RAD_H_ANNO")
        self.file_calib = join(self.dir_rod, "CAM_CALIB",
                               "calib", "2019_05_09", "cam_0.yaml")

        # appr. 0.2m approximate length of range bin
        # 5 cycles observation = 167 ms, pedestrian can travel 0.66m = 4 range bins in this time
        max_traveled_bins_r = np.ceil(
            self.observ_history / self.fps * vru_speed * 128 / 25 * 2)
        # the older the data, the lower the index
        self.width_r = np.linspace(
            start=max_traveled_bins_r, stop=1, num=self.observ_history)

        print(f"Range width: {self.width_r}, depth: {self.depth}.")

        self.r_arange, self.ang_arange = np.mgrid[0:128:1,
                                                  0:128:1]

        self.label_to_num = {"pedestrian": 0,
                             "cyclist": 1,
                             "car": 2}

        # open calib YAML
        with open(self.file_calib, 'r', encoding='utf-8') as yamlc:
            yamlloaded = yaml.safe_load(yamlc)
            #     [fx  0 cx]
            # K = [ 0 fy cy]
            #     [ 0  0  1]
            self.K = np.reshape(yamlloaded['camera_matrix']['data'], (3, 3))
            self.d = np.reshape(
                yamlloaded['distortion_coefficients']['data'], (5, 1))
            # rectification might not be required (unity matrix)
            self.R = np.reshape(
                yamlloaded['rectification_matrix']['data'], (3, 3))
            #     [fx'  0  cx' Tx]
            # P = [ 0  fy' cy' Ty]
            #     [ 0   0   1   0])
            self.P = np.reshape(
                yamlloaded['projection_matrix']['data'], (3, 4))

        # power spectrum of four chirps
        # range_cell x angle_cell x chirp_seq x (real/imaginary)
        self.pow_spec = np.zeros((128, 128, 4, 2))
        # absolute power spectrum with history
        self.pow_spec_abs_hist = np.zeros(
            (128, 128, 4, self.observ_history + 1))
        # cfar threshold per chirp for all cells (store last thresholds rolling)
        # range_cell x angle_cell x chirp_seq x age(t-observ_history, ..., t)
        self.cfar_th = np.zeros((128, 128, 4, self.observ_history + 1))

        # static 2D map for criticality(r, ang)
        self.crit_map_static = np.zeros((128, 128, 4))

        # current cfar threshold incl. region reduction
        # range_cell x angle_cell x chirp_seq
        self.cfar_th_red = np.zeros((128, 128, 4))

        # centroids of the CFAR blobs = "point cloud"
        self.blob_centroids = np.zeros((1, 2))

        # generate range and angle grids
        self.range_grid = ops.confmap2ra(name='range')
        self.angle_grid = ops.confmap2ra(name='angle')

        # detection storage for up to 1000 detections
        # [r, angle, x, y, criticality, vote]
        self.detections = np.zeros((1000, 6))
        self.num_detections = 0

        # ground truth storage for up to 100 ground truths
        # [r, angle, x, y, criticality, class]
        self.gt = np.zeros((100, 6))
        self.num_gt = 0

        # determine time stamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # prepare CFAR kernel bounds
        self.r_l_out = np.zeros((128), dtype=np.int16)
        self.r_r_out = np.zeros((128), dtype=np.int16)
        self.r_l_in = np.zeros((128), dtype=np.int16)
        self.r_r_in = np.zeros((128), dtype=np.int16)
        self.a_l_out = np.zeros((128), dtype=np.int16)
        self.a_r_out = np.zeros((128), dtype=np.int16)
        self.a_l_in = np.zeros((128), dtype=np.int16)
        self.a_r_in = np.zeros((128), dtype=np.int16)
        self.kernel_size = np.zeros((128, 128))

        # CA CFAR parameters
        # size of the frame around the cell-under-test
        # av_range = 10, av_angle = 10, guard_range = 4, guard_angle = 4
        # number of guard cells on each side neighboring the cell-under-test
        for i in range(128):
            self.r_l_out[i] = max(0, i - av_range)
            self.r_r_out[i] = min(127, i + av_range)
            self.r_l_in[i] = max(0, i - guard_range)
            self.r_r_in[i] = min(127, i + guard_range)
            self.a_l_out[i] = max(0, i - av_angle)
            self.a_r_out[i] = min(127, i + av_angle)
            self.a_l_in[i] = max(0, i - guard_angle)
            self.a_r_in[i] = min(127, i + guard_angle)
        for i in range(128):
            for j in range(128):
                # kernel size of averaging kernel
                n_out = (self.r_r_out[i] - self.r_l_out[i]) * \
                    (self.a_r_out[j] - self.a_l_out[j])
                # kernel size of guard cells
                n_in = (self.r_r_in[i] - self.r_l_in[i]) * \
                    (self.a_r_in[j] - self.a_l_in[j])
                # total CFAR kernel size
                self.kernel_size[i, j] = n_out - n_in

        # open csv files and write head line
        fn_gt = f"{ts}_avr{av_range}_ava{av_angle}_gr{guard_range}_ga{guard_angle}_gt.csv"
        self.f_gt = open(join(self.dir_export, fn_gt),
                         mode='a', newline='', encoding="utf-8")
        self.gt_writer = csv.writer(self.f_gt)
        self.gt_writer.writerow(
            ["file", "frame", "gt_id", "label_num", "x_gt", "y_gt",
             "detected", "cfar_cells", "det_disterror", "criticality"])

        fn_det = f"{ts}_avr{av_range}_ava{av_angle}_gr{guard_range}_ga{guard_angle}_detections.csv"
        self.f_det = open(join(self.dir_export, fn_det),
                          mode='a', newline='', encoding="utf-8")
        self.det_writer = csv.writer(self.f_det)
        self.det_writer.writerow(["file", "frame", "detection_id", "x_det",
                                  "y_det", "matching_gt", "gt_disterror", "votes", "criticality"])

        # default define of some variables
        self.dir_img = ""
        self.dir_radar = ""
        self.file_img = ""
        self.file_radar_anno = ""
        self.frame_len = 0
        self.annotations = None

        # yolo model
        self.model = None

        # range gt and azimuth ground truth bin
        self.range_gt = 0
        self.az_gt = 0

        # camera height (world coordinate system)
        self.p_z = -0.85

    def scenario_loader(self, scenario_name: str) -> None:
        """Loads one scenario of the data set.
        """
        # count number of frames
        self.dir_img = join(self.dir_image,
                            scenario_name, "IMAGES_0")
        # determine frame length
        self.frame_len = len([name for name in listdir(
            self.dir_img) if isfile(join(self.dir_img, name))])
        print(
            f"Total frame length of scenario {scenario_name} is: {self.frame_len}.")
        # build file directory paths
        self.dir_radar = join(self.dir_rod, "TRAIN_RAD_H", "TRAIN_RAD_H",
                              scenario_name, "RADAR_RA_H")
        self.file_radar_anno = join(
            self.dir_annotations, scenario_name + ".txt")

        # open radar annotations
        self.annotations = pd.read_csv(
            self.file_radar_anno, sep=" ", header=None)
        self.annotations.columns = ["no", "range", "azimuth", "label"]

    def find_max_cell(self, chirp_id: int, power_spec: np.array):
        """Prints maximum cell number

        Args:
            chirp_id (int): Chirp ID, must be in range [0, 3]
            power_spec (np.array): An absolute power spectrum with dimensions [128, 128, 4].
        """

        cell_max_idx = np.argmax(power_spec[:, :, chirp_id])
        cell_max = np.unravel_index(cell_max_idx, (128, 128))
        print(
            f"The max cell of the chirp {chirp_id} is {cell_max}.")

        # cell to physical
        rng_max, azm_max = ops.idx2ra(
            cell_max_idx[0], cell_max_idx[1], self.range_grid, self.angle_grid)
        print(
            f"This cell is at {rng_max} m with an angle of {np.rad2deg(azm_max)}°.")

    def load_one_frame(self, frame_id):
        # open one radar package = 4 chirps, determine absolute power value
        # self.pow_spec.shape = (128, 128, 4, 2) range_cell, angle_cell, chirp_id, re/im
        self.pow_spec[:, :, 0, :] = np.load(
            join(self.dir_radar, f'{frame_id:06d}_0000.npy'))
        self.pow_spec[:, :, 1, :] = np.load(
            join(self.dir_radar, f'{frame_id:06d}_0064.npy'))
        self.pow_spec[:, :, 2, :] = np.load(
            join(self.dir_radar, f'{frame_id:06d}_0128.npy'))
        self.pow_spec[:, :, 3, :] = np.load(
            join(self.dir_radar, f'{frame_id:06d}_0192.npy'))

        # open image of the frame
        self.file_img = join(self.dir_img, f"{frame_id:010d}.jpg")

        # self.show_dataset_rod2021(frame_id)

        # kick oldest element on the left hand side
        self.pow_spec_abs_hist = np.roll(self.pow_spec_abs_hist, -1, axis=3)

        # determine new absolute power values and fill them on the right hand side
        # self.pow_spec_abs_hist.shape = (128, 128, 4, observ_hist) range_cell, angle_cell, chirp_id, observation_history
        for i in range(4):
            self.pow_spec_abs_hist[:, :, i, -1] = vis.magnitude(
                self.pow_spec[:, :, i, :], 'RISEP')

        # read ground truth range/angle from annotations
        a = self.annotations[self.annotations["no"] == frame_id]
        self.num_gt = len(a)

        self.gt[:self.num_gt, 0] = a["range"]
        self.gt[:self.num_gt, 1] = a["azimuth"]
        self.gt[:self.num_gt, 2] = a["range"] * np.cos(-a["azimuth"])
        self.gt[:self.num_gt, 3] = a["range"] * np.sin(-a["azimuth"])
        self.gt[:, 4] = _criticality.get_criticality(
            self.gt[:, 2:4])
        self.gt[:self.num_gt, 5] = a["label"].map(self.label_to_num).to_numpy()

    def loop_frames(self, scenario_name) -> None:
        """
        Loops over all frames of the currently selected scene without additional treatment.
        """
        # loop with progress bar feature
        for frame_id in tqdm(range(self.frame_len)):
            t1 = time()
            self.load_one_frame(frame_id)
            t2 = time()
            # only the last (most right) column is used
            self.cfar_th[:, :, :, -1] = _cfar.compute_ca_cfar(self)
            t3 = time()
            _cfar.cfar_peak_detection(self, age=-1, reduced=False)
            # store detections (r_cell, ang_cell, x, y, _, _)
            _cfar.fill_detections_from_blobs(self)
            # adds criticality to each of "self.detections"
            _criticality.get_criticality_for_detections(self)
            _evaluation.gt_eval(self, scenario_name, frame_id)
            t4 = time()
            # print(
            #     f"Time for loading frame: {t2-t1}, ca_cfar: {t3-t2}, peak + eval: {t4-t3}.")

    def loop_frames_treat_current_step(self, scenario_name) -> None:
        """
        Loops over all frames of the currently selected scene with additional treatment just for this current step.
        """
        # determine offset caused by varying criticality per region
        _criticality.get_crit_map_static(self)

        # loop with progress bar feature
        for frame_id in tqdm(range(self.frame_len)):
            self.load_one_frame(frame_id)
            # compute regular threshold
            # only the last (most right) column is used
            self.cfar_th[:, :, :, -1] = _cfar.compute_ca_cfar(self)
            # compute reduced threshold which is lowered for critical regions JUST IN THIS FRAME
            _criticality.get_crit_aware_threshold(self)

            # apply peak detection with the reduced threshold
            _cfar.cfar_peak_detection(self, age=-1, reduced=True)
            # store detections (r_cell, ang_cell, x, y, _, _)
            _cfar.fill_detections_from_blobs(self)
            _criticality.get_criticality_for_detections(self)
            _evaluation.gt_eval(self, scenario_name, frame_id)

    def loop_frames_priori(self, scenario_name) -> None:
        """
        Loops over all frames of the currently selected scene and applies priori processing.
        1. Load current frame
        2. Compute CA-CFAR -> self.cfar_th = 2-D threshold for cfar
        3. Perform peak detections with high factor -> sure detections
        4. Perform peak detections with low factor -> detection candidates
        5. Identify criticality per detection candidate -> self.detection[N, [r, ang, x, y, criticality, vote]]
        6. For h in history_frames:
                get reduced threshold with Gaussian dimples
                compute new detections in h with reduced threshold
                associate each det_h with det from self.detections and vote detections
        7. Fuse sure detections with confirmed and critical detection candidates.
        """
        # loop with progress bar feature
        for frame_id in tqdm(range(self.frame_len)):
            # for frame_id in tqdm(range(210, 215)):
            # load frame
            self.load_one_frame(frame_id)
            # determine detections by standard CA-CFAR
            # kicks oldest threshold on the left hand side
            self.cfar_th = np.roll(self.cfar_th, -1, axis=3)
            # fill newest threshold on the right hand side
            self.cfar_th[:, :, :, -1] = _cfar.compute_ca_cfar(self)
            _cfar.cfar_peak_detection(
                self, age=-1, thresh_factor=5.0, reduced=False)
            # store detections (r, ang, x, y) in "self.detections"
            _cfar.fill_detections_from_blobs(self)
            # save 'standard' detections
            det_standard = copy(self.detections)
            num_det_standard = copy(self.num_detections)

            # identify weaker 'candidates'
            _cfar.cfar_peak_detection(
                self, age=-1, thresh_factor=4.0, reduced=False)
            # store detections (r, ang, x, y) in "self.detections"
            _cfar.fill_detections_from_blobs(self)

            # adds criticality to each of "self.detections"
            _criticality.get_criticality_for_detections(self)

            # revisit prior spectra
            for age in range(min(self.observ_history, frame_id)):
                # most right threshold is T_t, second right one is T_(t-1)
                age = -age - 2
                # determine adapted threshold function based on the critical zones of "cur_detections"
                _threshold.get_reduction_term(self, age)
                # peak detection applying reduced threshold
                _cfar.cfar_peak_detection(self, age=age, reduced=True)
                # associate investigated "cur_detections" with "self.blob_centroids" --> vote
                _cfar.associate_detections(self)
                # _, aid = ops.ra2idx(
                #     self.gt[0, 0], self.gt[0, 1], self.range_grid, self.angle_grid)
                # _threshold.vis_reduced_threshold_3d(self, age)
                # _threshold.vis_reduced_threshold_2d(self, aid, age)

            # add 'standard' and 'weak' candidates
            _evaluation.fuse_standard_and_priori(
                self, det_standard, num_det_standard)

            # evaluate detections
            _evaluation.gt_eval(self, scenario_name, frame_id)

    def loop_frames_yolo(self, scenario_name) -> None:
        """
        Loops over all frames of the currently selected scene and applies priori processing.
        """
        # loop with progress bar feature
        for frame_id in tqdm(range(self.frame_len)):
            # load the frame
            self.load_one_frame(frame_id)
            # compute regular threshold
            # only the last (most right) column is used
            self.cfar_th[:, :, :, -1] = _cfar.compute_ca_cfar(self)
            # identify critical zones with the CFAR threshold
            # input: image, output: reduced threshold self.cfar_th_red
            _yolotools.identify_critical_candidates(self)
            # apply peak detection with the reduced threshold
            _cfar.cfar_peak_detection(self, age=-1, reduced=True)
            # store detections (r_cell, ang_cell, x, y, _, _)
            _cfar.fill_detections_from_blobs(self)
            _criticality.get_criticality_for_detections(self)
            _evaluation.gt_eval(self, scenario_name, frame_id)

    def point_projection(self, pt_r: float, pt_angle: float):
        """Projects ground truth points into the camera image
        """
        p_x = pt_r * np.cos(pt_angle)
        p_y = pt_r * np.sin(pt_angle)
        p_z = 0.0

        # camera cosy
        roll = 0.0  # (cam z) in deg
        yaw = 0.0  # (cam y) in deg
        nick = 4.0  # (cam x) in deg

        euler_angles = [roll, yaw, nick]

        rot_vec = R.from_euler('ZYX', euler_angles, degrees=True).as_rotvec()
        rot_matrix = R.from_euler(
            'ZYX', euler_angles, degrees=True).as_matrix()
        trans_vec = np.array([[0.0, -1.65, 0.0]])
        # point to camera coordinate frame
        point3d = np.array(
            [-p_y, p_z, -p_x], dtype=np.float32).reshape(3, 1)

        point2d, _ = cv2.projectPoints(point3d,
                                       rot_vec,
                                       trans_vec,
                                       self.K,
                                       self.d)

        Rt = np.hstack((rot_matrix, trans_vec.T))
        uvw = self.K @ Rt @ np.array([[-p_y, p_z, -p_x, 1]]).T

        im = cv2.imread(self.file_img)

        print("Projected point")
        print(point2d[0][0][0], point2d[0][0][1])

        cv2.circle(im, (int(point2d[0][0][0]), int(
            point2d[0][0][1])), 10, (255, 0, 0), -1)

        cv2.imshow('Augmented Image', im)

        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        cv2.waitKey(0)

        # cv2.destroyAllWindows() simply destroys all the windows we created.
        cv2.destroyAllWindows()

    def show_dataset_rod2021(self, frame_id):
        """Applys a reduced method of the CRUW devkit to show camera image and RF image.
        """
        img = plt.imread(self.file_img)
        chirp_id = 0
        chirp = self.pow_spec[:, :, chirp_id, :]
        with open(self.file_radar_anno, 'r') as f:
            lines = f.readlines()
        center_ids = []
        categories = []
        for line in lines:
            fid, rng, azm, class_name = line.rstrip().split()
            fid = int(fid)
            if fid == frame_id:
                rng = float(rng)
                azm = float(azm)
                rid, aid = ops.ra2idx(
                    rng, azm, self.range_grid, self.angle_grid)
                center_ids.append([rid, aid])
                categories.append(class_name)
        center_ids = np.array(center_ids)
        n_obj = len(categories)

        fig = plt.figure()
        fig.set_size_inches(16, 5)
        gs = gridspec.GridSpec(1, 2)

        ax1 = plt.subplot(gs[0])
        ax1.axis('off')
        vis.draw_dets(ax1, img, [], [])
        ax1.set_title('RGB Image')

        ax2 = plt.subplot(gs[1])
        colors = vis.generate_colors_rgb(n_obj)
        vis.draw_centers(ax2, chirp, center_ids, colors,
                         self.range_grid, self.angle_grid, texts=categories)
        ax2.set_title(f'RF Image (BEV) of chirp {chirp_id}')

        fig.subplots_adjust(hspace=0, wspace=0)

        plt.show()

    def polar_to_cartesian(self):

        # range_cell=x, angle_cell=y
        # Generate a polar coordinate image
        # Define grid in polar coordinates
        R, Theta = np.meshgrid(self.range_grid, self.angle_grid)

        # Convert polar coordinates to Cartesian coordinates
        X = R * np.cos(-Theta)
        Y = R * np.sin(-Theta)

        # Flatten the arrays for interpolation
        points = np.column_stack((X.ravel(), Y.ravel()))
        values = self.pow_spec_abs_hist[:, :, 0, -1].ravel(order='F')

        # Create a Cartesian grid
        x_cart = np.linspace(0, 30, 500)
        y_cart = np.linspace(-30, 30, 500)
        X_cart, Y_cart = np.meshgrid(x_cart, y_cart)

        # Interpolate the polar data onto the Cartesian grid
        cartesian_data = griddata(
            points, values, (X_cart, Y_cart), method='linear')

        # Plot the Cartesian image
        plt.figure()
        plt.grid()
        plt.imshow(cartesian_data, extent=(0, 30, -30, 30),
                   origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.title('Radar RF Image in Cartesian Coordinates')
        plt.xlabel('x / m', fontsize=20)
        plt.ylabel('y / m', fontsize=20)
        plt.show()
