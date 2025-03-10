from copy import copy
import _criticality
import cv2
import numpy as np
import ops
from ultralytics import YOLO


def identify_critical_candidates(self):
    """
    Processes the image by applying custom YOLOv10 on it and returns a 
    """

    # Load a pre-trained YOLOv10m model
    if self.model is None:
        self.model = YOLO(self.dir_yolo_model)

    # Perform object detection on an image
    results = self.model(self.file_img, verbose=False)

    # find indices of those boxes which are "pedestrians"
    # np.argwhere(results[0].boxes.cls.cpu() == 0)

    self.num_detections = results[0].boxes.shape[0]
    self.detections = np.zeros((self.num_detections, 6))

    self.cfar_th_red = copy(self.cfar_th[:, :, :, -1])

    red_fcns = []

    # annotated_frame = results[0].plot()
    # cv2.imshow("YOLO11", annotated_frame)
    # cv2.waitKey(0)
    # # # cv2.destroyAllWindows() simply destroys all the windows we created.
    # cv2.destroyAllWindows()

    classes = results[0].boxes.cls.cpu()

    for i in range(self.num_detections):
        # adapt only for persons as they are relevant for criticality
        if classes[i] == 0:
            [u_im, v_im, w, h] = results[0].boxes[i].xywh.cpu()[0]

            # point to ground
            v_im += h / 2

            # reverse projection (d not included)
            (p_x_world, p_y_world) = yolo_img_to_world(self, u_im, v_im)
            # conversion into polar coordinates
            r, ang = ops.cart2pol_ramap(p_x_world, -p_y_world)
            ang -= np.pi / 2

            # fill detection candidates
            self.detections[i, 2] = p_x_world
            self.detections[i, 3] = p_y_world
            r_idx, ang_idx = ops.ra2idx(
                r, ang, self.range_grid, self.angle_grid)

            # determine criticality
            self.detections[i, 4] = _criticality.get_criticality(
                np.atleast_2d(self.detections[i, 2:4]))

            # if this cell is evaluated critically
            if self.detections[i, 4] > 0.0:
                # each reduction function is a Gaussian - function depends on age and criticality
                new_red_fcn = self.detections[i, 4] * 0.8 / 5.0 * \
                    np.exp(-(self.r_arange - r_idx)**2 / (7**2) -
                           (self.ang_arange - ang_idx)**2 / (7**2))
                # np.exp(-(self.r_arange - r_id)**2 / (self.width_r[age + 1]**2) -
                #        (self.ang_arange - ang_id)**2 / (self.width_ang**2))
                red_fcns.append(new_red_fcn)
                # print(
                #     f"New reduction function has max. value: {np.max(new_red_fcn)}")

    # determine overall profile of of all dimples and subtract it
    if red_fcns:
        self.cfar_th_red -= np.maximum.reduce(
            np.stack([red_fcns] * 4, axis=-1))

        # print(f"The point has coordinates {p_x_world}, {p_y_world}.")
        # print(f"The ground truth point is {self.gt[:self.num_gt, :2]}.")


def yolo_img_to_world(self, u_im: int, v_im: int) -> tuple[float, float]:
    # reverse projection (d not included)
    p_x_world = self.p_z * self.K[1, 1] / (self.K[1, 2] - v_im)
    p_y_world = (self.K[0, 2] - u_im) * p_x_world / self.K[0, 0]
    return (p_x_world.item(), p_y_world.item())
