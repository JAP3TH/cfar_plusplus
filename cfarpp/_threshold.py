from copy import copy
import numpy as np
import ops
from os.path import join
import matplotlib.pyplot as plt


def get_reduction_term(self, age: int):
    """Calculates a reduced threshold (reduced for all cells in red_cells list) for one specific age of the region (priori).

    Args:
        age (int): The age of the spectrum: (-1) for the current spectrum, (-2) for the previous spectrum etc.
    """

    # empty list of reduction functions
    red_fcns = []

    # set threshold initially to default threshold
    self.cfar_th_red = copy(self.cfar_th[:, :, :, age])

    # loop through all detections of the current=last cycle
    for i in range(self.num_detections):
        detection = self.detections[i, :]

        # if this cell is evaluated critically
        if detection[4] > 0.0:
            # each reduction function is a Gaussian - function depends on age and criticality
            new_red_fcn = detection[4] * 4 / 5.0 * self.depth[age + 1] * \
                np.exp(-(self.r_arange - detection[0])**2 / (self.width_r[age + 1]**2) -
                       (self.ang_arange - detection[1])**2 / (self.width_ang**2))
            red_fcns.append(new_red_fcn)

    # determine overall profile of of all dimples and subtract it
    if red_fcns:
        self.cfar_th_red -= np.maximum.reduce(
            np.stack([red_fcns] * 4, axis=-1))


def vis_reduced_threshold_3d(self, age: int, chirp: int = 0):
    """3D visualization of the threshold including areas where the threshold is reduced.
    Plots both the power spectrum (blue) and the threshold (red).

    Args:
        age (int): Age of the previous spectrum to be selected.
        chirp (int, optional): Chirp ID for displayed chirp. Defaults to 0.
    """

    # plot power spectrum and threshold
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf_th = ax.plot_surface(self.r_arange, self.ang_arange, 5.0 * self.cfar_th_red[:, :, age],
                              linewidth=0, antialiased=False)
    surf_spec = ax.plot_surface(self.r_arange, self.ang_arange, self.pow_spec_abs_hist[:, :, chirp, age],
                                linewidth=0, antialiased=False)
    surf_spec.set_facecolor("blue")
    surf_spec.set_alpha(1.0)
    surf_th.set_facecolor("red")
    surf_th.set_alpha(0.4)

    fig.tight_layout(rect=[0, 0, 1.1, 1.1])
    ax.set_xlabel(r"Range Cell $cell_{r}$", fontsize=16)
    ax.set_ylabel(r"Angle Cell $cell_{\varphi}$", fontsize=16)
    ax.set_zlabel(r"Additional Reduction Term $\Delta$", fontsize=16)
    ax.set_zlim(-1, 1)

    # rotate the view of the graph
    ax.view_init(15, 30)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # change z-axis font type
    for t in ax.zaxis.get_major_ticks():
        t.label1.set_fontsize(12)
        t.label2.set_fontsize(12)
    plt.grid()
    plt.savefig(join(self.dir_cfarplusplus, "figures", "surf_1.pdf"))
    plt.show()


def vis_reduced_threshold_2d(self, ang_cell: int, age: int, chirp: int = 0):
    """
    2D visualization of the threshold including areas where the threshold is reduced.
    The angular cell is set to a fixed, static value.
    Plots both the power spectrum (blue) and the threshold (red). Plots the detections and ground truth additionally.

    Args:
        ang_cell (int): ID of the angular cell which is fixed and should be plotted.
        age (int): Age of the previous spectrum to be selected.
        chirp (int, optional): Chirp ID for displayed chirp. Defaults to 0.
    """
    plt.figure(figsize=(10, 8))
    plt.grid()
    # plot the unreduced AND reduced threshold
    plt.plot(self.r_arange[:, 0], 5.0 * self.cfar_th[:, ang_cell, chirp, age], "r",
             linestyle="dotted", linewidth=2, label="Standard CFAR threshold")
    plt.plot(self.r_arange[:, 0], 5.0 * self.cfar_th_red[:, ang_cell, chirp],
             "r", linestyle="solid", linewidth=2, label="Reduced CFAR++ threshold")
    # plot the power spectrum
    plt.plot(
        self.r_arange[:, 0], self.pow_spec_abs_hist[:,
                                                    ang_cell, chirp, age], c="blue",
        linewidth=2, label="Received Signal")

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel(r"Range cell $cell_{r}$", fontsize=24)
    plt.ylabel("Signal Power", fontsize=24)
    dets = np.argwhere(
        5.0 * self.cfar_th_red[:, ang_cell, 0] < self.pow_spec_abs_hist[:,
                                                                        ang_cell, chirp, age])
    gtrange, _ = ops.ra2idx(
        self.gt[:self.num_gt, 0], self.gt[:self.num_gt, 1], self.range_grid, self.angle_grid)
    plt.scatter(dets, np.zeros(len(dets)), s=40, facecolors='none',
                edgecolors='k', label="Detections", linewidths=3)
    plt.scatter(gtrange, np.ones(gtrange.shape) * 0.02, s=40, facecolors='none',
                edgecolors='green', label="Ground Truth", linewidths=3)

    plt.legend(fontsize=20, loc='upper right')
    plt.tight_layout()
    plt.savefig(join(self.dir_cfarplusplus, "figures", "surf_2.pdf"))
    plt.show()
