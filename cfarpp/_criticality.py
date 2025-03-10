import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import ops
from os.path import join

# criticality parameters
width_car = 2.2  # m, width of the vehicle
width_marge = 0.2  # m, safety margin on each side of the vehicle

# time to react and initiate braking action / s
t_react = 0.3
# time to brake the vehicle (4.0 m/s / 8.0 m/s^2) / s
v = 8.0
a = 8.0
t_brake = v / a
t_stop = t_react + t_brake
# time which is required to observe / s
t_observe = 1.0


def get_criticality(points: np.array) -> np.array:
    """Determines 2-fold criticality from a given point: [x, y].

        ttc < t_react: crit_dist = 0 (cannot revert crash anymore nor reduce speed)
        t_react < ttc < t_stop: crit_dist = 0.8 .. 1 (can reduce speed if I react directly)
        t_stop < ttc < t_stop + t_observe: crit_dist = 0 .. 0.8 (some time is left until reaction must start)
        ttc > t_stop + t-observe:  crit_dist = 0 (observation time will start soon)
        Note: we need a time to react. This is the time remaining until a full stop could safe the pedestrian

    Args:
        points (np.array): An array with the size of (N, 2): x, y in meter.

    Returns:
        np.array: An array (N, 1) with the criticality measure values.
    """
    # (1) Distance Criticality
    ttc = points[:, 0] / v

    # basically, the point will not have any distance criticality
    crit_dist = np.zeros(shape=points.shape[0])
    # reaction must happen immediately -> high criticality
    np.putmask(crit_dist, (ttc >= t_react) & (ttc <= t_stop),
               0.2 / (- t_brake)**2 * (ttc - t_stop) ** 2 + 0.8)
    # observation time -> criticality slowly growing
    np.putmask(crit_dist, (ttc > t_stop) & (ttc <= t_stop + t_observe), -
               (0.8 / t_observe) * ttc + (0.8 + 0.8 / t_observe * t_stop))

    # (2) Tube Criticality

    # basically, the point will not have any tube criticality
    crit_tube = np.zeros(shape=points.shape[0])

    # inside drive tube
    np.putmask(crit_tube, np.abs(points[:, 1]) <= width_car, 1.0)
    # inside margin
    np.putmask(crit_tube, (np.abs(points[:, 1]) > width_car) & (np.abs(points[:, 1]) - width_car < width_marge), 2 * ((np.abs(points[:, 1]) -
               width_car) / width_marge)**3 - 3 * ((np.abs(points[:, 1]) - width_car) / width_marge)**2 + 1)

    return crit_dist * crit_tube


def get_crit_map_static(self):
    """Determines the static map for criticality values depending on range & angle measure of the cell.

        As an output, this function modifies the internal variable "self.crit_map_static".
        In a next step, the threshold could be adjusted depending on the "self.crit_map_static" variable.
    """
    crit_map_onelayer = np.zeros((128, 128))
    for i, r in enumerate(self.range_grid):
        for j, ang in enumerate(self.angle_grid):
            x = r * np.cos(ang)
            y = r * np.sin(ang)
            # (1) Distance Criticality
            ttc = x / v
            # reaction must happen immediately -> high criticality
            if ttc >= t_react and ttc <= t_stop:
                crit_dist = 0.2 / (- t_brake)**2 * (ttc - t_stop) ** 2 + 0.8
            # observation time -> criticality slowly growing
            elif ttc > t_stop and ttc <= t_stop + t_observe:
                crit_dist = -(0.8 / t_observe) * ttc + \
                    (0.8 + 0.8 / t_observe * t_stop)
            # too close or too far
            else:
                crit_dist = 0.0

            # (2) Tube Criticality
            # inside drive tube
            if (np.abs(y) <= width_car):
                crit_tube = 1.0
            # inside margin
            elif (np.abs(y) - width_car < width_marge):
                crit_tube = 2 * ((np.abs(y) - width_car) / width_marge)**3 - 3 * (
                    (np.abs(y) - width_car) / width_marge)**2 + 1
            else:
                crit_tube = 0.0

            crit_map_onelayer[i, j] = crit_dist * crit_tube
    # copy this layer for all four chirps to receive a compatible array
    self.crit_map_static = np.repeat(
        crit_map_onelayer[:, :, np.newaxis], 4, axis=2)


def show_crit_map_static(self, save_animation: bool = False):
    """3D visualization of the threshold including areas where the threshold is reduced.
    Plots both the power spectrum (blue) and the threshold (red).
    """
    # plot power spectrum and threshold
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # for VTC Paper only: show only reduction function
    surf_th = ax.plot_surface(self.r_arange, self.ang_arange, self.crit_map_static[:, :, 0],
                              linewidth=0, antialiased=False)
    surf_th.set_facecolor('#0854a5')
    surf_th.set_alpha(0.5)

    fig.tight_layout(rect=[0, 0, 1.1, 1.1])
    ax.set_xlabel(r"Range Cell $cell_{r}$", fontsize=16)
    ax.set_ylabel(r"Angle Cell $cell_{\varphi}$", fontsize=16)
    ax.set_zlabel("Criticality Value", fontsize=16)
    ax.set_zlim(0, 1)

    # rotate the view of the graph
    ax.view_init(15, 30)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # change z-axis font type
    for t in ax.zaxis.get_major_ticks():
        t.label1.set_fontsize(12)
        t.label2.set_fontsize(12)
    plt.grid()
    plt.savefig(join(self.dir_cfarplusplus, "figures", "crit_map_surf.pdf"))
    plt.show()

    def plt3d_update(frame):
        ax.view_init(elev=30, azim=frame)
        print(frame)
        return surf_th

    if save_animation:
        ani = FuncAnimation(fig, plt3d_update,
                            frames=np.arange(0, 360, 1), blit=False)
        ani.save(join(self.dir_cfarplusplus, "figures",
                      "plt3d_rotation.mp4"), writer='ffmpeg', fps=30)

    fig, ax = plt.subplots()
    CS = ax.contour(self.r_arange, self.ang_arange,
                    self.crit_map_static[:, :, 0], 10)
    ax.clabel(CS, fontsize=10)
    ax.set_title('Criticality Metric for Radar Field-of-View', fontsize=16)
    ax.set_xlabel(r"Range Cell $cell_{r}$", fontsize=16)
    ax.set_ylabel(r"Angle Cell $cell_{\varphi}$", fontsize=16)
    plt.grid()
    plt.savefig(join(self.dir_cfarplusplus, "figures", "crit_map_contour.pdf"))
    plt.show()


def get_crit_aware_threshold(self):
    # apply reduction for all four chirp sequences
    self.cfar_th_red[:, :, :4] = self.cfar_th[:, :,
                                              :4, -1] * (1 - 0.2 * self.crit_map_static)


def get_criticality_for_detections(self):
    """Assign a criticality value to each detection.
    """
    # loop over all detections
    # assign/update criticality to the detection
    self.detections[:, 4] = get_criticality(self.detections[:, 2:4])
    self.detections[:, 5] = 0
