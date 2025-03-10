import colorsys
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pycocotools.mask as cocomask

'''
This library is taken almost unmodified from the CRUW devkit.
Thanks to Yizhou Wang for providing the repository.
Link: https://github.com/yizhou-wang/cruw-devkit
'''


def draw_dets(ax, img, bboxes, colors, texts=None, masks=None):
    """
    Draw bounding boxes on image.
    :param ax: plt ax
    :param img: image
    :param bboxes: xywh n_bbox x 4
    :param colors: n_bbox colors
    :param texts: n_bbox text strings
    :param masks: n_bbox rle masks
    :return:
    """
    n_bbox = len(bboxes)
    for bbox_id in range(n_bbox):
        bbox = bboxes[bbox_id]
        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                               edgecolor=colors[bbox_id], facecolor='none'))
        if texts is not None:
            ax.text(bbox[0], bbox[1], texts[bbox_id], color=colors[bbox_id])
        if masks is not None:
            binary_mask = cocomask.decode(masks[bbox_id])
            apply_mask(img, binary_mask, colors[bbox_id])
    ax.imshow(img)


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.25):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def magnitude(chirp, radar_data_type):
    """
    Calculate magnitude of a chirp
    :param chirp: radar data of one chirp (w x h x 2) or (2 x w x h)
    :param radar_data_type: current available types include 'RI', 'RISEP', 'AP', 'APSEP'
    :return: magnitude map for the input chirp (w x h)
    """
    c0, c1, c2 = chirp.shape
    if radar_data_type == 'RI' or radar_data_type == 'RISEP':
        if c0 == 2:
            chirp_abs = np.sqrt(chirp[0, :, :] ** 2 + chirp[1, :, :] ** 2)
        elif c2 == 2:
            chirp_abs = np.sqrt(chirp[:, :, 0] ** 2 + chirp[:, :, 1] ** 2)
        else:
            raise ValueError
    elif radar_data_type == 'AP' or radar_data_type == 'APSEP':
        if c0 == 2:
            chirp_abs = chirp[0, :, :]
        elif c2 == 2:
            chirp_abs = chirp[:, :, 0]
        else:
            raise ValueError
    else:
        raise ValueError
    return chirp_abs


def draw_centers(ax, chirp, dts, colors, rgrid, anggrid, texts=None, chirp_type='RISEP', normalized=False):
    """
    Draw object centers on RF image.
    :param ax: plt ax
    :param chirp: radar chirp data
    :param dts: [n_dts x 2] object detections
    :param colors: [n_dts]
    :param texts: [n_dts] text to show beside the centers
    :param chirp_type: radar chirp type
    :param normalized: is radar data normalized or not
    :return:
    """
    chirp_abs = magnitude(chirp, chirp_type)
    if normalized:
        ax.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
        # ax.set_xticks(anggrid)
        # ax.set_yticks(rgrid)
    else:
        ax_img = ax.imshow(chirp_abs, origin='lower')
        ticks = np.linspace(0, 127, 8, dtype=np.int32)
        xticklabels = [np.round(np.rad2deg(anggrid[tick]), 1)
                       for tick in ticks]
        yticklabels = [np.round(rgrid[tick], 2) for tick in ticks]
        ax.set_xticks(ticks, xticklabels)
        ax.set_yticks(ticks, yticklabels)
        ax.set_xlabel("Azimuth Angle / °")
        ax.set_ylabel("Range / m")
        plt.colorbar(ax_img, ax=ax)
    n_dts = len(dts)
    for dt_id in range(n_dts):
        color = np.array(colors[dt_id])
        color = color.reshape((1, -1))
        ax.scatter(dts[dt_id][1], dts[dt_id][0],
                   s=100, c=color, edgecolors='white')
        if texts is not None:
            ax.text(dts[dt_id][1] + 2, dts[dt_id][0] + 2, '%s' %
                    texts[dt_id], c='white')


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors_rgb(n_colors):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 0.7
    hsv = [(i / n_colors, 1, brightness) for i in range(n_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [list(color) for color in colors]
    return colors
