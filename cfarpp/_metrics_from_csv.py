import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt
import pandas as pd


use('tkagg')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
})


def get_detection_metrics(dir_gt: str, dir_det: str, label_filter: list):
    """Determines the following metrics from a CSV log:
    Recall, precision, number of false positives, number of false negatives.   

    Args:
        dir_gt (str): Directory where Ground Truth CSV log is stored.
        dir_det (str): Directory where Detection CSV log is stored.
        label_filter (list): List of strings which labels shall be included in the metrics. 0=person, 1=bicyclist, 2=car

    Returns:
        _type_: _description_
    """

    ground_truth = pd.read_csv(dir_gt)
    detections = pd.read_csv(dir_det)

    # filter ground truth and detections
    ground_truth = ground_truth[ground_truth.label_num.isin(label_filter)]

    # count false positives and false negatives
    no_matching = detections.loc[:, "matching_gt"].sum()
    no_detected = ground_truth.loc[:, "detected"].sum()
    recall = no_detected / len(ground_truth)
    prec = no_matching / len(detections)
    f1score = 2 * prec * recall / (prec + recall)
    fp = len(detections) - no_matching
    fn = len(ground_truth) - no_detected

    print(
        f"Total ground truth objects: {len(ground_truth)}, total detections: {len(detections)}.")
    print(f"Number of False Positives: {fp}, Number of False Negatives: {fn}.")
    print(
        f"Number of Matching Detections: {no_matching}, Number of detected ground truths: {no_detected}.")
    print(f"Recall: {recall}, Precision: {prec}, F1-score: {f1score}.")

    return fp, fn, recall, prec
