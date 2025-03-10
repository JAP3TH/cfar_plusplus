import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import re

from os import listdir
from os.path import exists, join


def get_all_scenarios(self) -> [str]:
    """Lists all scenario names in the annotation directory.

    Returns:
        [str]: List of scenario names.
    """

    scenario_list = [anno_file[:-4]
                     for anno_file in listdir(self.dir_annotations) if anno_file.endswith(".txt")]

    return scenario_list


def get_rod2021_statistics(self):
    """Get the statistics of the ROD2021 data set. Counts all pedestrian and bicyclists ground truth objects and print them.
    """

    total_pedestrian_objects = 0  # number of frames
    total_pedestrian_scenes = 0  # number of scenes
    total_cyclist_objects = 0  # number of frames
    total_cyclist_scenes = 0  # number of scenes

    scenarios_to_play = get_all_scenarios(self)

    # play all files of the list
    for scenario in scenarios_to_play:
        # select first scenario from the list
        self.scenario_loader(scenario)
        if 'pedestrian' in self.annotations["label"].values:
            total_pedestrian_scenes += 1
            total_pedestrian_objects += (
                self.annotations["label"] == 'pedestrian').sum()
        if 'cyclist' in self.annotations["label"].values:
            total_cyclist_scenes += 1
            total_cyclist_objects += (
                self.annotations["label"] == 'cyclist').sum()

    print(
        f"Total pedestrian scenes: {total_pedestrian_scenes}, total bicycle scenes: {total_cyclist_scenes}.")
    print(
        f"Total pedestrian objects: {total_pedestrian_objects}, total bicycle objects: {total_cyclist_objects}.")


def get_detection_metrics(dir_gt: str, dir_det: str, label_filter: list):
    """Determines the following metrics from a CSV log:
    Recall, precision, number of false positives, number of false negatives.   

    Args:
        dir_gt (str): Directory where Ground Truth CSV log is stored.
        dir_det (str): Directory where Detection CSV log is stored.
        label_filter (list): List of strings which labels shall be included in the metrics. 0=person, 1=bicyclist, 2=car

    Returns:
        fp, fn, recall, precision: Two integers for number of False Positives/Negatives and two floats for recall and precision value.
    """

    try:
        ground_truth = pd.read_csv(dir_gt)
    except ValueError:
        print("Ground truth CSV file cannot be read, please check for file consistency.")

    try:
        detections = pd.read_csv(dir_det)
    except ValueError:
        print("Ground truth CSV file cannot be read, please check for file consistency.")

    # filter ground truth and detections
    ground_truth = ground_truth[ground_truth.label_num.isin(label_filter)]

    # count false positives and false negatives
    no_matching = detections.loc[:, "matching_gt"].sum()
    no_detected = ground_truth.loc[:, "detected"].sum()
    # print(f"no_matching {no_matching}, no_dets {len(detections)}")
    # print(f"no_detected {no_detected}, no_gt {len(ground_truth)}")
    recall = no_detected / len(ground_truth)
    prec = no_matching / len(detections)
    f1score = 2 * prec * recall / (prec + recall)
    fp = len(detections) - no_matching
    fn = len(ground_truth) - no_detected

    # now get all metrics just for critical points
    ground_truth_crit = ground_truth[ground_truth.criticality > 0.0]
    detections_crit = detections[detections.criticality > 0.0]
    no_crits_matching = detections_crit.loc[:, "matching_gt"].sum()
    no_crits_detected = ground_truth_crit.loc[:, "detected"].sum()
    recall_c = no_crits_detected / len(ground_truth_crit)
    prec_c = no_crits_matching / len(detections_crit)
    fp_c = len(detections_crit) - no_crits_matching
    fn_c = len(ground_truth_crit) - no_crits_detected

    print(
        f"Total ground truth objects: {len(ground_truth)}, critical gts: {len(ground_truth_crit)}, total detections: {len(detections)}.")
    print(f"False Positives: {fp}, False Negatives: {fn}.")
    print(
        f"Matching Detections: {no_matching}, detected ground truths: {no_detected}.")
    print(f"Recall: {recall}, Precision: {prec}, F1-score: {f1score}.")
    print("----------------------------------------------------------")
    print(
        f"Critical False Positives: {fp_c}, Critical False Negatives: {fn_c}.")
    print(
        f"Matching Critical Detections: {no_crits_matching}, Detected Critical Ground Truths: {no_crits_detected}.")
    print(f"Recall: {recall_c}, Precision: {prec_c}.")
    qresult = ground_truth.query(
        'criticality > 0.0 and detected == 0')['frame']
    print(f"False Negative frames with criticality: {qresult}")

    return fp, fn, recall, prec


def create_baseline_csv(dir_export: Path):
    """Evaluate CFAR for all parameters by looping all logs and store KPIs for each parameter set.
        Creates the baseline.csv file which is a mapping of the pandas data frame.

    Args:
        dir_export (Path): The export directory where your CSV files with varying parameters are stored. The file baseline.csv will also be stored in this directory.  
    """

    results_df = pd.DataFrame(
        columns=['file', 'fp', 'fn', 'f1', 'prec', 'recall'])

    labellist = [0, 1]

    # Store the pairs in a dictionary
    file_pairs = {}

    # File name pattern
    pattern = r"_(avr\d+_ava\d+_gr\d+_ga\d+)_"

    # Loop files and store the combinations of det and gt
    for file in listdir(dir_export):
        # find the correct files in the export directory
        match = re.search(pattern, file)
        if match:
            key = match.group(1)
            if "_detections" in file:
                file_pairs.setdefault(key, {})["det"] = file
            elif "_gt" in file:
                file_pairs.setdefault(key, {})["gt"] = file

    # Loop through the paris and store the parameters
    for i, (key, pair) in enumerate(tqdm(file_pairs.items())):
        print(i)
        file_gt = pair.get("gt")
        file_det = pair.get("det")
        if file_gt and file_det:
            file_gt_path = join(dir_export, file_gt)
            file_det_path = join(dir_export, file_det)
            fp, fn, recall, prec = get_detection_metrics(
                file_gt_path, file_det_path, labellist)
            f1 = 2 * prec * recall / (prec + recall)
            # write metrics to results data frame
            results_df.loc[i] = [
                file_gt_path[len(dir_export)+17:-4], fp, fn, f1, recall, prec]

    # sort results ascending by
    results_df = results_df.sort_values(by=['f1'])

    # store results dataframe
    results_df.to_csv(join(dir_export, 'baseline.csv'), index=False)


def plot_baseline_experiments(dir_export: Path):
    """Reads "baseline.csv" file generated by "get_baseline", analyses the results, and plots a precision-recall curve. 

    Args:
        dir_export (Path): The export directory where the file baseline.csv is stored.
    """
    results_df = pd.read_csv(join(dir_export, 'baseline.csv'))

    print(results_df)
    print(
        f"Max. recall: {results_df['recall'].max()}, max. precision: {results_df['prec'].max()}.")
    print(
        f"Min. false positives: {results_df['fp'].min()}, min. false negatives: {results_df['fn'].min()}.")

    print("Configuration with minimal false negatives:")
    print(results_df[results_df['fn'] == results_df['fn'].min()])

    plt.figure()
    plt.scatter(results_df['recall'], results_df['prec'], s=5)
    plt.grid()
    plt.show()


def plot_spectrum(self, chirp_id: int = 0):
    """_summary_

    Args:
        chirp_id (int, optional): _description_. Defaults to 0.
    """

    plt.figure()

    spectrum = self.pow_spec_abs_hist[:, :, chirp_id, -1]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(self.r_arange, self.ang_arange, spectrum, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.grid()
    plt.show()
