from meld_classifier.evaluation import Evaluator
from meld_classifier.experiment import Experiment, load_config
from meld_classifier.meld_cohort import MeldCohort
import argparse
import numpy as np
import json
import os
import importlib
import sys
import h5py
from meld_classifier.paths import BASE_PATH, EXPERIMENT_PATH, MELD_DATA_PATH, MODEL_PATH, MODEL_NAME
import subprocess
import pandas as pd


def create_dataset_file(subjects_ids, output_path):
    df = pd.DataFrame()
    # subjects_ids = [subject for subject in subjects_ids]
    df["subject_id"] = subjects_ids
    df["split"] = ["test" for _ in subjects_ids]
    df.to_csv(output_path)
    return df


def predict_subjects(subjects_ids, new_data_parameters, site_codes = None, plot_images=False, saliency=False):
    # read subjects
    # subjects_ids = np.loadtxt(list_ids, dtype="str", ndmin=1)
    # create dataset csv
    create_dataset_file(subjects_ids, os.path.join(BASE_PATH, new_data_parameters["dataset"]))
    # load models
    experiment_path = os.path.join(EXPERIMENT_PATH, MODEL_PATH)
    exp = Experiment(experiment_path=experiment_path, experiment_name=MODEL_NAME)
    exp.init_logging()
    # load information to predict on new subjects
    exp.cohort = MeldCohort(
        hdf5_file_root=new_data_parameters["hdf5_file_root"], dataset=new_data_parameters["dataset"], site_codes=site_codes
    )
    subjects = exp.cohort.get_meld_subjects(site_codes=site_codes, lesional_only=False)
    # subject_ids = [s.subject_id for s in subjects]
    print(f"Predicting {len(subjects)} Subjects ...")
    save_dir = new_data_parameters["saved_hdf5_dir"]
    # create sub-folders if do not exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
    if plot_images:
        os.makedirs(os.path.join(save_dir, "results", "images"), exist_ok=True)
    # launch evaluation
    eva = Evaluator(exp, mode="inference", subjects=subjects, save_dir=save_dir)
    for subject in subjects:
        eva.load_predict_single_subject(subject, fold="", plot=plot_images, saliency=saliency, suffix="")
