## This script runs the MELD surface-based FCD classifier on the patient using the output features from script 2.
## The predicted clusters are then saved as file " " in the /output/<pat_id>/xhemi/classifier folder
## The predicted clusters are then registered back to native space and saved as a .mgh file in the /output/<pat_id>/classifier folder
## The predicted clusters are then registered back to the nifti volume and saved as nifti in the input/<pat_id>/predictions folder
## Individual reports for each identified cluster are calculated and saved in the input/<pat_id>/predictions/reports folder
## These contain images of the clusters on the surface and on the volumetric MRI as well as saliency reports
## The saliency reports include the z-scored feature values and how "salient" they were to the classifier

## To run : python new_pt_pirpine_script3.py -ids <text_file_with_ids> -site <site_code>


import os
import sys
import argparse
import subprocess as sub
import glob
from tempfile import NamedTemporaryFile
from meld_classifier.paths import (
    BASE_PATH,
    FS_SUBJECTS_PATH,
    MELD_DATA_PATH,
    SCRIPTS_DIR,
    NEWSUBJECTS_DATASET,
    DEFAULT_HDF5_FILE_ROOT,
)
from meld_classifier.predict_newsubject import predict_subjects
from scripts.manage_results.move_predictions_to_mgh import move_predictions_to_mgh
import numpy as np

from scripts.manage_results.plot_prediction_report import generate_prediction_report


def predict_new_subjects(
    subject_ids, site_code, subjects_dir=FS_SUBJECTS_PATH, experiment_folder="output/classifier_outputs"
):
    scripts_dir = os.path.join(SCRIPTS_DIR, "scripts")

    # Run MELD surface-based FCD classifier on the patient
    new_data_parameters = {
        "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
        "dataset": NEWSUBJECTS_DATASET,
        "saved_hdf5_dir": f"{MELD_DATA_PATH}/output/classifier_outputs",
    }

    print(new_data_parameters["hdf5_file_root"])
    print(new_data_parameters["dataset"])
    print(new_data_parameters["saved_hdf5_dir"])

    predict_subjects(subject_ids, new_data_parameters, plot_images=True, saliency=True)

    # Register predictions to native space
    # command = format(f"python {scripts_dir}/manage_results/move_predictions_to_mgh.py --experiment_folder {exp_fold} --subjects_dir {subjects_dir} --list_ids {subject_ids}")
    # sub.check_call(command, shell=True)

    move_predictions_to_mgh(subject_ids, subjects_dir, experiment_folder)

    # temporary file with subject ids:

    f = NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    listpath = f.name
    f.write("\n".join(list(subject_ids)))
    f.close()

    # Register prediction back to nifti volume
    output_dir = os.path.join(MELD_DATA_PATH, "input")
    for s in subject_ids:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)
    print("output_dir", output_dir)
    command = format(
        f"bash {scripts_dir}/manage_results/register_back_to_xhemi.sh {subjects_dir} {listpath} {output_dir} {scripts_dir}/manage_results"
    )
    sub.check_call(command, shell=True)

    # Create individual reports of each identified cluster
    # command = format(
    #     f"python {scripts_dir}/manage_results/plot_prediction_report.py --experiment_folder {experiment_folder} --subjects_dir {MELD_DATA_PATH} --list_ids {listpath}"
    # )
    # sub.check_call(command, shell=True)

    generate_prediction_report(subject_ids, experiment_folder=experiment_folder, subjects_dir=MELD_DATA_PATH)

    os.unlink(listpath)


if __name__ == "__main__":

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-ids",
        "--list_ids",
        help="Subjects IDs in a text file",
        required=True,
    )
    parser.add_argument(
        "-site",
        "--site_code",
        help="Site code",
        required=True,
    )
    args = parser.parse_args()
    ids_list = str(args.list_ids)
    site_code = str(args.site_code)
    scripts_dir = os.path.join(SCRIPTS_DIR, "scripts")
    exp_fold = "output/classifier_outputs"

    subject_ids = np.loadtxt(ids_list, dtype=str)

    predict_new_subjects(subject_ids, FS_SUBJECTS_PATH, site_code, exp_fold)
