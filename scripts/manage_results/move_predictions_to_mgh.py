#%%
import numpy as np
import h5py
import pandas as pd
import matplotlib_surface_plotting as msp
from meld_classifier.meld_cohort import MeldCohort
import os
import meld_classifier.paths as paths
import nibabel as nb
import argparse
import matplotlib.pyplot as plt
#%%

def load_prediction(subject, hdf5):
    results = {}
    with h5py.File(hdf5, "r") as f:
        for hemi in ["lh", "rh"]:
            results[hemi] = f[subject][hemi]["prediction"][:]
    return results


def save_mgh(filename, array, demo):
    """save mgh file using nibabel and imported demo mgh file"""
    mmap = np.memmap("/tmp/tmp", dtype="float32", mode="w+", shape=demo.get_data().shape)
    mmap[:, 0, 0] = array[:]
    output = nb.MGHImage(mmap, demo.affine, demo.header)
    nb.save(output, filename)
    
def move_predictions_to_mgh(subject_ids,subjects_dir,experiment_basepath, experiment_name='ensemble_iteration', fold=None):

    if fold == None: 
        prediction_file = os.path.join(paths.MELD_DATA_PATH,
            experiment_basepath, "results", f"predictions_{experiment_name}.hdf5"
        )
    else : 
        prediction_file = os.path.join(paths.MELD_DATA_PATH,
            experiment_basepath, f"fold_{fold}", "results", f"predictions_{experiment_name}.hdf5"
        )
    
    hemis = ["lh", "rh"]
    c = MeldCohort()
    vertices = c.surf_partial["coords"]
    faces = c.surf_partial["faces"]
    subjectfiles_list = os.listdir(subjects_dir)

    if os.path.isfile(prediction_file):
        for subject in subject_ids:
            if subject in subjectfiles_list:
                print(subject)
                # create classifier directo ry if not exist
                classifier_dir = os.path.join(subjects_dir, subject, "xhemi", "classifier")
                if not os.path.isdir(classifier_dir):
                    os.mkdir(classifier_dir)
                predictions = load_prediction(subject, prediction_file)
                for hemi in hemis:
                    prediction_h = predictions[hemi]
                    overlay = np.zeros_like(c.cortex_mask, dtype=int)
                    overlay[c.cortex_mask] = prediction_h
                    demo = nb.load(os.path.join(subjects_dir, subject, "xhemi", "surf_meld", f"{hemi}.on_lh.thickness.mgh"))
                    filename = os.path.join(subjects_dir, subject, "xhemi", "classifier", f"{hemi}.prediction.mgh")
                    save_mgh(filename, overlay, demo)
                    print(f"prediction saved at {filename}")
            else:
                print(f"Subject {subject} does not have a freesurfer folder at {subjects_dir}")

#%%
if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="create mgh file with predictions from hdf5 arrays")
    parser.add_argument(
        "--experiment-folder",
        help="Experiments folder",
    )
    parser.add_argument(
        "--experiment_name",
        help="subfolder to use, typically the ensemble model",
        default="ensemble_iteration",
    )
    parser.add_argument("--fold", default=None, help="fold number to use (by default all)")
    parser.add_argument(
        "--subjects_dir", default="", help="folder containing freesurfer outputs. It will store predictions there"
    )
    parser.add_argument("--list_ids", default=None, help="texte file containing list of ids to process")

    args = parser.parse_args()


    experiment_basepath = args.experiment_folder #os.path.join(paths.MELD_DATA_PATH, args.experiment_folder)

    if args.list_ids:
        subject_ids = np.loadtxt(args.list_ids, dtype="str", ndmin=1)
    # else:
    #     df = pd.read_csv(result_file, index_col=False)
    #     subject_ids = np.array(df["ID"])
    subjects_dir = args.subjects_dir
    fold = args.fold
    experiment_name = args.experiment_name

    move_predictions_to_mgh(subject_ids,subjects_dir, experiment_basepath, experiment_name, fold) 
# %%
