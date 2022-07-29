# this pipeline should simplyfy the whole meld process
# specifically differentiate parts that use freesurfer and that dont so things can be parallelized properly

# it calls:
# (new_pt_pipeline_script1.py)
# ini_freesurfer
# fastsurfer_subject
# create_xhemi
# (sample_FLAIR_smooth_features)
# (move_to_xhemi_flip)
# (lesion_labels)
# (new_pt_pipeline_script2.py)
#
#
#
#
#
#
#
# (new_pt_pipeline_script3.py)
#


#%% imports
from genericpath import isfile
import os
from os.path import join as opj
from subprocess import Popen, check_call, DEVNULL, STDOUT
import shutil
import pandas as pd
import multiprocessing
from functools import partial
import time
from datetime import datetime

# from multiprocessing import set_start_method
# set_start_method("spawn")
from meld_classifier.paths import BASE_PATH, NEWSUBJECTS_DATASET, CLIPPING_PARAMS_FILE

import numpy as np
import nibabel as nb
import argparse
from scripts.data_preparation.extract_features import io_meld

from new_patient_pipeline.new_pt_pipeline_script1 import fastsurfer_subject, fastsurfer_flair, init
from scripts.data_preparation.extract_features.create_xhemi import run_parallel, create_xhemi
from scripts.data_preparation.extract_features.create_identity_reg import create_identity
from scripts.new_patient_pipeline.new_pt_pipeline_script3 import predict_new_subjects

def sample_flair_smooth_features(subject, subjects_dir):
    # if [ ! -e "$s"/surf_meld/rh.w-g.pct.mgh ];  then

    ds_features_to_generate = []
    dswm_features_to_generate = []

    os.makedirs(opj(subjects_dir, subject, "surf_meld"), exist_ok=True)

    create_identity(opj(subjects_dir, subject))

    for h in ["rh", "lh"]:
        for d in [0.5, 0.25, 0.75, 0]:
            if not os.path.isfile(f"{subjects_dir}/{subject}/surf_meld/{h}.gm_FLAIR_{d}.mgh"):
                ds_features_to_generate.append((h, d))
        for dwm in [0.5, 1]:
            if not os.path.isfile(f"{subjects_dir}/{subject}/surf_meld/{h}.wm_FLAIR_{dwm}.mgh"):
                dswm_features_to_generate.append((h, dwm))

        print(ds_features_to_generate)

        for dsf in ds_features_to_generate:
            # sampling volume to surface
            # mri_vol2surf --src "$s"/mri/FLAIR.mgz --out "$s"/surf_meld/"$h".gm_FLAIR_"$d".mgh --hemi "$h" --projfrac "$d" --srcreg "$s"/mri/transforms/Identity.dat --trgsubject "$s" --surf white
            hemi = dsf[0]
            d = dsf[1]
            command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject}/mri/FLAIR.mgz --out {subjects_dir}/{subject}/surf_meld/{hemi}.gm_FLAIR_{d}.mgh --hemi {hemi} --projfrac {d} --srcreg {subjects_dir}/{subject}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject} --surf white"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()

        print(dswm_features_to_generate)
        # Sample FLAIR 0.5mm and 1mm subcortically & smooth using 10mm Gaussian kernel
        for dswmf in dswm_features_to_generate:
            hemi = dswmf[0]
            dwm = dswmf[1]
            # mri_vol2surf --src "$s"/mri/FLAIR.mgz --out "$s"/surf_meld/"$h".wm_FLAIR_"$d_wm".mgh --hemi "$h" --projdist -"$d_wm" --srcreg "$s"/mri/transforms/Identity.dat --trgsubject "$s" --surf white
            command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject}/mri/FLAIR.mgz --out {subjects_dir}/{subject}/surf_meld/{hemi}.wm_FLAIR_{dwm}.mgh --hemi {hemi} --projdist -{dwm} --srcreg {subjects_dir}/{subject}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject} --surf white"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()

    hemispheres = ["rh", "lh"]

    # if os.path.isfile(opj(subjects_dir,subject,'surf_meld/lh.w-g.pct.mgh')):
    #     print('sample_flair_smooth_features on lh already run for', subject)
    # else:
    #     hemispheres.append('lh')

    # if os.path.isfile(opj(subjects_dir,subject,'surf_meld/rh.w-g.pct.mgh')):
    #     print('sample_flair_smooth_features on rh already run for', subject)
    # else:
    #     hemispheres.append('rh')

    # python "$script_dir"/create_identity_reg.py "$s"
    #
    # mkdir "$s"/surf_meld
    #
    #  H="lh rh"

    if not os.path.isfile(opj(subjects_dir, subject, "mri", "FLAIR.mgz")):
        print("FLAIR.mgz not found for subject", subject)
        return -1

    for hemi in hemispheres:

        # Calculate curvature
        # mris_curvature_stats -f white -g --writeCurvatureFiles "$s" "$h" curv
        command = (
            f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f white -g --writeCurvatureFiles {subject} {hemi} curv"
        )
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # mris_curvature_stats -f pial -g --writeCurvatureFiles "$s" "$h" curv
        command = (
            f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f pial -g --writeCurvatureFiles {subject} {hemi} curv"
        )
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

        # Convert mean curvature and sulcal depth to .mgh file type
        # mris_convert -c "$s"/surf/"$h".curv "$s"/surf/"$h".white "$s"/surf_meld/"$h".curv.mgh
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject}/surf/{hemi}.curv {subjects_dir}/{subject}/surf/{hemi}.white {subjects_dir}/{subject}/surf_meld/{hemi}.curv.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # mris_convert -c "$s"/surf/"$h".sulc "$s"/surf/"$h".white "$s"/surf_meld/"$h".sulc.mgh
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject}/surf/{hemi}.sulc {subjects_dir}/{subject}/surf/{hemi}.white {subjects_dir}/{subject}/surf_meld/{hemi}.sulc.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # mris_convert -c "$s"/surf/"$h".pial.K.crv "$s"/surf/"$h".white "$s"/surf_meld/"$h".pial.K.mgh
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject}/surf/{hemi}.pial.K.crv {subjects_dir}/{subject}/surf/{hemi}.white {subjects_dir}/{subject}/surf_meld/{hemi}.pial.K.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # echo "Filtering and smoothing intrinsic curvature"
        # python "$script_dir"/filter_intrinsic_curvature.py
        input = opj(subjects_dir, subject, "surf_meld", f"{hemi}.pial.K.mgh")
        output = opj(subjects_dir, subject, "surf_meld", f"{hemi}.pial.K_filtered.mgh")
        demo = nb.load(input)
        curvature = io_meld.load_mgh(input)
        curvature = np.absolute(curvature)
        curvature = np.clip(curvature, 0, 20)
        io_meld.save_mgh(output, curvature, demo)

        # mris_fwhm --s "$s" --hemi "$h" --cortex --smooth-only --fwhm 20\
        # --i "$s"/surf_meld/"$h".pial.K_filtered.mgh --o "$s"/surf_meld/"$h".pial.K_filtered.sm20.mgh
        command = f"SUBJECTS_DIR={subjects_dir} mris_fwhm --s {subject} --hemi {hemi} --cortex --smooth-only --fwhm 20 --i {subjects_dir}/{subject}/surf_meld/{hemi}.pial.K_filtered.mgh --o {subjects_dir}/{subject}/surf_meld/{hemi}.pial.K_filtered.sm20.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

        # mris_convert -c "$s"/surf/"$h".thickness "$s"/surf/"$h".white "$s"/surf_meld/"$h".thickness.mgh
        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject}/surf/{hemi}.thickness {subjects_dir}/{subject}/surf/{hemi}.white {subjects_dir}/{subject}/surf_meld/{hemi}.thickness.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

        # cp "$s"/surf/"$h".w-g.pct.mgh "$s"/surf_meld/"$h".w-g.pct.mgh
        shutil.copy(
            opj(subjects_dir, subject, "surf", f"{hemi}.w-g.pct.mgh"),
            opj(subjects_dir, subject, "surf_meld", f"{hemi}.w-g.pct.mgh"),
        )


def move_to_xhemi_flip(subject, subjects_dir):
    measures = [
        "thickness.mgh",
        "w-g.pct.mgh",
        "curv.mgh",
        "sulc.mgh",
        "gm_FLAIR_0.75.mgh",
        "gm_FLAIR_0.5.mgh",
        "gm_FLAIR_0.25.mgh",
        "gm_FLAIR_0.mgh",
        "wm_FLAIR_0.5.mgh",
        "wm_FLAIR_1.mgh",
        "pial.K_filtered.sm20.mgh",
    ]

    os.makedirs(f"{subjects_dir}/{subject}/xhemi/surf_meld/", exist_ok=True)

    if not os.path.isfile(f"{subjects_dir}/{subject}/xhemi/surf_meld/zeros.mgh"):
        # create one all zero overlay for inversion step
        shutil.copy(
            f"{subjects_dir}/fsaverage_sym/surf/lh.white.avg.area.mgh",
            f"{subjects_dir}/{subject}/xhemi/surf_meld/zeros.mgh",
        )

        command = f"SUBJECTS_DIR='' mris_calc --output {subjects_dir}/{subject}/xhemi/surf_meld/zeros.mgh {subjects_dir}/{subject}/xhemi/surf_meld/zeros.mgh set 0"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

    for measure in measures:
        if not os.path.isfile(f"{subjects_dir}/{subject}/xhemi/surf_meld/lh.on_lh.{measure}"):
            command = f"SUBJECTS_DIR='' mris_apply_reg --src {subjects_dir}/{subject}/surf_meld/lh.{measure} --trg {subjects_dir}/{subject}/xhemi/surf_meld/lh.on_lh.{measure} --streg {subjects_dir}/{subject}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()

        if not os.path.isfile(f"{subjects_dir}/{subject}/xhemi/surf_meld/rh.on_lh.{measure}"):
            command = f"SUBJECTS_DIR='' mris_apply_reg --src {subjects_dir}/{subject}/surf_meld/rh.{measure} --trg {subjects_dir}/{subject}/xhemi/surf_meld/rh.on_lh.{measure} --streg {subjects_dir}/{subject}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()


def lesion_labels(subject, subjects_dir):
    if os.path.isfile(f"{subjects_dir}/{subject}/surf_meld/lh.lesion_linked.mgh"):
        print("isfile")
        if not os.path.isfile(f"{subjects_dir}/{subject}/xhemi/surf_meld/lh.on_lh.lesion.mgh"):
            print("isnotfile")
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subject}/surf_meld/lh.lesion_linked.mgh --trg {subject}/xhemi/surf_meld/lh.on_lh.lesion.mgh --streg {subjects_dir}/{subject}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()
        print("isfile")

    elif os.path.isfile(f"{subjects_dir}/{subject}/surf_meld/rh.lesion_linked.mgh"):
        print("isfile")
        if not os.path.isfile(f"{subjects_dir}/{subject}/xhemi/surf_meld/rh.on_lh.lesion.mgh"):
            print("isnotfile")
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subject}/surf_meld/rh.lesion_linked.mgh --trg {subject}/xhemi/surf_meld/rh.on_lh.lesion.mgh --streg {subjects_dir}/{subject}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()
        print("isfile")


def create_training_data(subject, subjects_dir, output_dir, cortex_label=None):
    features = np.array(
        [
            ".on_lh.thickness.mgh",
            ".on_lh.w-g.pct.mgh",
            ".on_lh.curv.mgh",
            ".on_lh.sulc.mgh",
            ".on_lh.gm_FLAIR_0.75.mgh",
            ".on_lh.gm_FLAIR_0.5.mgh",
            ".on_lh.gm_FLAIR_0.25.mgh",
            ".on_lh.gm_FLAIR_0.mgh",
            ".on_lh.wm_FLAIR_0.5.mgh",
            ".on_lh.wm_FLAIR_1.mgh",
            ".on_lh.pial.K_filtered.sm20.mgh",
        ]
    )
    n_vert = 163842
    if cortex_label is None:
        cortex_label = nb.freesurfer.io.read_label(os.path.join(subjects_dir, "fsaverage_sym/label/lh.cortex.label"))

    medial_wall = np.delete(np.arange(n_vert), cortex_label)

    print("saving subject " + subject + "...")
    io_meld.save_subject(subject, features, medial_wall, subjects_dir, output_dir)


#%% code


def fast_freesurfer_preprocessing_single_subject(subject, subjects_dir, site_code="", init=None, cortex_label=None):

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("processing subject", subject, "started at", dt_string)
    starting_time = time.time()
    # ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    # proc = check_call(ini_freesurfer, shell=True)
    print("START SURFING FOR ", subject)
    if init:
        init(multiprocessing.Lock())
    print("STEP 1: FASTSURFER")
    fastsurfer_subject(subject, subjects_dir)
    print("STEP 2: FREESURFER PIAL")
    fastsurfer_flair(subject, subjects_dir)
    print("STEP 3: CREATE XHEMI")
    create_xhemi(subject, subjects_dir)
    print("STEP 4: SAMPLE FLAIR SMOOTH FEATURES")
    sample_flair_smooth_features(subject, subjects_dir)
    print("STEP 5: MOVE TO XHEMI FLIP")
    move_to_xhemi_flip(subject, subjects_dir)
    print("STEP 6: LESION LABELS")
    lesion_labels(subject, subjects_dir)
    t_sec = round(time.time() - starting_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print("DONE SURFING FOR ", subject, f"took {t_hour}hour:{t_min}min:{t_sec}sec")


from scripts.data_preparation.run_data_smoothing_new_subjects import create_dataset_file
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.meld_cohort import MeldCohort


def smooth_features(subjects, output_dir):
    features = {
        ".on_lh.thickness.mgh": 10,
        ".on_lh.w-g.pct.mgh": 10,
        ".on_lh.pial.K_filtered.sm20.mgh": None,
        ".on_lh.sulc.mgh": 5,
        ".on_lh.curv.mgh": 5,
        ".on_lh.gm_FLAIR_0.25.mgh": 10,
        ".on_lh.gm_FLAIR_0.5.mgh": 10,
        ".on_lh.gm_FLAIR_0.75.mgh": 10,
        ".on_lh.gm_FLAIR_0.mgh": 10,
        ".on_lh.wm_FLAIR_0.5.mgh": 10,
        ".on_lh.wm_FLAIR_1.mgh": 10,
    }
    feat = Feature()
    features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
    dataset_newSubject = os.path.join(BASE_PATH, NEWSUBJECTS_DATASET)

    create_dataset_file(subjects, dataset_newSubject)
    c_raw = MeldCohort(
        hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=dataset_newSubject, data_dir=output_dir
    )

    smoothing = Preprocess(
        c_raw, write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed.hdf5", data_dir=output_dir
    )
    for feature in np.sort(list(set(features))):
        print(feature)
        smoothing.smooth_data(feature, features[feature], clipping_params=CLIPPING_PARAMS_FILE)


from scripts.data_preparation.run_data_processing_new_subjects import process_new_subjects

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
    parser.add_argument(
        "-id",
        "--subject_id",
        help="Subject ID.",
        default="",
        required=False,
    )
    parser.add_argument(
        "-sd",
        "--subjects_dir",
        help="Subjects directory.",
        default="",
        required=False,
    )
    parser.add_argument(
        "-ids",
        "--subject_ids",
        help="Subject IDs.",
        default="",
        required=False,
    )

    args = parser.parse_args()

    sid = args.subject_id
    sd = args.subjects_dir
    sids = args.subject_ids

    if sd == "":
        print("Please provide a subjects directory path with --sd ")
        os.sys.exit(-1)
    starting = None
    if sid != "":
        fast_freesurfer_preprocessing_single_subject(sid, sd, init=init)
        os.sys.exit(0)

    if sids != "":
        try:
            df = pd.read_csv(sids)
        except:
            print("ERROR reading", sids)
            os.sys.exit(-1)

    num_procs = 16

    print("Processing", len(df.participant_id.values), "with", num_procs, "processes")

    if not starting:
        init(multiprocessing.Lock())

    cortex_label = nb.freesurfer.io.read_label(os.path.join(sd, "fsaverage_sym/label/lh.cortex.label"))

    # with multiprocessing.Pool(num_procs) as pool:
    #     pool.map(partial(fast_freesurfer_preprocessing_single_subject, subjects_dir=sd, cortex_label=cortex_label), df.participant_id.values)

    # for sub in df.participant_id.values:
    # fast_freesurfer_preprocessing_single_subject(sub, sd)

    print("STEP 7: CREATE TRAINING DATA")
    # site_code = ""
    # for sub in df.participant_id.values:
    #     if site_code == "":
    #         try:
    #             site_code = sub.split("_")[1]  ##according to current MELD naming convention TODO
    #         except ValueError:
    #             print("Could not recover site code from", sub)
    #             os.sys.exit(-1)

    #     output_dir = os.path.join(BASE_PATH, f"MELD_{site_code}")
    #     print(BASE_PATH)
    #     os.makedirs(output_dir, exist_ok=True)

    #     create_training_data(sub, sd, output_dir, cortex_label)

    print("STEP 8: smooth features and process_new_subjects")
    chunked_subject_list = list()
    chunk_size = 5
    for i in range(0, len(df.participant_id.values), chunk_size):
        chunked_subject_list.append(df.participant_id.values[i : i + chunk_size])

    # for chunk in chunked_subject_list:
    #     smooth_features(chunk, BASE_PATH)
    #     smooth_features(chunk, BASE_PATH)
    # process_new_subjects(df.participant_id.values, "H101", BASE_PATH)


    for chunk in chunked_subject_list:
        predict_new_subjects(chunk, site_code='H101' )


# %%
