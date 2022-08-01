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
from tempfile import NamedTemporaryFile, TemporaryFile
import pandas as pd
import multiprocessing
from functools import partial
import time
from datetime import datetime
from bids import BIDSLayout

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

from scripts.data_preparation.run_data_smoothing_new_subjects import create_dataset_file
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.meld_cohort import MeldCohort
from scripts.data_preparation.run_data_processing_new_subjects import process_new_subjects
from scripts.new_patient_pipeline.new_site_harmonisation_script import run_site_harmonization
#%% code

def sample_flair_smooth_features(subject_id, subjects_dir):
    # if [ ! -e "$s"/surf_meld/rh.w-g.pct.mgh ];  then

    ds_features_to_generate = []
    dswm_features_to_generate = []

    os.makedirs(opj(subjects_dir, subject_id, "surf_meld"), exist_ok=True)

    create_identity(opj(subjects_dir, subject_id))

    for h in ["rh", "lh"]:
        for d in [0.5, 0.25, 0.75, 0]:
            if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{h}.gm_FLAIR_{d}.mgh"):
                ds_features_to_generate.append((h, d))
        for dwm in [0.5, 1]:
            if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{h}.wm_FLAIR_{dwm}.mgh"):
                dswm_features_to_generate.append((h, dwm))

        print(ds_features_to_generate)

        for dsf in ds_features_to_generate:
            # sampling volume to surface
            # mri_vol2surf --src "$s"/mri/FLAIR.mgz --out "$s"/surf_meld/"$h".gm_FLAIR_"$d".mgh --hemi "$h" --projfrac "$d" --srcreg "$s"/mri/transforms/Identity.dat --trgsubject "$s" --surf white
            hemi = dsf[0]
            d = dsf[1]
            command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject_id}/mri/FLAIR.mgz --out {subjects_dir}/{subject_id}/surf_meld/{hemi}.gm_FLAIR_{d}.mgh --hemi {hemi} --projfrac {d} --srcreg {subjects_dir}/{subject_id}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject_id} --surf white"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()

        print(dswm_features_to_generate)
        # Sample FLAIR 0.5mm and 1mm subcortically & smooth using 10mm Gaussian kernel
        for dswmf in dswm_features_to_generate:
            hemi = dswmf[0]
            dwm = dswmf[1]
            # mri_vol2surf --src "$s"/mri/FLAIR.mgz --out "$s"/surf_meld/"$h".wm_FLAIR_"$d_wm".mgh --hemi "$h" --projdist -"$d_wm" --srcreg "$s"/mri/transforms/Identity.dat --trgsubject "$s" --surf white
            command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject_id}/mri/FLAIR.mgz --out {subjects_dir}/{subject_id}/surf_meld/{hemi}.wm_FLAIR_{dwm}.mgh --hemi {hemi} --projdist -{dwm} --srcreg {subjects_dir}/{subject_id}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject_id} --surf white"
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

    if not os.path.isfile(opj(subjects_dir, subject_id, "mri", "FLAIR.mgz")):
        print("FLAIR.mgz not found for subject", subject_id)
        return -1

    for hemi in hemispheres:

        # Calculate curvature
        # mris_curvature_stats -f white -g --writeCurvatureFiles "$s" "$h" curv
        command = (
            f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f white -g --writeCurvatureFiles {subject_id} {hemi} curv"
        )
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # mris_curvature_stats -f pial -g --writeCurvatureFiles "$s" "$h" curv
        command = (
            f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f pial -g --writeCurvatureFiles {subject_id} {hemi} curv"
        )
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

        # Convert mean curvature and sulcal depth to .mgh file type
        # mris_convert -c "$s"/surf/"$h".curv "$s"/surf/"$h".white "$s"/surf_meld/"$h".curv.mgh
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.curv {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.curv.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # mris_convert -c "$s"/surf/"$h".sulc "$s"/surf/"$h".white "$s"/surf_meld/"$h".sulc.mgh
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.sulc {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.sulc.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # mris_convert -c "$s"/surf/"$h".pial.K.crv "$s"/surf/"$h".white "$s"/surf_meld/"$h".pial.K.mgh
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.pial.K.crv {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()
        # echo "Filtering and smoothing intrinsic curvature"
        # python "$script_dir"/filter_intrinsic_curvature.py
        input = opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.pial.K.mgh")
        output = opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.pial.K_filtered.mgh")
        demo = nb.load(input)
        curvature = io_meld.load_mgh(input)
        curvature = np.absolute(curvature)
        curvature = np.clip(curvature, 0, 20)
        io_meld.save_mgh(output, curvature, demo)

        # mris_fwhm --s "$s" --hemi "$h" --cortex --smooth-only --fwhm 20\
        # --i "$s"/surf_meld/"$h".pial.K_filtered.mgh --o "$s"/surf_meld/"$h".pial.K_filtered.sm20.mgh
        command = f"SUBJECTS_DIR={subjects_dir} mris_fwhm --s {subject_id} --hemi {hemi} --cortex --smooth-only --fwhm 20 --i {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K_filtered.mgh --o {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K_filtered.sm20.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

        # mris_convert -c "$s"/surf/"$h".thickness "$s"/surf/"$h".white "$s"/surf_meld/"$h".thickness.mgh
        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.thickness {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.thickness.mgh"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

        # cp "$s"/surf/"$h".w-g.pct.mgh "$s"/surf_meld/"$h".w-g.pct.mgh
        shutil.copy(
            opj(subjects_dir, subject_id, "surf", f"{hemi}.w-g.pct.mgh"),
            opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.w-g.pct.mgh"),
        )


def move_to_xhemi_flip(subject_id, subjects_dir):
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

    os.makedirs(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/", exist_ok=True)

    if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh"):
        # create one all zero overlay for inversion step
        shutil.copy(
            f"{subjects_dir}/fsaverage_sym/surf/lh.white.avg.area.mgh",
            f"{subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh",
        )

        command = f"SUBJECTS_DIR='' mris_calc --output {subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh {subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh set 0"
        print(command)
        proc = Popen(command, shell=True, stderr=STDOUT)
        proc.wait()

    for measure in measures:
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.{measure}"):
            command = f"SUBJECTS_DIR='' mris_apply_reg --src {subjects_dir}/{subject_id}/surf_meld/lh.{measure} --trg {subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.{measure} --streg {subjects_dir}/{subject_id}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()

        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.{measure}"):
            command = f"SUBJECTS_DIR='' mris_apply_reg --src {subjects_dir}/{subject_id}/surf_meld/rh.{measure} --trg {subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.{measure} --streg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()


def lesion_labels(subject_id, subjects_dir):
    if os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/lh.lesion_linked.mgh"):
        print("isfile")
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.lesion.mgh"):
            print("isnotfile")
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subject_id}/surf_meld/lh.lesion_linked.mgh --trg {subject_id}/xhemi/surf_meld/lh.on_lh.lesion.mgh --streg {subjects_dir}/{subject_id}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()
        print("isfile")

    elif os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/rh.lesion_linked.mgh"):
        print("isfile")
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.lesion.mgh"):
            print("isnotfile")
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subject_id}/surf_meld/rh.lesion_linked.mgh --trg {subject_id}/xhemi/surf_meld/rh.on_lh.lesion.mgh --streg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            print(command)
            proc = Popen(command, shell=True, stderr=STDOUT)
            proc.wait()
        print("isfile")


def create_training_data(subject_id, subjects_dir, output_dir, cortex_label=None, site_code='', group='', scanner=''):
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

    print("saving subject " + subject_id + "...")
    io_meld.save_subject(subject_id, features, medial_wall, subjects_dir, output_dir, site_code=site_code, group=group, scanner=scanner)


def fast_freesurfer_preprocessing_single_subject(subject, subjects_dir, site_code="", init=None, cortex_label=None, use_fastsurfer =True):

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("processing subject", subject, "started at", dt_string)
    starting_time = time.time()
    # ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    # proc = check_call(ini_freesurfer, shell=True)
    print("START SURFING FOR ", subject['id'])
    if init:
        init(multiprocessing.Lock())

    if use_fastsurfer:
        print("STEP 1: FASTSURFER")
        fastsurfer_subject(subject, subjects_dir)
        print("STEP 2: FREESURFER PIAL")
        fastsurfer_flair(subject, subjects_dir)
    else:
        print('USING FREESURFER FOR STEP 1 RECON-ALL and 2 PIAL')
        #TODO
    print("STEP 3: CREATE XHEMI")
    create_xhemi(subject['id'], subjects_dir)
    print("STEP 4: SAMPLE FLAIR SMOOTH FEATURES")
    sample_flair_smooth_features(subject['id'], subjects_dir)
    print("STEP 5: MOVE TO XHEMI FLIP")
    move_to_xhemi_flip(subject['id'], subjects_dir)
    print("STEP 6: LESION LABELS")
    lesion_labels(subject['id'], subjects_dir)
    t_sec = round(time.time() - starting_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print("DONE SURFING FOR ", subject['id'], f"took {t_hour}hour:{t_min}min:{t_sec}sec")


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
        hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=dataset_newSubject, data_dir=BASE_PATH
    )

    smoothing = Preprocess(
        c_raw, write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed.hdf5", data_dir=output_dir
    )
    for feature in np.sort(list(set(features))):
        print(feature)
        smoothing.smooth_data(feature, features[feature], clipping_params=CLIPPING_PARAMS_FILE)


def bids_df_to_dictlist(bids_df):
    dictlist = []
    nifti_selector = bids_df.extension =='.nii.gz'
    flair_selector = (bids_df.suffix =='FLAIR') * nifti_selector
    t1_selector = (bids_df.suffix =='T1w') * nifti_selector

    for s in bids_df.subject.dropna().unique():
        sub_selector = bids_df.subject == s
        flair_paths = bids_df[sub_selector * flair_selector].path
        if len(flair_paths) != 1:
            print('Mutliple FLAIR files found for subject',s)
            flair_path = ''
        else:
            flair_path = flair_paths.iloc[0]

        t1_paths = bids_df[sub_selector * t1_selector].path
        if len(t1_paths) != 1:
            print('Mutliple T1w files found for subject',s)
            t1_path = ''
        else:
            t1_path = t1_paths.iloc[0]
        dictlist.append(
            {
                'id':f'sub-{s}',
                'flair_path':flair_path,
                't1_path':t1_path
            }
        )
    
    return dictlist


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

    parser.add_argument(
        '-bids',
        '--bids_dir',
        help="BIDS directory containing subjects",
        default="",
        required=False,
    )

    #TODO currently not used
    parser.add_argument(
        '--predict_only',
        help="Only run predictions",
        action='store_false',
    )

    #TODO currently not used
    parser.add_argument(
        '-dmp',
        '--demo_file_path',
        default='',
    )


    args = parser.parse_args()

    sid = args.subject_id
    sd = args.subjects_dir
    sids = args.subject_ids
    bids_dir = args.bids_dir

    if sd == "":
        print("Please provide a subjects directory path with --sd ")
        os.sys.exit(-1)
    starting = None

    if sid != "":
        fast_freesurfer_preprocessing_single_subject(sid, sd, init=init)
        os.sys.exit(0)

    if bids_dir != '':
        bids_df = BIDSLayout(bids_dir, validate=False).to_df()
        subjects = bids_df_to_dictlist(bids_df)

        demographic_file = NamedTemporaryFile(mode='w', delete=False)
        demographic_file_path = demographic_file.name
        demographic_file.close()

        participants_df = pd.read_csv(bids_df[ (bids_df.suffix == 'participants')*(bids_df.extension == '.tsv') ].path.iloc[0],sep='\t')

        demo_df = pd.DataFrame()
        demo_df['ID'] = participants_df['participant_id']
        demo_df['Age at preop'] = participants_df['age at scan']
        demo_df['Sex'] = (participants_df['sex']=='Male')*1

        demo_df.to_csv(demographic_file_path, index=False)
    
    
    elif sids != "":
        try:
            df = pd.read_csv(sids)
            subjects = [{'id':s} for s in df.participant_id.values]
        except:
            print("ERROR reading", sids)
            os.sys.exit(-1)

    # subjects = subjects[0:2]

    num_procs = min(len(subjects), 16)

    print("Processing", len(subjects), "with", num_procs, "processes")

    if not starting:
        init(multiprocessing.Lock())
    # print(opj(os.getenv('FREESURFER_HOME'),'subjects/fsaverage_sym'))
    # print(sd)

    if not os.path.isfile(os.path.join(sd, "fsaverage_sym/label/lh.cortex.label")):
        # cp -r $FREESURFER_HOME/subjects/fsaverage_sym ./
        shutil.copytree(opj(os.getenv('FREESURFER_HOME'),'subjects/fsaverage_sym'),opj(sd,'fsaverage_sym'), dirs_exist_ok=True)

    if not os.path.isfile(os.path.join(BASE_PATH, "fsaverage_sym/label/lh.cortex.label")):
        shutil.copytree(opj(os.getenv('FREESURFER_HOME'),'subjects/fsaverage_sym'),opj(BASE_PATH,'fsaverage_sym'), dirs_exist_ok=True)

    cortex_label = nb.freesurfer.io.read_label(os.path.join(sd, "fsaverage_sym/label/lh.cortex.label"))

    #First do all surfer stuff in parallell
    with multiprocessing.Pool(num_procs) as pool:
        pool.map(partial(fast_freesurfer_preprocessing_single_subject, subjects_dir=sd, cortex_label=cortex_label), subjects)

    # for sub in df.participant_id.values:
    # fast_freesurfer_preprocessing_single_subject(sub, sd)

    print("STEP 7: CREATE TRAINING DATA")
    #TODO read from bids dataset
    site_code = "H101"
    scanner = '3T'
    groups = {}
    #####CAREFUL BIG HACK TO MAKE IT WORK
    for i in range(1,181):
        if i <= 146:
            groups['sub-'+str(i).zfill(5)] = 'patient'
        else:
            groups['sub-'+str(i).zfill(5)] = 'control'

    #create training data for all subjects
    for subject in subjects:
        subject_id = subject['id']
        if site_code == "":
            try:
                site_code = subject_id.split("_")[1]  ##according to current MELD naming convention TODO
            except ValueError:
                print("Could not recover site code from", subject_id)
                os.sys.exit(-1)

        output_dir = os.path.join(BASE_PATH, f"MELD_{site_code}")
        print(BASE_PATH)
        os.makedirs(output_dir, exist_ok=True)

        create_training_data(subject_id, sd, output_dir, cortex_label, site_code=site_code, scanner=scanner, group=groups[subject_id])


    chunked_subject_list = list()
    chunk_size = min(len(subjects),20)
    subject_ids_list = [s['id'] for s in subjects]
    for i in range(0, len(subjects), chunk_size):
        chunked_subject_list.append(subject_ids_list[i : i + chunk_size])

    print("STEP 8: SMOOTH FEATURES")
    
    for chunk in chunked_subject_list:
        smooth_features(chunk, BASE_PATH)

    ###run combat site harmonization
    print('run site harmonization for', subject_ids_list)
    # print(demographic_file_path)
    # print(pd.read_csv(demographic_file_path))
    run_site_harmonization(subject_ids_list,site_code='H101',demographic_file_path=demographic_file_path)
    os.remove(demographic_file_path.name)

    ###continue

    print("STEP 9: PROCESS NEW SUBJECTS")
    process_new_subjects(subject_ids_list, "H101", BASE_PATH)


    print("STEP 10: PREDICT NEW SUBJECTS")
    for chunk in chunked_subject_list:
        predict_new_subjects(chunk, site_code='H101')


# %%
