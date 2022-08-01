## This script runs a FreeSurfer reconstruction on a participant
## Within your  MELD folder should be an input folder that contains folders
## for each participant. Within each participant folder should be a T1 folder
## that contains the T1 in nifti format ".nii" and where available a FLAIR
## folder that contains the FLAIR in nifti format ".nii"

## To run : python new_pt_pipeline_script1.py -id <sub_id> -site <site_code>

## export FASTSURFER_HOME=/home/lennartw/wdir/other_projects/FastSurfer/
#%%

import os
import sys
import argparse
import subprocess as sub
import threading
import multiprocessing
from functools import partial
import glob
import tempfile
from meld_classifier.paths import BASE_PATH, SCRIPTS_DIR, MELD_DATA_PATH, FS_SUBJECTS_PATH
import pandas as pd

#%%


def init(lock):
    global starting
    starting = lock


def fastsurfer_subject(subject, fs_folder):
    # # NECESSARY ??
    # if site_code == '':
    #     try:
    #         site_code = subject.split('_')[1] ##according to current MELD naming convention TODO
    #     except ValueError:
    #         print('Could not recover site code from', subject)
    #         sys.exit(-1)

    if type(subject) == dict:
        subject_id = subject['id']
        subject_t1_path = subject['t1_path']
    else:
        subject_id = subject
        subject_t1_path =''

    print("called fastsurfer_subject", subject_id)

    # get subject folder

    # If freesurfer outputs already exist for this subject, continue running from where it stopped
    # Else, find inputs T1 and FLAIR and run FS
    if os.path.isdir(os.path.join(fs_folder, subject_id)):
        print(f"STEP 1:Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        #         recon_all = format("$FREESURFER_HOME/bin/recon-all -sd {} -s {} -FLAIRpial -all -no-isrunning"
        #                    .format(fs_folder, subject,))
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = os.path.join(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(os.path.join(subject_dir, "T1", "*T1w.nii*"))

        # check T1 and FLAIR exist
        if len(subject_t1_path) > 1:
            raise FileNotFoundError(
                "Find too much volumes for T1. Check and remove the additional volumes with same key name"
            )
        elif not subject_t1_path:
            raise FileNotFoundError(f"Could not find T1 volume. Check if name follow the right nomenclature")
        else:
            subject_t1_path = subject_t1_path[0]

    # setup cortical segmentation command
    print("STEP 1: Segmentation using T1 only with FastSurfer")
    command = format(
        "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {} --parallel".format(fs_folder, subject_id, subject_t1_path)
    )

    # call fastsurfer
    print(f"INFO : Start cortical parcellation for {subject_id} (up to 2h). Please wait")
    print(f"INFO : Results will be stored in {fs_folder}")
    starting.acquire()  # no other process can get it until it is released
    proc = sub.Popen(command, shell=True, stdout=sub.DEVNULL)
    threading.Timer(120, starting.release).start()  # release in two minutes
    proc.wait()
    print(f"INFO : Finished cortical parcellation for {subject_id} !")


def fastsurfer_flair(subject, fs_folder):
    if type(subject) == dict:
        subject_id = subject['id']
        subject_flair_path = subject['flair_path']
    else:
        subject_id = subject
        subject_flair_path =''


    # if os.path.isfile(os.path.join(fs_folder, subject, 'stats','lh.aparc.pial.stats')):
    if os.path.isfile(os.path.join(fs_folder, subject_id, "mri", "FLAIR.mgz")):
        print(f"STEP 1.2: Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        #         recon_all = format("$FREESURFER_HOME/bin/recon-all -sd {} -s {} -FLAIRpial -all -no-isrunning"
        #                    .format(fs_folder, subject,))
        return

    if subject_flair_path == '':
        # get subject folder
        #assume meld data structure
        subject_dir = os.path.join(MELD_DATA_PATH, "input", subject_id)
        subject_flair_path = glob.glob(os.path.join(subject_dir, "FLAIR", "*FLAIR.nii*"))

        if len(subject_flair_path) > 1:
            raise FileNotFoundError(
                "Find too much volumes for FLAIR. Check and remove the additional volumes with same key name"
            )

        if not subject_flair_path:
            print("No FLAIR file has been found for subject", subject_id)
            return 

        subject_flair_path = subject_flair_path[0]


    print("Starting FLAIRpial for subject", subject_id)
    command = format(
        "recon-all -sd {} -subject {} -FLAIR {} -FLAIRpial -autorecon3".format(fs_folder, subject_id, subject_flair_path)
    )
    proc = sub.Popen(command, shell=True, stdout=sub.DEVNULL)
    proc.wait()
    print("finished FLAIRpial for subject", subject_id)


def extract_features(subject, scripts_dir, fs_folder, site_code=""):
    # Launch script to extract surface-based features from freesurfer outputs
    print("STEP 2: Extract surface-based features", subject)
    if site_code == "":
        try:
            site_code = subject.split("_")[1]  ##according to current MELD naming convention TODO
        except ValueError:
            print("Could not recover site code from", subject)
            sys.exit(-1)
    #### EXTRACT SURFACE-BASED FEATURES #####
    # Create the output directory to store the surface-based features processed
    output_dir = os.path.join(BASE_PATH, f"MELD_{site_code}")
    os.makedirs(output_dir, exist_ok=True)
    # Create temporary list of ids
    tmp = tempfile.NamedTemporaryFile(mode="w")
    tmp.write(subject)

    command = format(
        f"bash {scripts_dir}/data_preparation/meld_pipeline.sh {fs_folder} {site_code} {tmp.name} {scripts_dir}/data_preparation/extract_features {output_dir}"
    )
    sub.check_call(command, shell=True)  # ,stdout=sub.DEVNULL)
    tmp.close()
    # os.remove(tmp.name)


def smooth_features(subjects, scripts_dir):
    # Create temporary list of ids
    subject_ids = tempfile.NamedTemporaryFile(mode="w")
    subject_ids.write(subject)

    # Launch script to smooth features
    print("STEP 3: SMOOTH FEATURES")
    command = format(
        f"python {scripts_dir}/data_preparation/run_data_smoothing_new_subjects.py -ids {subject_ids.name} -d {BASE_PATH}"
    )
    sub.check_call(command, shell=True, stdout=sub.DEVNULL)
    subject_ids.close()
    # os.remove(subject_ids.name)


from scripts.data_preparation.extract_features.create_xhemi import run_parallel


def run_subjects_fastsurfer_and_smoothing_parallel(subject_list, num_procs=20, site_code=""):
    # parallel version of the pipeline, finish each stage for all subjects first

    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    sub.check_call(ini_freesurfer, shell=True, stdout=sub.DEVNULL)

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)
    arguments = []
    ## first processing stage with fastsurfer: segmentation
    pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
    for _ in pool.imap_unordered(partial(fastsurfer_subject, fs_folder=fs_folder, site_code=site_code), subject_list):
        pass

    ## flair pial correction
    pool = multiprocessing.Pool(processes=num_procs)
    for _ in pool.imap_unordered(partial(fastsurfer_flair, fs_folder=fs_folder), subject_list):
        pass

    ## EXTRACT SURFACE-BASED FEATURES
    ## CANT PARALLELIZE!!!

    scripts_dir = os.path.join(SCRIPTS_DIR, "scripts")
    # pool = multiprocessing.Pool(processes=num_procs,
    #         initializer=init, initargs=[multiprocessing.Lock()])
    # for _ in pool.imap_unordered(partial(extract_features,scripts_dir=scripts_dir,fs_folder=fs_folder,site_code = site_code), subject_list):
    #     pass

    # parallelize create xhemi because it takes a while!
    run_parallel(subject_list, fs_folder, num_procs=num_procs)

    for subject in subject_list:
        extract_features(subject, scripts_dir=scripts_dir, fs_folder=fs_folder, site_code=site_code)

    #### SMOOTH FEATURES #####
    # Launch script to smooth features
    print("STEP 3: SMOOTH FEATURES")
    # pool = multiprocessing.Pool(processes=num_procs,
    #         initializer=init, initargs=[multiprocessing.Lock()])
    # for _ in pool.imap_unordered(partial(smooth_features,scripts_dir=scripts_dir), subject_list):
    #     pass
    for subject in subject_list:
        smooth_features(subject, scripts_dir=scripts_dir, fs_folder=fs_folder, site_code=site_code)


def new_patient_freesurfer_and_smoothing(subject, site_code="", use_fastsurfer=False, parallel=True):
    if site_code == "":
        try:
            site_code = subject.split("_")[1]  ##according to current MELD naming convention TODO
        except ValueError:
            print("Could not recover site code from", subject)
            sys.exit(-1)

    scripts_dir = os.path.join(SCRIPTS_DIR, "scripts")

    # get subject folder
    subject_dir = os.path.join(MELD_DATA_PATH, "input", subject)

    #### FREESURFER RECON-ALL #####

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    # initialise freesurfer variable environment
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")

    # If freesurfer outputs already exist for this subject, continue running from where it stopped
    # Else, find inputs T1 and FLAIR and run FS
    if os.path.isdir(os.path.join(fs_folder, subject)):
        print(f"STEP 1:Freesurfer outputs already exists for subject {subject}. \nFreesurfer will be skipped")
        #         recon_all = format("$FREESURFER_HOME/bin/recon-all -sd {} -s {} -FLAIRpial -all -no-isrunning"
        #                    .format(fs_folder, subject,))
        pass
    else:
        # select inputs files T1 and FLAIR
        T1_file = glob.glob(os.path.join(subject_dir, "T1", "*.nii*"))
        FLAIR_file = glob.glob(os.path.join(subject_dir, "FLAIR", "*.nii*"))
        # check T1 and FLAIR exist
        if len(T1_file) > 1:
            raise FileNotFoundError(
                "Find too much volumes for T1. Check and remove the additional volumes with same key name"
            )
        elif not T1_file:
            raise FileNotFoundError(f"Could not find T1 volume. Check if name follow the right nomenclature")
        else:
            T1_file = T1_file[0]
        if len(FLAIR_file) > 1:
            raise FileNotFoundError(
                "Find too much volumes for FLAIR. Check and remove the additional volumes with same key name"
            )
        elif not FLAIR_file:
            print("No FLAIR file has been found")
            isflair = False
        else:
            FLAIR_file = FLAIR_file[0]
            isflair = True
        if use_fastsurfer:
            command = format(
                "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {}".format(fs_folder, subject, T1_file)
            )
            if isflair == True:
                command += format(
                    "recon-all -sd {} -subject {} -FLAIR {} -FLAIRpial -autorecon3".format(
                        fs_folder, subject, FLAIR_file
                    )
                )
        else:
            # setup cortical segmentation command
            if isflair == True:
                print("STEP 1:Segmentation using T1 and FLAIR")
                recon_all = format(
                    "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -FLAIR {} -FLAIRpial -all".format(
                        fs_folder, subject, T1_file, FLAIR_file
                    )
                )
            else:
                print("STEP 1:Segmentation using T1 only")
                recon_all = format(
                    "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -all".format(fs_folder, subject, T1_file)
                )

        # call freesurfer
        command = ini_freesurfer + ";" + recon_all
        print(f"INFO : Start cortical parcellation for {subject} (up to 36h). Please wait")
        print(f"INFO : Results will be stored in {fs_folder}")
        sub.check_call(command, shell=True, stdout=sub.DEVNULL)

    #### EXTRACT SURFACE-BASED FEATURES #####
    # Create the output directory to store the surface-based features processed
    output_dir = os.path.join(BASE_PATH, f"MELD_{site_code}")
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary list of ids
    subject_ids = os.path.join(BASE_PATH, "subject_for_freesurfer.txt")
    with open(subject_ids, "w") as f:
        f.write(subject)

    # Launch script to extract surface-based features from freesurfer outputs
    print("STEP 2: Extract surface-based features")
    command = format(
        f"bash {scripts_dir}/data_preparation/meld_pipeline.sh {fs_folder} {site_code} {subject_ids} {scripts_dir}/data_preparation/extract_features {output_dir}"
    )
    sub.check_call(command, shell=True, stdout=sub.DEVNULL)

    #### SMOOTH FEATURES #####
    # Launch script to smooth features
    print("STEP 3: SMOOTH FEATURES")
    command = format(
        f"python {scripts_dir}/data_preparation/run_data_smoothing_new_subjects.py -ids {subject_ids} -d {BASE_PATH}"
    )
    sub.check_call(command, shell=True, stdout=sub.DEVNULL)

    # delete temporary list ids
    os.remove(subject_ids)


if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
    parser.add_argument(
        "-id",
        "--id_subj",
        help="Subject ID.",
        default="",
        required=False,
    )
    parser.add_argument(
        "-sl",
        "--subject_list",
        default="",
        help="Relative path to subject List containing id and site_code.",
        required=False,
    )
    parser.add_argument(
        "-bd",
        "--bids_dir",
        default="",
        help="Relative path to bids directory",
        required=False,
    )
    parser.add_argument(
        "-site",
        "--site_code",
        help="Site code",
        default="",
        required=False,
    )
    parser.add_argument(
        "-fs", "--fastsurfer", help="use fastsurfer instead of freesurfer", required=False, default=False
    )
    parser.add_argument("-p", "--parallel", help="use fastsurfer instead of freesurfer", required=False, default=True)
    args = parser.parse_args()
    subject = str(args.id_subj)
    site_code = str(args.site_code)
    subject_list = str(args.subject_list)
    bids_dir = str(args.bids_dir)
    use_fastsurfer = args.fastsurfer
    use_parallel = args.parallel
    print(args)
    if bids_dir != "":
        print("Found BIDS dir input, ignoring all other subject specific input..")
        try:
            pass
        except:
            print("Could not load bids directory, is it valid BIDS?")
            sys.exit(-1)

    if subject_list != "":
        if site_code != "" or subject != "":
            print("Ignoring...")
        try:
            sub_list_df = pd.read_csv(subject_list)
        except ValueError:
            print("Could not open, subject_list")
            sys.exit(-1)
        if use_fastsurfer:
            run_subjects_fastsurfer_and_smoothing_parallel(list(sub_list_df.participant_id.values))
    else:
        if subject == "":
            print("Please specify both subject and site_code...")
        else:
            new_patient_freesurfer_and_smoothing(subject, site_code, use_fastsurfer)


# %%
