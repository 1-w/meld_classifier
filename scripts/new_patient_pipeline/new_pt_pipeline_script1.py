## This script runs a FreeSurfer reconstruction on a participant
## Within your  MELD folder should be an input folder that contains folders 
## for each participant. Within each participant folder should be a T1 folder 
## that contains the T1 in nifti format ".nii" and where available a FLAIR 
## folder that contains the FLAIR in nifti format ".nii"

## To run : python new_pt_pipeline_script1.py -id <sub_id>


import os
import sys
import argparse
import subprocess as sub
import glob
from meld_classifier.paths import MELD_DATA_PATH, FS_SUBJECTS_PATH
        
if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='perform cortical parcellation using recon-all from freesurfer')
    parser.add_argument('-id','--id_subj',
                        help='Subject ID.',
                        required=True,)
    args = parser.parse_args()
    subject=str(args.id_subj)
    
    # get subject folder and fs folder 
    subject_dir = os.path.join(MELD_DATA_PATH,'input',subject)
    
    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)
    
    #initialise freesurfer variable environment
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
        
    # If freesurfer outputs already exist for this subject, continue running from where it stopped
    # Else, find inputs T1 and FLAIR and run FS
    if os.path.isdir(os.path.join(fs_folder, subject)):
        print(f'Freesurfer outputs already exists for subject {subject}. \nFS will be continued from where it stopped')
        recon_all = format("$FREESURFER_HOME/bin/recon-all -sd {} -s {} -FLAIRpial -all -no-isrunning"
                   .format(fs_folder, subject,))
    else : 
        #select inputs files T1 and FLAIR
        T1_file = glob.glob(os.path.join(subject_dir, 'T1', '*.nii*'))
        FLAIR_file = glob.glob(os.path.join(subject_dir, 'FLAIR', '*.nii*'))
        #check T1 and FLAIR exist
        if len(T1_file)>1 :
            raise FileNotFoundError('Find too much volumes for T1. Check and remove the additional volumes with same key name')   
        elif not T1_file:
            raise FileNotFoundError(f'Could not find T1 volume. Check if name follow the right nomenclature')
        else:
            T1_file = T1_file[0]
        if len(FLAIR_file)>1:
            raise FileNotFoundError('Find too much volumes for FLAIR. Check and remove the additional volumes with same key name')
        elif not FLAIR_file:
            print('No FLAIR file has been found')
            isflair = False
        else:
            FLAIR_file=FLAIR_file[0]
            isflair = True     
        #setup cortical segmentation command
        if isflair == True:
            print('Segmentation using T1 and FLAIR')
            recon_all = format("$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -FLAIR {} -FLAIRpial -all"
                       .format(fs_folder, subject, T1_file, FLAIR_file))
        else:
            print('Segmentation using T1 only')
            recon_all = format("$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -all"
                           .format(fs_folder, subject, T1_file))
    
    #call freesurfer 
    command = ini_freesurfer + ';' + recon_all
    print(f"INFO : Start cortical parcellation for {subject} (up to 36h). Please wait")
    print(f"INFO : Results will be stored in {fs_folder}")
    sub.check_call(command, shell=True)
    

    
