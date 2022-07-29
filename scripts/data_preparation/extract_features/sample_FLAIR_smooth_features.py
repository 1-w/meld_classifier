# single subject version of sample_FLAIR_smooth_features.sh
# imports
import os
from os.path import join as opj
from subprocess import Popen, DEVNULL
# function

def call_command(command, verbose=False, wait=True):
    if verbose:
        proc = Popen(command, shell=True)
    else:
        proc = Popen(command, shell=True, stdout=DEVNULL)
    if wait:
        return proc.wait() #could be parallelized
    else:
        return proc



def sample_flair_smooth_features(subject_id, subject_dir, verbose=False):
    sub_dir = opj(subject_dir,subject_id)
    if os.path.isfile(opj(sub_dir,'surf_meld','rh.w-g.pct.mgh')):
        print('sample_flair_smooth_features already ran for', subject_id)
        return 0
    
    ##TODO python "$script_dir"/create_identity_reg.py "$s"

    
    os.makedirs(opj(sub_dir,'surf_meld'),exist_ok=True)

    for hem in ['lh','rh']:
        if os.path.isfile(opj(sub_dir,'mri','FLAIR.mgz')):
            for d in [0.5, 0.25, 0.75, 0]:
                command = 'mri_vol2surf --src "$s"/mri/FLAIR.mgz --out "$s"/surf_meld/"$h".gm_FLAIR_"$d".mgh --hemi "$h" --projfrac "$d" --srcreg "$s"/mri/transforms/Identity.dat --trgsubject "$s" --surf white'
                call_command(command,verbose=verbose)

            for d in [0.5, 1]:
                command = 'mri_vol2surf --src "$s"/mri/FLAIR.mgz --out "$s"/surf_meld/"$h".wm_FLAIR_"$d_wm".mgh --hemi "$h" --projdist -"$d_wm" --srcreg "$s"/mri/transforms/Identity.dat --trgsubject "$s" --surf white'
                call_command(command,verbose=verbose)

        command = 'mris_curvature_stats -f white -g --writeCurvatureFiles "$s" "$h" curv'
        call_command(command,verbose=verbose)

        command = 'mris_curvature_stats -f pial -g --writeCurvatureFiles "$s" "$h" curv'
        call_command(command,verbose=verbose)

        command ='mris_convert -c "$s"/surf/"$h".curv "$s"/surf/"$h".white "$s"/surf_meld/"$h".curv.mgh'
        call_command(command,verbose=verbose)

        command ='mris_convert -c "$s"/surf/"$h".sulc "$s"/surf/"$h".white "$s"/surf_meld/"$h".sulc.mgh'
        call_command(command,verbose=verbose)

        command ='mris_convert -c "$s"/surf/"$h".pial.K.crv "$s"/surf/"$h".white "$s"/surf_meld/"$h".pial.K.mgh'
        call_command(command,verbose=verbose)

        if verbose:
            print("Filtering and smoothing intrinsic curvature")

        ###TODO filter_intrinsic_curvature.py "$s"/surf_meld/"$h".pial.K.mgh "$s"/surf_meld/"$h".pial.K_filtered.mgh

        command ='mris_fwhm --s "$s" --hemi "$h" --cortex --smooth-only --fwhm 20\
--i "$s"/surf_meld/"$h".pial.K_filtered.mgh --o "$s"/surf_meld/"$h".pial.K_filtered.sm20.mgh'
        call_command(command,verbose=verbose)

        command ='mris_convert -c "$s"/surf/"$h".thickness "$s"/surf/"$h".white "$s"/surf_meld/"$h".thickness.mgh'
        call_command(command,verbose=verbose)

        command ='cp "$s"/surf/"$h".w-g.pct.mgh "$s"/surf_meld/"$h".w-g.pct.mgh'
        call_command(command,verbose=verbose)
    return 0