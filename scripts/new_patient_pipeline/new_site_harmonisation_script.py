## This script harmonise data from a new site or new scanner that were not part of the MELD training 
## Needs to have at leat 30 subjects, with MELD ids listed in a txt file
## need to provide the demographic csv file and the new site code


## To run : python new_site_harmonisation_script.py -ids <text_file_with_subject_ids>  -site <site_code> -demographics <path_to_demographics_file>


# Import packages
import os
import argparse
import pandas as pd
import numpy as np
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, NEWSUBJECTS_DATASET
 
def create_dataset_file(subjects, output_path):
    df=pd.DataFrame()
    subjects_id = [subject for subject in subjects]
    df['subject_id']= subjects_id
    df['split']=['test' for _ in subjects]
    df.to_csv(output_path)


def run_site_harmonization(subject_ids, site_code, demographic_file_path, output_dir=BASE_PATH, dataset_newSubject=''):
    if dataset_newSubject == '':
        dataset_newSubject = os.path.join(BASE_PATH, NEWSUBJECTS_DATASET)

    # Set features and smoothed values
    features = {
		".on_lh.thickness.mgh": 10,
		".on_lh.w-g.pct.mgh" : 10,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 5,
		'.on_lh.curv.mgh' : 5,
		'.on_lh.gm_FLAIR_0.25.mgh' : 10,
		'.on_lh.gm_FLAIR_0.5.mgh' : 10,
		'.on_lh.gm_FLAIR_0.75.mgh' : 10,
		".on_lh.gm_FLAIR_0.mgh": 10,
		'.on_lh.wm_FLAIR_0.5.mgh' : 10,
		'.on_lh.wm_FLAIR_1.mgh' : 10,
    			}
    feat = Feature()
    features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
    
    ### INITIALISE ###
    #create dataset
    create_dataset_file(subject_ids, dataset_newSubject)
   
    ### COMBAT DISTRIBUTED DATA ###
    #-----------------------------------------------------------------------------------------------
    print('PROCESS : COMPUTE COMBAT PARAMETERS NEW SITE')
        
    #create cohort for the new subject
    c_smooth= MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', 
                       dataset=dataset_newSubject, data_dir=output_dir)
    #create object combat
    combat =Preprocess(c_smooth,
                           site_codes=[site_code],
                           write_hdf5_file_root="MELD_{site_code}/{site_code}_combat_parameters.hdf5",
                           data_dir=output_dir)
    #features names
    for feature in features_smooth:
        print(feature)
        combat.get_combat_new_site_parameters(feature, demographic_file_path)
    

       

    
    
if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='perform cortical parcellation using recon-all from freesurfer')
    parser.add_argument('-ids','--list_ids',
                        help='Subjects IDs in a text file',
                        default='',
                        required=False)
    parser.add_argument('-site','--site_code',
                        help='Site code',
                        required=True,)
    parser.add_argument('-demos','--demographic_file',
                        help='path to the new site demographic file',
                        required=True,)
    args = parser.parse_args()
    demographic_file_path = args.demographic_file

    if args.list_ids != '':
        subject_ids = np.array(np.loadtxt(args.list_ids, dtype='str',ndmin=1))
    else:
        subject_ids = pd.read_csv(demographic_file_path)['ID'].values
    site_code=str(args.site_code)

    run_site_harmonization(subject_ids, site_code, demographic_file_path)

    

    
