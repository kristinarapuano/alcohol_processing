#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
from os.path import join
from os import makedirs
import glob
from nltools.data import Brain_Data
from nltools.mask import expand_mask, collapse_mask
from scipy import stats
import argparse
import nibabel as nib
from nilearn.masking import compute_epi_mask, apply_mask, unmask
from nilearn.plotting import plot_stat_map
#from subprocess import call

base_dir = '/dartfs/rc/lab/D/DBIC/cosanlab/datax/Projects/alcohol/'
mask = Brain_Data('http://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz')
mask_x = expand_mask(mask)

sessions = ['ses-early', 'ses-late']

# Setup script argument that takes as input a job array index
parser = argparse.ArgumentParser(description='GLM script')
parser.add_argument('--array_id',required=True)
args = parser.parse_args()

# Use the index to grab the actual subject id we want to preprocess
subjects = glob.glob(join(base_dir,'sub-*'))
sub = os.path.split(subjects[int(args.array_id)])[-1]

sub_dir = join(join(base_dir, 'derivatives', 'roi_cleaned', sub))

if not os.path.exists(sub_dir):
    makedirs(sub_dir)

for ses in sessions:

    # smoothed (4mm FWHM)
    #fns = sorted(glob.glob(join(base_dir,'derivatives','fmriprep', sub, ses, 'func', '*_preproc.nii.gz')))
    #masks = sorted(glob.glob(join(base_dir,'derivatives','fmriprep', sub, ses, 'func', '*_brainmask.nii.gz')))

    #for fn,m in zip(fns,masks):
    #    call("3dBlurToFWHM -prefix {0}_4fwhm.nii.gz -input "
    #        " {1} -FWHM 4 -mask {2}".format(fn[:-7], fn, m), shell=True)
    
    # create subject-level wholebrain (wb) masks (inclusive of all voxels in run-wise masks) 
    #for m in masks:
    #   call("3dcalc -a m[0] -b m[1] -c m[2] -d m[3] -expr 'ispositive((a+b+c+d)-0)' "
    #   " -prefix '{0}_brainmask.nii.gz').format(sub), shell=True)

    smoothed_fns = sorted(glob.glob(join(base_dir,'derivatives','fmriprep', sub, ses, 'func', '*_4mm.nii.gz')))
    wb_mask = join(base_dir,'derivatives','fmriprep', sub, ses, 'func', '{0}_brainmask.nii.gz').format(sub)

    alldat = Brain_Data()
    for fn in smoothed_fns:
        dat = Brain_Data(fn)
        alldat = alldat.append(dat)
        
    maskeddat = alldat.apply_mask(nib.load(wb_mask))
    mn = np.mean(maskeddat.data, axis=0)
    sd = np.std(maskeddat.data, axis=0)
    tsnr = np.true_divide(mn, sd)
    # Compute mean across voxels within each TR
    global_mn = np.mean(maskeddat.data, axis=1)
    global_sd = np.std(maskeddat.data, axis=1)
    # Unmask data for plotting below
    mn = unmask(mn, wb_mask)
    sd = unmask(sd, wb_mask)
    tsnr = unmask(tsnr, wb_mask)

    # Identify global signal outliers
    global_outliers = np.append(np.where(global_mn > np.mean(global_mn) + np.std(global_mn) * 3),
                               np.where(global_mn < np.mean(global_mn) - np.std(global_mn) * 3))

    # Identify frame difference outliers
    frame_diff = np.mean(np.abs(np.diff(maskeddat.data, axis=0)), axis=1)
    frame_outliers = np.append(np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * 3),
                              np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * 3))

    fd_file_name = join(sub_dir, "fd_outliers.txt")
    global_file_name = join(sub_dir, "global_outliers.txt")
    np.savetxt(fd_file_name, frame_outliers)
    np.savetxt(global_file_name, global_outliers)
    
    confound_fns = sorted(glob.glob(join(base_dir,'derivatives','fmriprep', sub, ses, 'func', '*_confounds.tsv')))

    all_confounds = []
    for fn in confound_fns:
        confounds = pd.read_table(rp, skipinitialspace=True, usecols=
                    #['aCompCor0' + str(x) for x in range(0,6)] +
                    ['X', 'Y', 'Z','RotX', 'RotY', 'RotZ'])
        all_confounds.append(confounds)

    confound_regs = pd.concat(all_confounds)
    confound_regs = confound_regs-confound_regs.mean() #mean center
    confound_regs[['rpsq' + str(x) for x in range(1,7)]] = confound_regs.iloc[:,-6:]**2 #add squared
    confound_regs[['rpdiff' + str(x) for x in range(1,7)]] = pd.DataFrame(confound_regs.iloc[:,-6:][confound_regs.iloc[:,-6:].columns[0:6]].diff()) #derivative
    confound_regs[['rpdiffsq' + str(x) for x in range(1,7)]] = pd.DataFrame(confound_regs.iloc[:,-6:][confound_regs.iloc[:,-6:].columns[0:6]].diff())**2 #derivatives squared
    confound_regs['Intercept'] = 1 # Add Intercept
    confound_regs['LinearTrend'] = range(confound_regs.shape[0])-np.mean(range(confound_regs.shape[0])) # Add Linear Trend
    confound_regs['QuadraticTrend'] = confound_regs['LinearTrend']**2
    confound_regs = confound_regs.fillna(value=0) # fill "diff" NAs w/0 (first row)
    zeros = np.zeros(len(maskeddat))
    zeros[frame_outliers] = 1
    confound_regs['FrameOutliers'] = zeros
    zeros = np.zeros(len(maskeddat))
    zeros[global_outliers] = 1
    confound_regs['GlobalOutliers'] = zeros  
        
    maskeddat.X = confound_regs # Add to dat
    maskeddat_reg = maskeddat.regress()
    maskeddat_reg['residual'].write(join(sub_dir,'{0}_{1}_wholebrain_4mm_resid.nii.gz'.format(sub,ses)))

    for roi,m in enumerate(mask_x):
        maskeddat = alldat.apply_mask(m)
        maskeddat.X = confound_regs # Add to dat
        maskeddat_reg = maskeddat.regress()
        pd.DataFrame(maskeddat_reg['residual'].data).to_csv(join(sub_dir,
                                                   '{0}_{1}_ROI{2}_resid.csv'.format(sub,ses,roi)),index=None)
