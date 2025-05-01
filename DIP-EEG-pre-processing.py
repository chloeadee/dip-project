#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:25:41 2024

@author: chloedziego
"""

#EEG pre-processing script for DIP Project
#created by Chloe Dziego 11.09.2024

import mne
import numpy as np
import os.path as op
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt

from mne.preprocessing import ICA
from mne_icalabel import label_components

#make sure that plots are interactive
%matplotlib

#Define the ICA function
def compute_ica_correction(raw,subjnr,session, exercise_type):
    raw_copy = raw.copy()
    raw_copy.filter(l_freq = 1., h_freq = 0, n_jobs = 1, method='fir')
    
    #Here, I have changed eog=True and misc = True
    #picks that exclude eogs for ICA fitting and apply
    ica_picks = mne.pick_types(raw_copy.info, eeg=True, eog=False, exclude='bads')
    #picks that include eogs for eog_epochs
    eog_picks = mne.pick_types(raw_copy.info, eeg=True, eog=True, misc=True, exclude='bads')

    #set ICA parameters
    reject = dict(eeg=250e-6)
    method = 'fastica'
    decim = 3
    random_state = 23
    n_components = 0.99
    ica = ICA(n_components=n_components, method=method, random_state=random_state)

    #Fit ICA
    ica.fit(raw_copy, picks=ica_picks, decim=decim, reject=reject)

    #Find EOG artefacts
    eog_epochs = create_eog_epochs(raw_copy, picks= eog_picks, reject=reject)
    eog_average = eog_epochs.average()
    eog_inds, scores = ica.find_bads_eog(eog_epochs)
    
    #Find movement artefacts
    #acc_picks = mne.pick_types(raw_copy.info, misc=True, exclude='bads')
    #movement_epochs = create_eog_epochs(raw_copy, picks=acc_picks, reject=reject)  # Using same EOG detection for movement
    #movement_inds, _ = ica.find_bads_eog(movement_epochs)  # Reuse find_bads_eog to detect movement artifacts
    #eog_inds.extend(movement_inds)

    #Plot components identified as corresponding to EOG artefacts and overlays
    ica_plot_base = 'ica/' + subjnr + '_' + session + "_" + exercise_type + '_ica_plot'
    ica_overlay_name = 'ica/' + subjnr + '_' + session + "_" + exercise_type + '_ica_overlay.jpg'

    ica.plot_overlay(eog_average, exclude=eog_inds, show=False).savefig(ica_overlay_name)
    plt.close()
    
    for comp in range(len(eog_inds)):
        index = str(comp+1)
        ica_plot_name = ica_plot_base + index + '.jpg'
        ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax':35.}, image_args={'sigma':1.}, show=False)[comp].savefig(ica_plot_name)
    plt.close()
    
    #Apply ICA (to unfiltered data)
    ica.exclude.extend(eog_inds)
    ica.apply(raw)

    #Record components
    comps = ica.labels_
    ica_comp_filename = 'ica/' + subjnr + '_' + session + "_" + exercise_type + '_ica_comps.csv'
    w = csv.writer(open(ica_comp_filename, "w"))
    for key, val in comps.items():
        w.writerow([key, val])

    return raw

#if need to check mark_bads function:
#bad_chan_file = 'bad_chans_2.txt'
#subjnr = s_number

def mark_bads(bad_chan_file, raw, subjnr, exercise_type):
    if op.isfile(bad_chan_file):
        reader = csv.DictReader(open(bad_chan_file),dialect=csv.excel_tab)
        bad_channels = [r['Channel'] for r in reader if r['Subject'] == subjnr and r["Cond"] == exercise_type]
        
    if bad_channels == ['']: print("No channels to mark as bad")
    #if bad_channels == []: print("No channels to mark as bad version 2")
    else: 
        raw.info['bads'] = bad_channels[0].split(',')
        print("The following channels were marked as bad: "+str(raw.info['bads']))

    return raw

# Define pre-processing parameters
montage = mne.channels.make_standard_montage('standard_1020')
reject_ica = dict(eeg = 150e-6)
run_ica = True 

#           - - - - - - - - - - - -
#   PRE-PROCESSING FOR CORRECTLY MARKED FILES 
#           - - - - - - - - - - - -

# Get lists of on-task EEG recordings files
subjects = glob.glob('/Volumes/CHLOE/Data-Analysis-Work/DIP-project/experiment_data/P1-S1-M.vhdr')

# loop through each subject and session 
#this script uses bad_chans_1.txt and bad_chan_2.txt to identify and mark bad channels

for s in subjects:
    file_name_string = op.split(s)[1].split("-")
    s_number = file_name_string[0]
    exercise_type = file_name_string[2][0:1]
    session_no = file_name_string[1]
    
    #If full file is indentified; skip it!
    if exercise_type == "Ex":
        print("//// Identified subject " + s_number + " and session " + session_no + " exercise condition: " + exercise_type + " ... skipping pre-processing... /////")
        continue
        
    print("//// Processing subject " + s_number + " and session " + session_no + " exercise condition: " + exercise_type + " /////")
    raw = mne.io.read_raw_brainvision(s, preload = True, eog = ['LLeye', 'ROcanthus'], misc = ['ECG', 'RespRate', 'x_dir', 'y_dir', 'z_dir'])
    
    raw.set_eeg_reference(ref_channels = ['TP9','TP10'])
    raw.drop_channels(['TP9', 'TP10', 'ECG', 'RespRate'])
    
    raw.set_montage(montage)
    
    if session_no == 'S1':
        raw = mark_bads('bad_chans_1.txt',raw, s_number, exercise_type)
    else: raw = mark_bads('bad_chans_2.txt',raw, s_number, exercise_type)

    if run_ica:
        print("Computing ICA-based EOG correction")
        raw = compute_ica_correction(raw,s_number,session_no,exercise_type)
        
    #Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    
    raw.filter(0.1, 40., l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto', 
               filter_length = 'auto', method = 'fir', fir_window = 'hamming', phase = 'zero', n_jobs=2)
    
    processed_psd = raw.compute_psd(fmin = 0, fmax = 40).plot()
    plt.close('all')
    processed_psd.savefig("check_data/processed_psds/" + s_number + "_" + session_no + "_" + exercise_type + "_processed.jpg")

    outfile = 'success_processed_EEG_files/' + s_number + '_' + session_no + "_" + exercise_type + '_processed_raw.fif.gz'
    raw.save(outfile, fmt = 'single', overwrite = True)
    
#           - - - - - - - - - - - -
#   PRE-PROCESSING FOR FILES WITH EDITED MARKERS
#           - - - - - - - - - - - -

# Get lists of on-task EEG recordings files
edited_subjects = glob.glob('/Volumes/CHLOE/Data-Analysis-Work/DIP-project/files_cropped_from_opensesame_outputs/*.vhdr')

for s in edited_subjects:
    file_name_string = op.split(s)[1].split("-")
    s_number = file_name_string[0]
    exercise_type = file_name_string[2][0:1]
    session_no = file_name_string[1]
    
    if exercise_type == "E":
        print("//// Identified subject " + s_number + " and session " + session_no + " exercise condition: " + exercise_type + " ... skipping pre-processing... /////")
        continue
        
    print("//// Processing subject " + s_number + " and session " + session_no + " exercise condition: " + exercise_type + " /////")
    raw = mne.io.read_raw_brainvision(s, preload = True, eog = ['LLeye', 'ROcanthus'], misc = ['ECG', 'RespRate', 'x_dir', 'y_dir', 'z_dir'])
    
    raw.set_eeg_reference(ref_channels = ['TP9','TP10'])
    raw.drop_channels(['TP9', 'TP10', 'ECG', 'RespRate', 'x_dir', 'y_dir', 'z_dir'])
    
    raw.set_montage(montage)
    
    if session_no == 'S1':
        raw = mark_bads('bad_chans_1.txt',raw, s_number, exercise_type)
    else: raw = mark_bads('bad_chans_2.txt',raw, s_number, exercise_type)
    
    #find events (in this case, just start of cognitive training)
    events = mne.events_from_annotations(raw)[0]
    event_id = mne.events_from_annotations(raw)[1]

    if run_ica:
        print("Computing ICA-based EOG correction")
        raw = compute_ica_correction(raw,s_number,session_no,exercise_type)

    #Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    
    raw.filter(0.1, 40., l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto', 
               filter_length = 'auto', method = 'fir', fir_window = 'hamming', phase = 'zero', n_jobs=2)
    
    processed_psd = raw.compute_psd(fmin = 0, fmax = 40).plot()
    plt.close('all')
    processed_psd.savefig("check_data/processed_psds/" + s_number + "_" + session_no + "_" + exercise_type + "_processed.jpg")

    outfile = 'success_processed_EEG_files/edited_OS/' + s_number + '_' + session_no + "_" + exercise_type + '_processed_raw.fif.gz'
    raw.save(outfile, fmt = 'single', overwrite = True)
    
#checking how many files should be there in total
#file_number = [x for x in subjects if 'Ex.vhdr' not in x]     

    