#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:23:25 2024

@author: chloedziego
"""

#ERP epoch and window average script for DIP Project
#Created by Chloe Dziego 11.09.2024

import mne
import numpy as np
import os.path as op
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
from autoreject import AutoReject, get_rejection_threshold
from philistine.mne import retrieve

#Defining the retrieve function if needed
#def retrieve(epochs, windows, subj=None, items=None):
#    df = epochs.to_data_frame(picks=None, index=['epoch','time'])
#    eeg_chs = [c for c in df.columns if c not in ('condition')]
#    factors = ['epoch'] # the order is important here! otherwise the shortcut with items later won't work
#    sel = factors + eeg_chs
#    df = df.reset_index()

#    retrieve = []
#    for w in windows:
#        temp = df[ df.time >= windows[w][0] ]
#        dfw = temp[ temp.time <= windows[w][1] ]
#        dfw_mean = dfw[sel].groupby(factors).mean()
#        dfw_mean["win"] = "{}..{}".format(*windows[w])
#        dfw_mean["wname"] = w
#        retrieve.append(dfw_mean)
#
#    retrieve = pd.concat(retrieve)
#    return retrieve

#Make sure that plots are interactive
%matplotlib

#Create epochs around cue and target related triggers
#Epoching and pre-processing parameters
tmin, tmax = -0.2, 1.2
baseline = None
flat = dict(eeg=5e-6)
reject = {'eeg': 150e-6,
          'eog': 250e-6}

#Create list for export
epochs_for_export = []

#Creating the combined events for epoching around
combined_events_id = {}

combined_events_id['cue-left'] = 99 #Comment/Cue/C  1
combined_events_id['cue-right'] = 98 #Comment/Cue/C  2

combined_events_id['target-attend-nogo-correct'] = 51 #Comment/Target/T  1 + Comment/Outcome/O  1
combined_events_id['target-attend-nogo-wrong'] = 52 #Comment/Target/T  1 + Comment/Target/T  1 + (Response/R  1) + Comment/Outcome/O  2
combined_events_id['target-attend-go-wrong'] = 53 #Comment/Target/T  1 + Comment/Target/T  3 + Comment/Outcome/O  2
combined_events_id['target-attend-go-correct'] = 54 #Comment/Target/T  1 + Comment/Target/T  3 + (Response/R  1) + Comment/Outcome/O  1

combined_events_id['target-ignore-nogo-correct'] = 61 #Comment/Target/T  2 + Comment/Outcome/O  1
combined_events_id['target-ignore-nogo-wrong'] = 62 #Comment/Target/T  2 + (Response/R  1) + Comment/Outcome/O  2
combined_events_id['target-ignore-go-wrong'] = 63 #Comment/Target/T  2 + Comment/Target/T  3 + (Response/R  1) + Comment/Outcome/O  2
combined_events_id['target-ignore-go-correct'] = 64 #Comment/Target/T  2 + Comment/Target/T  + Comment/Outcome/O  1

#Set up headers for csv to count the number of epochs
with open('solution_epoch_count.csv', 'w', newline = '') as output:
    writer = csv.writer(output)
    writer.writerow(['subject', 'cue-left', 'cue-right', 'target-attend-nogo-correct', 'target-attend-nogo-wrong', 'target-attend-go-wrong', 'target-attend-go-correct', 'target-ignore-nogo-correct', 'target-ignore-nogo-wrong', 'target-ignore-go-wrong', 'target-ignore-go-correct'])

# *** added an extra csv to check the new ones!
with open('edited_solution_epoch_count.csv', 'w', newline = '') as output:
    writer = csv.writer(output)
    writer.writerow(['subject', 'cue-left', 'cue-right', 'target-attend-nogo-correct', 'target-attend-nogo-wrong', 'target-attend-go-wrong', 'target-attend-go-correct', 'target-ignore-nogo-correct', 'target-ignore-nogo-wrong', 'target-ignore-go-wrong', 'target-ignore-go-correct'])
    
#Get the list of processed files list
processed_files = glob.glob('success_processed_EEG_files/*_processed_raw.fif.gz')

#This code is to process all of the edited OS files
processed_files = glob.glob('success_processed_EEG_files/edited_OS/*.fif.gz')

#Creating Epoch files for Grand Averages/Averaging across Windows
for p in processed_files:
    file_name_string = op.split(p)[1].split("_")
    s_number = file_name_string[0]
    exercise_type = file_name_string[2][0:1]
    session_no = file_name_string[1]
    
    #Read in the epochs file
    epochs_file = op.join('epochs_processed/' + s_number + '_' + session_no + "_" + exercise_type +'_epochs.fif.gz')
    
    print('   ')
    print('   ')
    print("/////    Reading preprocessed data for participant " + s_number + ' for session ' + session_no + " during exercise condition: " + exercise_type + ' /////')
    print('   ')
    print('   ')
    raw = mne.io.read_raw_fif(p, preload=True)
    
    #Extract events
    events, event_id = mne.events_from_annotations(raw)
    events_vis = mne.viz.plot_events(events, sfreq=None, first_samp=0, color=None, event_id=None, axes=None, equal_spacing=True, show=True, on_missing='raise', verbose=None)
    events_vis.savefig('check_data/visualised_events/' + s_number + '-' + session_no + '_visualised_events_plot.png')
    plt.close('all')
    
    #Edit individual triggers in events to match with combined_event_id
    #initialise new events array
    events_edited = events.copy()
    
    #set everything empty to not introduce old information into new iterations
    cue_triggers = []
    target_triggers = []
    responded_trigger =[]
    correct = []
    wrong = []
	
	#Extract all of the relevant trigger information, avoiding any issues due to the misalignment of triggers
    if 'Comment/98' in event_id: 
        cue_triggers.append(event_id['Comment/98'])
        if 'Comment/99' in event_id: cue_triggers.append(event_id['Comment/99'])
    else:
        if'Comment/Cue/C  1' in event_id: cue_triggers.append(event_id['Comment/Cue/C  1'])
        if 'Comment/Cue/C  2' in event_id: cue_triggers.append(event_id['Comment/Cue/C  2'])

    if 'Comment/Target/T  2' in event_id: target_triggers.append(event_id['Comment/Target/T  2'])
    if 'Comment/Target/T  1' in event_id: target_triggers.append(event_id['Comment/Target/T  1'])
    if 'Comment/Target/T  3' in event_id: go_trigger = event_id['Comment/Target/T  3']
    
    if 'Response/R  1' in event_id: responded_trigger = event_id['Response/R  1']
    
    if 'Comment/Outcome/O  1' in event_id: correct = event_id['Comment/Outcome/O  1']
    if 'Comment/Outcome/O  2' in event_id: wrong = event_id['Comment/Outcome/O  2']

	#Check that these are correct
    print('Using triggers as assigned...')
    print("Cues:") 
    print(cue_triggers) 
    print("Targets:") 
    print(target_triggers)
    print("Responses:")
    print(responded_trigger)
    print("Correct:")
    print(correct)
    print('Incorrect (empty if no incorrect trials):')
    print(wrong)
    
    print("Participant has the following number of triggers:")
    print(len(events_edited))
    length = len(events_edited)
    last_triggers = [length, length-1, length-2]
    
    #Go through the list of extracted triggers and create new ones based off of what is needed for analysis
    for x in range(0,len(events_edited)):
        # get trigger info from column 3 of events
        trigger = events_edited[x,2]
        
        if trigger in cue_triggers:
            if 'Comment/Cue/C  1' in event_id and trigger == event_id['Comment/Cue/C  1']:
                events_edited[x,2] = 99
            if 'Comment/Cue/C  2' in event_id and trigger == event_id['Comment/Cue/C  2']:
                events_edited[x,2] = 98
            if 'Comment/99' in event_id and trigger == event_id['Comment/99']:
                events_edited[x,2] = 99
            if 'Comment/98' in event_id and trigger == event_id['Comment/98']:
                events_edited[x,2] = 98

        if trigger in target_triggers and x not in last_triggers:
            
            if'Comment/Target/T  1' in event_id and trigger == event_id['Comment/Target/T  1']:
                second_trigger = events_edited[x+1,2]
                third_trigger = events_edited[x+2,2]
                
                if second_trigger == go_trigger and third_trigger == responded_trigger:
                    events_edited[x,2] = 54
                if second_trigger == go_trigger and third_trigger == wrong:
                    events_edited[x,2] = 53
                if second_trigger == responded_trigger:
                    events_edited[x,2] = 52
                if second_trigger == correct:
                    events_edited[x,2] = 51
            
            if 'Comment/Target/T  2' in event_id and trigger == event_id['Comment/Target/T  2']:
                second_trigger = events_edited[x+1,2]
                third_trigger = events_edited[x+2,2]
                if second_trigger == go_trigger and third_trigger == correct:
                    events_edited[x,2] = 64
                if second_trigger == go_trigger and third_trigger == responded_trigger:
                    events_edited[x,2] = 63
                if second_trigger == responded_trigger:
                    events_edited[x,2] = 62
                if second_trigger == correct:
                    events_edited[x,2] = 61  
    
    #Create the new events dictionary
    events = events_edited
    event_id = combined_events_id
    
    #create epochs based on new conbined_events_id
    epochs = mne.Epochs(raw, events_edited, combined_events_id, tmin=tmin, tmax=tmax, 
                        proj=False,baseline=baseline, detrend=0, reject_by_annotation=False,preload=True,
                        on_missing = 'warn', event_repeated = 'merge')
    
    print(epochs)
    
    #apply the autoreject function to the epoched data
    ar = AutoReject(thresh_method='bayesian_optimization', random_state=42)
    ar.fit(epochs) 
    epochs_clean,reject_log = ar.transform(epochs,return_log=True)
    evoked_clean = epochs_clean.average()
    evoked = epochs.average()
    plt.close('all')
    
    #visualise the data before and after autoreject application
    print("Plotting effects of AutoReject algorithm")
    ar_figure_name = 'check_data/autoreject/' + s_number + '_' + session_no + '_' + exercise_type + '_autoreject.png'
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom='off', top='off')
        ax.tick_params(axis='y', which='both', left='off', right='off')
    ylim = dict(eeg=(-100, 100))
    evoked.pick_types(eeg=True, exclude=[])
    evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
    axes[1].set_title('Before autoreject')
    evoked_clean.pick_types(eeg=True, exclude=[])
    evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
    axes[1].set_title('After autoreject')
    plt.tight_layout()
    plt.savefig(ar_figure_name)
    plt.close('all')
    
    scalings = dict(eeg=100e-6)
    #rejections = reject_log.plot_epochs(epochs, scalings=scalings)
    rejections_drop = epochs_clean.plot_drop_log(show=False)
    rejections_drop.savefig('check_data/rejections/' + s_number + '-' + session_no + '_' + exercise_type + '_rejections_drop.png')
    plt.close('all')
    
    #generate a power spectral density plot from the EPOCH data
    print("Plotting subject's new psd plot following referencing, filtering and epoching")
    psd_cleaned = epochs_clean.plot_psd(fmax = 30)
    psd_cleaned.savefig('check_data/epoch_plots/' + s_number + '_' + session_no + '_' + exercise_type + '_cleaned_psd.png') 
    plt.close('all')
    
    #Save the epochs file for plotting/statistical analyses scripts later
    print("Saving epochs...")
    epochs.save(epochs_file, overwrite = True)
    
    #now count the number of trials per participant following removal and save to csv
    cueL = len(epochs_clean['cue-left'])
    cueR = len(epochs_clean['cue-right'])
    
    try: ANGC = len(epochs_clean['target-attend-nogo-correct'])
    except: ANGC = 0
    
    try: ANGW = len(epochs_clean['target-attend-nogo-wrong'])
    except: ANGW = 0
    
    try: AGW = len(epochs_clean['target-attend-go-wrong'])
    except: AGW = 0
    
    try: AGC = len(epochs_clean['target-attend-go-correct'])
    except: AGC = 0
    
    try: INGC = len(epochs_clean['target-ignore-nogo-correct'])
    except: INGC = 0
    
    try: INGW = len(epochs_clean['target-ignore-nogo-wrong'])
    except: INGW = 0
    
    try: IGW = len(epochs_clean['target-ignore-go-wrong'])
    except: IGW = 0
    
    try: IGC = len(epochs_clean['target-ignore-go-correct'])
    except: ICG = 0
        
    ##################
    
    with open('edited_solution_epoch_count.csv', 'a', newline = '') as outcsv:
        # create the csv writer
        writer = csv.writer(outcsv)
        # write a row to the csv file
        writer.writerow([s_number, cueL, cueR, ANGC, ANGW, AGW, AGC, INGC, INGW, IGW, IGC])
    
    #Create dataframe for plotting (with resample/writing dataframe)
    # Downsample to 100 Hz for plotting purposes
    print('Original sampling rate:', epochs.info['sfreq'], 'Hz')
    epochs_resampled = epochs.copy().resample(100, npad='auto')
    print('New sampling rate for plotting:', epochs_resampled.info['sfreq'], 'Hz')

    print('Convert to data frame')
    df = epochs_resampled.to_data_frame()
    #df = epochs.to_data_frame()
    df['subj'] = s_number
    df['session'] = session_no
    df['exercise_type'] = exercise_type
    epochs_for_export.append(df)
    
#Once looped through, and all downsampled epochs are appended, save the larger file.
df_export = pd.concat(epochs_for_export, ignore_index=False)
df_export.to_csv('edited_OS_solution_epochs_for_plotting.csv', sep=',')

## Extracting the window information
#create (initially empty) lists to store the individual data in
dfs = []
retrieved = []
grand_avg = []

#define windows of interest
windows = {
        "prestim": (-1.3, -1.0),
        "readiness": (-1.0, 0.0),
        }

#create list of epoch files
epoch_files = glob.glob('solution_epochs/*epo.fif.gz')
print(epoch_files)
print(len(epoch_files))

#loop through epoch files to load for averaging and creating a grand average
for epoch_name in epoch_files:
    
    #create empty to list to append to
    epochs_df = []
    
    #extract participant number
    subj = op.basename(epoch_name).split('_')[0]
    session_no = op.split(epoch_name)[1][5:6]
    
    print('processing epochs for participant ' + subj)
    
    #read epochs
    epochs = mne.read_epochs(epoch_name,preload=True)
    
    sf = epochs.info['sfreq']
    print("sampling freq is", sf)
    
    #create 'wins' and add the window data for each participant
    #this will be used for statistical analyses later
    wins = retrieve(epochs, windows)
    
    #wins = retrieve(epochs, windows, epoch_name, summary_fnc=dict(mean=np.mean,sem=pd.DataFrame.sem))
    wins['subj']=subj
    wins['session']=session_no
    retrieved.append(wins)
    
#save csv file with windows from retrieved object
win_df=pd.concat(retrieved, ignore_index=False)

win_df.to_csv('solution_prestim_window_on_task.csv', sep=',', mode='a', header=True)
print("saving data to window csv")

