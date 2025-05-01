#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:14:36 2025

@author: chloedziego
"""

import mne
import matplotlib.pyplot as plt
from more_itertools import locate
import pandas as pd
import glob
import csv
import os.path as op
import os
import re
import numpy as np

#Create new dictionary for corrected triggers
new_triggers_dict = {"Cue/C  1": "right", "Cue/C  2": "left"}

#Specify all directories of critical information for markers and condition order
bv_file_paths = glob.glob("files_with_problems/*.vmrk")
intensity_order = pd.read_excel('files_with_problems/Intensity Order.xlsx',sheet_name=0,header=0, index_col=0)
output_filepath = "files_with_problems/edited_event_arrays/"

# define directories for cropping
incoming_dir = op.join('files_with_problems')
cropped_fif_dir = op.join('files_with_problems', 'fixed_and_cropped_bv')
cropped_dir  = op.join('files_with_problems', 'fixed_and_cropped_fif')
log_dir = 'files_with_problems/logs'

# other params
tasks = ['Ex']
eogs = ['LLeye', 'ROcanthus']

# Work out number of cores and n_jobs to use
try:
    import psutil
    n_jobs = psutil.cpu_count(logical=False)
    n_jobs = np.max([n_jobs-1,2])
except ImportError:
    import multiprocessing
    n_jobs = multiprocessing.cpu_count() // 2

n_jobs = int(n_jobs)

#code to make one file run through
#files = 'files_with_problems/P45-S2-Ex.vmrk'

#Begin loop to extract timing from 3 x opensesame files (L, M, H) for each full brainvision recording (Ex)
for files in bv_file_paths:
    
    #Create new empty lists to ensure there is no overlap of data/arrays
    aligned_markers = []
    all_condition_markers = []
    
    #extract participant and condition information
    file_parse = files.split('/')[1]
    subject_number = file_parse.split('-')[0]
    subj = int(subject_number.removeprefix('raw/P').replace('P',""))
    session = file_parse.split('-')[1]
    
    #find all relevant information and parallel behavioural files
    os_files = glob.glob(f"files_with_problems/opensesame_outputs/{subject_number}-{session}*.csv")
    print(' ')
    print('- - - - - - - ' * 5)
    print("           Creating new events array for " + subject_number + " session " + session)
    print('- - - - - - - ' * 5)
    
    for behavioural_file in os_files:
        
        #if needing one singular file to test, use this code:
        #behavioural_file = 'files_with_problems/opensesame_outputs/P48-S2-M.csv'
        
        #extract exercise condition information
        exercise_condition = behavioural_file.split('-')[2][0]
        
        #extract opensesame timing information across all three exercise sessions
        os_all_output = pd.read_csv(behavioural_file, sep = ",")  # Adjust separator if not tab-delimited
        os_first_target = os_all_output["time_trigger_vision"][0]
        trigger_info = os_all_output[["time_trigger_cue_if_left", "cue"]]
        trigger_info = trigger_info.pivot(columns='cue', values='time_trigger_cue_if_left')
        left = list(trigger_info["left"].dropna())
        right = list(trigger_info["right"].dropna())
        
        frame = {'left': left,
             'right': right}
        os_information = pd.DataFrame(frame)

        with open(files, 'r') as f:
            lines = f.readlines()
        header = [line for line in lines if line.startswith('[') or 'Brain Vision Data' in line]
        markers = [line for line in lines if line.startswith('Mk')]
        
        print("")
        print(f"Successfully extracted markers from BrainVision file for {subject_number} session {session}.") 
        print("The total number of markers appears to be: ")
        print(len(markers))
        print("")

        # Find the first vision trigger (not start trigger recorded) and its timing in both files
        opensesame_matchtime = os_first_target
        print(f"First target trigger timing from Opensesame File: {os_first_target}")
        
        # Find the trigger timing in the BrainVision marker file (ORIGINAL)
        #Participant P48-S2 has not comment for "light"
        if behavioural_file == 'files_with_problems/opensesame_outputs/P48-S2-L.csv': 
            brainvision_time = 366601
            print("This participant does not have a light comment in brainvision marker file, brainvision time manually specified.") 
        elif behavioural_file == 'files_with_problems/opensesame_outputs/P19-S2-M.csv': 
            brainvision_time = 366601
            print("This participant does not have a light comment in brainvision marker file, brainvision time manually specified.") 
        else:
            if exercise_condition == "L": 
                condition_trigger = "Light"
            elif exercise_condition == "M": 
                condition_trigger = "Moderate"
            elif exercise_condition == "H": 
                condition_trigger = "Hard"
                
            #empty the list to ensure old values are not carrying over      
            first_target_trigger = []
            types_of_target_triggers = ["T  1", "T  2", "Target/T  2", "Target/T  1"]
        
            for marker in markers:
                parts = marker.strip().split(',')
    
                if parts[1].strip() == condition_trigger:
                    print("Trigger for start of exercise condition has been found at")
                    print(f"{parts}")
                    condition_marker_index = (markers.index(marker))
                    print(f"Marker index is {condition_marker_index}")
                    
                    following_markers = [condition_marker_index+1, condition_marker_index+2, condition_marker_index+3, condition_marker_index+4, condition_marker_index+5]
                    
                    if markers[following_markers[0]].strip().split(',')[1] in types_of_target_triggers:   
                        first_target_trigger = markers[following_markers[0]]
                    elif markers[following_markers[1]].strip().split(',')[1] in types_of_target_triggers:
                        first_target_trigger = markers[following_markers[1]]
                    elif markers[following_markers[2]].strip().split(',')[1] in types_of_target_triggers:
                        first_target_trigger = markers[following_markers[2]]
                    elif markers[following_markers[3]].strip().split(',')[1] in types_of_target_triggers:
                        first_target_trigger = markers[following_markers[3]]
                    elif markers[following_markers[4]].strip().split(',')[1] in types_of_target_triggers:
                        first_target_trigger = markers[following_markers[4]]
                    else: 
                        raise ValueError(f"First target trigger after {condition_trigger} comment not found in the BrainVision marker file.")
                        
                    print(f"First target trigger from Brainvision marker file found: {first_target_trigger}")
                    brainvision_time = int(first_target_trigger.strip().split(',')[2])
                    print(f"Timing extracted: {brainvision_time}")
                    break

        offset_time = brainvision_time
        
        #Get rid of first cue trigger in each segment, as I can only indentify correct timing after the aligned target triggers
        if left[0] > right[0]: first_cue_trigger = right[0]
        if left[0] < right[0]: first_cue_trigger = left[0]
        
        # Insert new triggers based on the dictionary
        #Empty for_mne each time the code is run to ensure that only new triggers are added
        for_mne = []
        
        for new_trigger, column_name in new_triggers_dict.items():
            if column_name in os_information.columns:
                for _, row in os_information.iterrows():
                    opensesame_time = row[column_name]
                    if opensesame_time != first_cue_trigger:
                        trigger_mapping_no = []
                        
                        #Adjust BrainVision time to consider sampling interval (2000 microseconds = 2 ms)
                        marker_brainvision_time = int(((opensesame_time - opensesame_matchtime) / 2) + offset_time)
                        new_marker = f"Mk={new_trigger},{marker_brainvision_time},1,0"
                        if column_name == "left": trigger_mapping_no = 99
                        if column_name == "right": trigger_mapping_no = 98
                        newmarker_line = [marker_brainvision_time, 0, trigger_mapping_no]
                        
                        #for mne is the array with timing and duration and trigger numbers for saving as numpy, matches event format.
                        for_mne.append(newmarker_line)
                
                        #aligned markers is the brainvision marker looking information for potentially saving if cannot use arrays
                        #aligned_markers.append(new_marker)
       
        #check the length of for_mne 
        
        print("Check length of the new array of cues:")
        print(len(for_mne))  
        
        all_condition_markers.extend(for_mne)
        markers_to_add_to_events = np.array(all_condition_markers)
        sorted_to_add_markers = markers_to_add_to_events[np.argsort(markers_to_add_to_events[:, 0])]
   
    print("Aligned markers have been added to the full array. Current trials completed (number of arrays):")
    print(len(sorted_to_add_markers))
    
    print(f"**** Success! All triggers for {subject_number} {session} have been added to sorted_to_add_markers.")
        
    #Open up corresponding file in mne to extract events and add new event markers in
    corresponding_raw_file = f"files_with_problems/{subject_number}-{session}-Ex.vhdr"
    raw = mne.io.read_raw_brainvision(corresponding_raw_file, preload = True, eog = ['ROcanthus', 'LLeye'])
    
    events, event_id = mne.events_from_annotations(raw)
    
    #The annotation numbers change depending on the other annotations; thus, I have created entirely new numbers for left and right
    #These merge events functions will need to work off of Cue, C  1 and Cue, C  3 though!
    
    if 'Comment/Cue/C  1' in event_id: 
        error_cue_one = event_id['Comment/Cue/C  1']
        error_cue_three = event_id['Comment/Cue/C  3']
    elif 'Cue/C  1' in event_id:     
        error_cue_one = event_id['Cue/C  1']
        error_cue_three = event_id['Cue/C  3']
    else: 
        raise ValueError("Event ID does not match previous iterations of event_ids")

    merged_events = mne.merge_events(events, [error_cue_one, error_cue_three], 10099)
    combined_events = np.vstack((events, markers_to_add_to_events))
    sorted_events = combined_events[np.argsort(combined_events[:, 0])]
    
    only_new_cues = sorted_events[np.where(sorted_events[:,2]!=10099)[0]]
    
    print(f"Events have been merged and new markers combined into array for {subject_number} and session {session}. See here:")
    print(only_new_cues)
    print("New number of events:")
    print(len(only_new_cues))
    
    #To check the new markers are in the correct spots
    #raw.plot(events = only_new_cues, event_id = event_id)
    
    #Save the new events to be used for cropping later on.
    np.save(f"files_with_problems/edited_event_arrays/{subject_number}-{session}-new-array.npy", sorted_events)       

print('- - - - - - - ')
print("**** Success! All new arrays have been created. *****")
print('- - - - - - - ')

## ----------------------------------------------------
#Editing Ina's original cropping script to crop problem files with newly created event_ids
## ----------------------------------------------------

#Collate all problematic raw files
raw_files = glob.glob(os.path.join(incoming_dir, '*.vhdr'))

#This code is to singularly process one file to check all code is working correctly
#raw_file = os.path.join(incoming_dir, 'P48-S2-Ex.vhdr')

# open log file to record any BIDS conversion errors
with open(op.join(log_dir,'crop_log.csv'),'w', newline='') as crop_log:
    writer = csv.DictWriter(crop_log, fieldnames=["id", "session", "comment"])
    writer.writeheader()
    
    for raw_file in raw_files:

        file_name = raw_file.removesuffix('.vhdr').lstrip('P')
        subj = op.basename(raw_file).split('-')[0]
        subject = int(subj.removeprefix('raw/P').replace('P',""))
        session = op.basename(raw_file).split('-')[1]
        task = op.splitext(op.basename(raw_file).split('-')[2])[0]
    
        if task == 'Ex':
            #import files
            try:
                raw = mne.io.read_raw_brainvision(raw_file, preload = True, eog = ['ROcanthus', 'LLeye'])
                print('Processing ' + file_name)
            except FileNotFoundError:
                print('No file found for ' + file_name)
                writer.writerow({"id": subj, "session": session, "comment": f"No file found for ' + {file_name}"})

            events_to_add = np.load(f"files_with_problems/edited_event_arrays/{subj}-{session}-new-array.npy")
            
            cues = mne.pick_events(events_to_add, include=[98, 99])
            print(f"Cues present: {len(cues)}") # should be approximately 216. Occasionally there is one or two extras near the end that get cropped out.  
            
            #append the new annotations onto the original
            #cues_to_add = mne.Annotations(cues[:,0]/500, cues[:,1], cues[:,2])
            raw.annotations.append(cues[:,0]/500, cues[:,1], cues[:,2])
            
            #extract events
            events, trigger_mappings = mne.events_from_annotations(raw)

            try:
                block_1_start_time, block_2_start_time, block_3_start_time = cues[0, 0]/500, cues[71, 0]/500, cues[142, 0]/500
            except IndexError:
                writer.writerow({"id": subj, "session": session, "comment": f"Error cropping {file_name}: Index out of range"})
                
            # create copies of raw data (as raw.crop acts on the orginal) 
            raw_2 = raw.copy()
            raw_3 = raw.copy()

            # crop separate data for each block of testing
            try: 
                block_1_raw = raw.crop(tmin=block_1_start_time -1, tmax=block_1_start_time+320)
                block_2_raw = raw_2.crop(tmin=block_2_start_time -1, tmax=block_2_start_time+320)
                block_3_raw = raw_3.crop(tmin=block_3_start_time -1, tmax=block_3_start_time+320)

            except ValueError:
                writer.writerow({"id": subj, "session": session, "comment": f"Error cropping {file_name}: Recording duration exceeded"})

            # determine exercise intensity of each block
            if session == 'S1':
                order = intensity_order.loc[subject, 's1_order']
            else:
                order = intensity_order.loc[subject, 's2_order']
            intensity_1, intensity_2, intensity_3 = order[0], order[1], order[2]

            outfolder = op.join(cropped_dir, f'{subj}-{session}')
            fif_outfolder = op.join(cropped_fif_dir, f'{subj}-{session}')

            block_1_raw.export(f'{outfolder}/{subj}-{session}-{intensity_1}.vhdr',fmt='brainvision',overwrite=True)
            block_2_raw.export(f'{outfolder}/{subj}-{session}-{intensity_2}.vhdr',fmt='brainvision',overwrite=True)
            block_3_raw.export(f'{outfolder}/{subj}-{session}-{intensity_3}.vhdr',fmt='brainvision',overwrite=True)
            
            
    #This is code to check that the new events and trigger mappings include the new cues and align with the old C3 trigger consistently
    #Draft while editing, checking the numbers for the cropped recordings.
    #events_two, trigger_mappings_two = mne.events_from_annotations(raw_cut_recording)
    #raw_cut_recording = mne.io.read_raw_brainvision(f'{outfolder}/{subj}-{session}-{intensity_3}.vhdr', preload=True)
    #raw_cut_recording.plot()
            








