
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configurations import *
from scipy.signal import find_peaks, peak_prominences


def find_predicted_peaks(cascade_predictions, return_peaks = True):
## make sure fits peak plots ##
    peaks_list = []
    amplitudes_list = []

    for cell in cascade_predictions:
    
        peaks, _ = find_peaks(cell, distance = 5)  ## adjust !!!
        amplitudes = cell[peaks]

        peaks_list.append(peaks)
        amplitudes_list.append(amplitudes)


    if return_peaks:
        return peaks_list
    else:
        return amplitudes_list


def basic_stats_per_cell(predictions_file):
    '''returns means, SDs, cvs for all cells in file, mean/SD/cv based on predicited spikes for this cell'''
    means = []
    sds = []
    cvs = []
    for cell in predictions_file:
        mean=np.nanmean(cell)
        means.append(mean)
        sd=np.nanstd(cell)
        sds.append(sd)
        if mean != 0:
            cv_cell = sd/mean
        else:
            cv_cell = np.nan ## cells that don't fire (--> mean spike probability 0) --> makes cv nan
        cvs.append(cv_cell)
    return means, sds, cvs

 
def summed_spike_probs_per_cell(prediction_deltaF_file):

    summed_spike_probs_cell = []

    for cell in prediction_deltaF_file:
        summed_spike_probs_cell.append(np.nansum(cell))

    return summed_spike_probs_cell


def calculate_deltaF(F_file):

    savepath = rf"{F_file}".replace("\\F.npy","") ## make savepath original folder, indicates where deltaF.npy is saved

    F = np.load(rf"{F_file}", allow_pickle=True)

    Fneu = np.load(rf"{F_file[:-4]}"+"neu.npy", allow_pickle=True)

    deltaF= []

    for f, fneu in zip(F, Fneu):
        corrected_trace = f - (0.7*fneu) ## neuropil correction

        amount = int(0.125*len(corrected_trace))
        middle = 0.5*len(corrected_trace)
        F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
        F_baseline = np.median(F_sample)
        deltaF.append((corrected_trace-F_baseline)/F_baseline)

    deltaF = np.array(deltaF)

    np.save(f"{savepath}/deltaF.npy", deltaF, allow_pickle=True)

    print(f"delta F calculated for {F_file[len(main_folder)+1:-21]}")

    csv_filename = f"{F_file[len(main_folder)+1:-21]}".replace("\\", "-") ## prevents backslahes being replaced in rest of code

    if not os.path.exists(main_folder + r'\csv_files_deltaF'): ## creates directory if it doesn't exist
        os.mkdir(main_folder + r'\csv_files_deltaF')

    np.savetxt(f"{main_folder}/csv_files_deltaF/{csv_filename}.csv", deltaF, delimiter=";") ### can be commented out if you don't want to save deltaF as .csv files (additionally to .npy)

    ## if done by pandas, version needs to be checked, np.savetxt might be enough anyways ##
    # df = pd.DataFrame(deltaF)
    # df.to_csv(f"{main_folder}"+"/csv_files/"+f"{file[len(main_folder)+1:-21]}"+".csv", index = False, header = False)

    print(f"delta F traces saved as deltaF.npy under {savepath}\n")

from functions_data_transformation import get_file_name_list

def overview(main_folder, groups):
    dictionary_list = []
    for group in groups:
        groups_predictions_deltaF_files = get_file_name_list(folder_path = group, file_ending = "predictions_deltaF.npy", supress_printing = True)
        for file in groups_predictions_deltaF_files:
            array = np.load(rf"{file}")
            neuron_count = len(array)
            estimated_spikes = []
            for i in range(len(array)):
                estimated_spikes.append(np.nansum(array[i]))
            total_estimated_spikes = round(sum(estimated_spikes), 2)
            dictionary_list.append({'Prediction_File': file[len(main_folder)+1:], 'Neuron_Count': neuron_count, 'Total Estimated Spikes': total_estimated_spikes, "Group":group[len(main_folder)+1:]})
    df = pd.DataFrame(dictionary_list)
    return df
