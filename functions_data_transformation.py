
import os
import pandas as pd
import numpy as np
from functions_general import *
from configurations import *
from functions_plots import *

SUITE2P_STRUCTURE = {
    "F": ["suite2p", "plane0", "F.npy"],
    "Fneu": ["suite2p", "plane0", "Fneu.npy"],
    'spks': ["suite2p", "plane0", "spks.npy"],
    "stat": ["suite2p", "plane0", "stat.npy"],
    "iscell": ["suite2p", "plane0", "iscell.npy"],
    'deltaF': ['suite2p', 'plane0', 'deltaF.npy'],
    'ops':["suite2p", "plane0", "ops.npy"],
    'cascade_predictions': ['suite2p', 'plane0', 'predictions_deltaF.npy']
}

def get_file_name_list(folder_path, file_ending, supress_printing = False): ## accounts for possible errors if deltaF files have been created before
    file_names = []
    other_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_ending=="F.npy" and file.endswith(file_ending) and not file.endswith("deltaF.npy"):
                    file_names.append(os.path.join(root, file))
            elif file_ending=="deltaF.npy" and file.endswith(file_ending) and not file.endswith("predictions_deltaF.npy"):
                    file_names.append(os.path.join(root, file))
            elif file_ending=="predictions_deltaF.npy" and file.endswith(file_ending):
                 file_names.append(os.path.join(root, file))
            elif file_ending=="samples":
                if file.endswith("F.npy") and not file.endswith("deltaF.npy"):
                    file_names.append(os.path.join(root, file)[:-21])
            else:
                 if file.endswith(file_ending): other_files.append(os.path.join(root, file))
    if file_ending=="F.npy" or file_ending=="deltaF.npy" or file_ending=="predictions_deltaF.npy":
        if not supress_printing:
            print(f"{len(file_names)} {file_ending} files found:")
            print(file_names)
        return file_names
    elif file_ending=="samples":
        check_deltaF(file_names)  #checks if deltaf exists, else calculates it
        if not supress_printing:
            print(f"{len(file_names)} folders containing {file_ending} found:")
            print(file_names)
        return file_names
    else:
        print("Is the file ending spelled right?")
        return other_files

def load_npy_array(npy_path):
    return np.load(npy_path, allow_pickle=True) #functionally equivalent to np.load(npy_array) but iterable; w/ Pickle

def load_npy_df(npy_path):
    return pd.DataFrame(np.load(npy_path, allow_pickle=True)) #load suite2p outputs as pandas dataframe

def check_deltaF(folder_name_list):
    for folder in folder_name_list:
        location = os.path.join(folder, *SUITE2P_STRUCTURE["deltaF"])
        if os.path.exists(location):
            continue
        else:
            calculate_deltaF(location.replace("deltaF.npy","F.npy"))
            if os.path.exists(location):
                continue
            else:
                print("something went wrong, please calculate delta F manually by inserting the following code above: \n F_files = get_file_name_list(folder_path = main_folder, file_ending = 'F.npy') \n for file in F_files: calculate_deltaF(file)")

def get_sample_dict(main_folder):
    '''returns a dictionary of all wells and the corresponding sample/replicate, the samples are sorted by date, everything sampled on the first date is then sample1, on the second date sample2, etc.'''
    well_folders = get_file_name_list(main_folder, "samples", supress_printing = True)
    date_list= []
    sample_dict = {}
    for well in well_folders:
        date_list.append(os.path.basename(well)[:6]) ## append dates
    distinct_dates = [i for i in set(date_list)]
    distinct_dates.sort(key=lambda x: int(x))
 
    for i1 in range(len(well_folders)):
        for i2, date in enumerate(distinct_dates):
            if date in well_folders[i1]: # if date in list
                sample_dict[well_folders[i1]]=f"sample_{i2+1}"
    return sample_dict

def create_df(suite2p_dict): ## creates df structure for single sample (e.g. well_x) csv file, input is dict resulting from load_suite2p_paths
    """this is the principle function in which we will create our .csv file structure; and where we will actually use
        our detector functions for spike detection and amplitude extraction"""
 
    ## spike_amplitudes = find_predicted_peaks(suite2p_dict["cascade_predictions"], return_peaks = False) ## removed
    # spikes_per_neuron = find_predicted_peaks(suite2p_dict["cascade_predictions"]) ## removed
 
    estimated_spike_total = np.array(summed_spike_probs_per_cell(suite2p_dict["cascade_predictions"]))
 
    basic_stats = basic_stats_per_cell(suite2p_dict["cascade_predictions"])
   
    ## all columns of created csv below ##
 
    df = pd.DataFrame({"IsUsed": suite2p_dict["IsUsed"],
                       "Skew": suite2p_dict["stat"]["skew"],
                       "EstimatedSpikes": estimated_spike_total,
                       "SD_ES":basic_stats[1],
                       "cv_ES":basic_stats[2],
                       "Total Frames": len(suite2p_dict["F"].T)-64,
                       "SpikesFreq": (estimated_spike_total / ((len(suite2p_dict["F"].T)-64)) * frame_rate), ## -64 because first and last entries in cascade are NaN, thus not considered in estimated spikes)
                       "group": suite2p_dict["Group"],
                       "dataset":suite2p_dict["sample"],
                       "file_name": suite2p_dict["file_name"]})
                      
    df.index.set_names("NeuronID", inplace=True)
    return df

def load_suite2p_paths(data_folder, groups, main_folder, use_iscell=False):  ## creates a dictionary for the suite2p paths in the given data folder (e.g.: folder for well_x)
    """here we define our suite2p dictionary from the SUITE2P_STRUCTURE...see above"""
    suite2p_dict = {
        "F": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["F"])),
        "Fneu": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["Fneu"])),
        "stat": load_npy_df(os.path.join(data_folder, *SUITE2P_STRUCTURE["stat"]))[0].apply(pd.Series),
        "ops": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE['ops'])).item(),
        "cascade_predictions": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["cascade_predictions"])),
    }
 
    if use_iscell == False:
        suite2p_dict["IsUsed"] = [(suite2p_dict["stat"]["skew"] >= 1)] 
        suite2p_dict["IsUsed"] = pd.DataFrame(suite2p_dict["IsUsed"]).iloc[:,0:].values.T
        suite2p_dict["IsUsed"] = np.squeeze(suite2p_dict["IsUsed"])
    else:
        suite2p_dict["IsUsed"] = load_npy_df(os.path.join(path, *SUITE2P_STRUCTURE["iscell"]))[0].astype(bool)
 
    for group in groups: ## creates the group column based on groups list from configurations file
        if (str(group)) in data_folder:
            suite2p_dict["Group"] = str(group[len(main_folder)+1:])
 
    sample_dict = get_sample_dict(main_folder) ## creates the sample number dict
   
    suite2p_dict["sample"] = sample_dict[data_folder]  ## gets the sample number for the corresponding well folder from the sample dict
 
    suite2p_dict["file_name"] = str(os.path.join(data_folder, *SUITE2P_STRUCTURE["cascade_predictions"]))
 
    return suite2p_dict
"""
Possible to append this function further for synapse exclusion
 for example, append the document based on 
suite2p_dict["stat"] using values for ["skew"]/["npix"]/["compactness"]
"""

#     ImgShape = getImg(suite2p_dict['ops']).shape
#     ImgShape = f"[{ImgShape[0]},{ImgShape[1]}]"
#     ImgShape = [ImgShape] * len(suite2p_dict['IsUsed'])
#     df = pd.DataFrame({"experiment_date": suite2p_dict["experiment_date"],
#                        "IsUsed": suite2p_dict["IsUsed"],
#                        "Skew": suite2p_dict["stat"]["skew"],
#                        "ImgShape": ImgShape,

#     # suite2p_dict["experiment_date"] = int(os.path.split(data_folder)[1][8:14]) #TODO change if needed in future
#     # suite2p_dict["file_name"] = os.path.split(data_folder)[1]


def create_output_csv(input_path, overwrite=False, check_for_iscell=False): ## creates output csv for all wells and saves them in .csv folder
    """This will create .csv files for each video loaded from out data fram function below.
        The structure will consist of columns that list: "Amplitudes": spike_amplitudes})
        
        col1: ROI #, col2: IsUsed (from iscell.npy); boolean, col3: Skew (from stats.npy); could be replaced with any 
        stat >> compactness, col3: spike frames (relative to input frames), col4: amplitude of each spike detected measured 
        from the baseline (the median of each trace)"""
    
    well_folders = get_file_name_list(input_path, "samples", supress_printing = True)

    output_path = input_path+r"\csv_files"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for folder in well_folders:
        output_directory = (os.path.relpath(folder, input_path)).replace('\\', '-')
        translated_path = os.path.join(output_path, f"{output_directory}.csv")
        if os.path.exists(translated_path) and not overwrite:
            print(f"CSV file {translated_path} already exists!")
            continue

        output_df = create_df(load_suite2p_paths(folder, groups, input_path))

        output_df.to_csv(translated_path)
        print(f"csv created for {folder}")

        suite2p_dict = load_suite2p_paths(folder, groups, input_path, use_iscell=check_for_iscell)
        ops = suite2p_dict['ops']
        Img = getImg(ops)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron = getStats(suite2p_dict['stat'], Img.shape, output_df)

        image_save_path = os.path.join(input_path, f"{folder}_plot.png")
        dispPlot(Img, scatters, nid2idx, nid2idx_rejected, pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path)

    print(f"{len(well_folders)} .csv files were saved under {main_folder+r'/csv_files'}")

## create .pkl and final df ##
def get_pkl_file_name_list(folder_path): 
    pkl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def list_all_files_of_type(input_path, filetype):
    return [os.path.join(input_path, path) for path in os.listdir(input_path) if path.endswith(filetype)]

def csv_to_pickle(main_folder, output_path, overwrite=True):
    '''creates pkl, output -> main_folder+r"\pkl_files"'''
    csv_files = list_all_files_of_type(main_folder+r"/csv_files", ".csv")
    print((csv_files))

    if not os.path.exists(main_folder+r"/pkl_files"):
        os.mkdir(main_folder+r"/pkl_files")

    for file in csv_files:
        df = pd.read_csv(file)
        pkl_path = os.path.join(output_path, 
                                        f"{os.path.basename(file[:-4])}"
                                        f"Dur{int(EXPERIMENT_DURATION)}s"
                                        f"Int{int(FRAME_INTERVAL*1000)}ms"
                                        f"Bin{int(BIN_WIDTH*1000)}ms"
                                            + ("_filtered" if FILTER_NEURONS else "") +
                                        ".pkl")
        if os.path.exists(pkl_path) and not overwrite:
            print(f"Processed file {pkl_path} already exists!")
            continue

        df.to_pickle(pkl_path)
        print(f"{pkl_path} created")
    print(f".pkl files saved under {main_folder+r'/pkl_files'}")

def create_final_df(main_folder):
    ''' creates the final datat frame (all the wells in one dataframe) from which further analyses can be done'''
    pkl_files = get_pkl_file_name_list(main_folder)
    df_list = []
    for file in pkl_files:
        df = pd.read_pickle(file)
        df_list.append(df)
    final_df = pd.concat(df_list, ignore_index=True)
    if len(get_file_name_list(main_folder, "samples")) != len(pkl_files):
        raise Exception("The amount of .pkl files doesn't match the amount of samples, please delete all .csv and .pkl files and start over")
    return final_df
    ##alternative df from cell_stats dict, add previous functions back in then

