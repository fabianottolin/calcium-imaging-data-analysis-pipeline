import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import binary_dilation, binary_fill_holes
from configurations import *

def random_individual_cell_histograms(deltaF_file, plot_number):
    ## for individual cells, random sample of plot_number, (can also be set to randoms sample of size plot_number, i this case use code below to calculate plot number and then pass it to function) ##
    ### ROI_number = len(np.load(file)) ## needs to be connected with plot number below if we always want to show fixed percentage of all possible histograms
    ### plot_number = int(0.05*ROI_number) # plots random 5% of all cells
    ### if plot_number <4: plot_number = 4
    
    array = np.load(rf"{deltaF_file}")
    sample = random.sample(range(0, len(array)), plot_number)
    for i in sample: ## alterantive i in range(len(array)) to plot all
      plt.figure(figsize=(5,5))
      plt.hist(array[i], density=True, bins=200)
      plt.title(f'Histogram df/F fluorescence cell {i}')
      plt.show()

def deltaF_histogram_across_cells(deltaF_file):
    array = np.load(rf"{deltaF_file}")
    list = array.flatten()
    list_cleaned = [x for x in list if not np.isnan(x)]
    plt.figure(figsize=(5,5))
    plt.hist(list_cleaned, density=True, bins=200)
    plt.title(f'Histogram df/F {deltaF_file[len(main_folder)+1:]}')
    plt.show()

def histogram_total_estimated_spikes(prediction_deltaF_file):
    array = np.load(rf"{prediction_deltaF_file}")
    print(f"\n{prediction_deltaF_file}\nNumber of neurons in dataset: {len(array)}")
    estimated_spikes = []
    for i in range(len(array)):
        estimated_spikes.append(np.nansum(array[i]))
    print(f"For {prediction_deltaF_file[len(main_folder)+1:-38]} {int(sum(estimated_spikes))} spikes were predicted in total")
    plt.figure(figsize=(5,5))
    plt.hist(estimated_spikes, bins=50)
    plt.xlabel("Number of total estimated spikes per neuron")
    plt.title(f'Histogram total number of spikes per neuron \n {prediction_deltaF_file[len(main_folder)+1:-38]}')
    plt.show()

def plot_group_histograms(groups): ## plots histograms of total spikes per neuron for each group, possible to add a third group
    from functions_data_transformation import get_file_name_list
    for group in groups:
        predictions_deltaF_files_group = get_file_name_list(folder_path = group, file_ending = "predictions_deltaF.npy") 
        group_arrays = []
        estimated_spikes = []
        for file in predictions_deltaF_files_group:
            array = np.load(rf"{file}")
            group_arrays.append(array)        
        print(f"{len(group_arrays)} files found for group {group[len(main_folder)+1:]}")
        group_array = np.concatenate(group_arrays, axis=0)
        print(f"{len(group_array)} total neurons in {group}")
        for i in range(len(group_array)):
            estimated_spikes.append(np.nansum(group_array[i]))
        print(f"For group {group[len(main_folder)+1:]} {int(sum(estimated_spikes))} spikes were predicted in total")
        plt.figure(figsize=(5,5))
        plt.hist(estimated_spikes, bins=50, density=True)
        plt.ylim(0, 0.5)
        plt.xlim(0,100) ## maybe make dynamic (get_max_spike_across_frames() could be useful or slight alteration), so it's the same for all groups
        plt.title(f'Histogram estimated total number of spikes, {group[len(main_folder)+1:]}') ## y proprtion of neurons, x number of events, title estimated distribution total spike number
        plt.xlabel("Number of estimated spikes")
        plt.show()
        ## add titles axes labeling etc.

def single_cell_peak_plotting(input_f, title): ## input f needs to be single cell
    threshold = np.nanmedian(input_f)+np.nanstd(input_f)
    peaks, _ = find_peaks(input_f, distance = 5, height = threshold)
    plt.figure(figsize=(5,5))
    plt.plot(input_f)
    plt.plot(peaks, input_f[peaks], "x")
    plt.plot(np.full_like(input_f, threshold), "--",color = "grey") ## height in find_peaks
    plt.plot(np.full_like(input_f, np.nanmean(input_f)), "--", color = 'r')
    plt.title(title)
    plt.xlabel("frames")
    plt.show()

    ## not sure how useful, maybe calculate peaks by AUC??? ##

def visualization_process_single_cell(F_files, deltaF_files, predictions_deltaF_files, cells_plotted):
    for file_number in range(len(predictions_deltaF_files)):
        ## try with corrected trace too ??
        prediction_array = np.load(rf"{predictions_deltaF_files[file_number]}", allow_pickle=True)
        rawF_array = np.load(rf"{F_files[file_number]}", allow_pickle=True)
        deltaF_array = np.load(rf"{deltaF_files[file_number]}", allow_pickle=True)
        sample = np.random.randint(0,len(prediction_array), cells_plotted)
        for cell in sample:
            print(f"raw fluorescence {predictions_deltaF_files[file_number][len(main_folder)+1:-38]}, cell {cell}")
            single_cell_peak_plotting(rawF_array[cell], f"Raw fluorescence {predictions_deltaF_files[file_number][-45:-38]}, cell {cell}")
            print(f"delta F {predictions_deltaF_files[file_number][len(main_folder)+1:-38]}, cell {cell}")
            single_cell_peak_plotting(deltaF_array[cell], f"DeltaF {predictions_deltaF_files[file_number][-45:-38]}, cell {cell}")
            print(f"cascade predictions {predictions_deltaF_files[file_number][len(main_folder)+1:-38]}, cell {cell}")
            single_cell_peak_plotting(prediction_array[cell], f"Cascade predictions {predictions_deltaF_files[file_number][-45:-38]}, cell {cell}")
## maybe move those not used anymore to unused to other functions script

def get_max_spike_across_frames(predictions_deltaF_file_list):
    total_list=[]
    for file in predictions_deltaF_file_list:
        prediction_array = np.load(rf"{file}", allow_pickle=True)
        sum_rows = np.nansum(prediction_array, axis=0)
        total_list.extend(sum_rows)
    return(max(total_list))
## maybe move cause not related to plotting

def plot_total_spikes_per_frame(prediction_deltaF_file, max_spikes_all_samples):
    '''calculates the total spikes across whole culture at certain time point \n the first input is a prediction_deltaF_file, the second input determines the scaling of the y axis and can be calculated by get_max_spikes_across_data()'''
    prediction_array = np.load(rf"{prediction_deltaF_file}", allow_pickle=True)
    sum_rows = np.nansum(prediction_array, axis=0)
    plt.figure(figsize=(10,5))
    plt.plot(sum_rows, color = "green")
    plt.title(f'Total spike probability summed across cells per frame')
    plt.text(0.315, -0.115, f"{prediction_deltaF_file[len(main_folder)+1:-38]}", horizontalalignment='center', verticalalignment = "center", transform=plt.gca().transAxes)
    plt.ylim(0,max_spikes_all_samples+10) ## make dynamic
    plt.show

def plot_average_spike_probability_per_frame(predictions_deltaF_file):
    ''' plots average spike probability across all cells divided by total number of cells in dataset (regardless of active or not), standardizes output of plot_total_spikes_per_frame()'''
    prediction_array = np.load(rf"{predictions_deltaF_file}", allow_pickle=True)
    sum_rows = np.nansum(prediction_array, axis=0)
    average = sum_rows/(len(prediction_array))
    plt.figure(figsize=(10,5))
    plt.plot(average, color = "green", label="average spike probability")
    ## actief_aandeel = (get_active_proportion_list(file)) ##used to also plot "proportion" line, not used anymore cause interpretation difficult
    #plt.plot(actief_aandeel, color = "magenta", label = "proportion of active cells")
    #plt.legend()
    plt.title(f'Average spike probability across cells per frame')
    plt.text(0.315, -0.115, f"{predictions_deltaF_file[len(main_folder)+1:-38]}", horizontalalignment='center', verticalalignment = "center", transform=plt.gca().transAxes)
    plt.ylim(0,1)
    plt.show

## ROI image
def getImg(ops):
    """Accesses suite2p ops file (itemized) and pulls out a composite image to map ROIs onto"""
    Img = ops["meanImg"] # Also "max_proj", "meanImg", "meanImgE"
    mimg = Img # Use suite-2p source-code naming
    mimg1 = np.percentile(mimg,1)
    mimg99 = np.percentile(mimg,99)
    mimg = (mimg - mimg1) / (mimg99 - mimg1)
    mimg = np.maximum(0,np.minimum(1,mimg))
    mimg *= 255
    mimg = mimg.astype(np.uint8)
    return mimg

    #redefine locally suite2p.gui.utils import boundary
def boundary(ypix,xpix):
    """ returns pixels of mask that are on the exterior of the mask """
    ypix = np.expand_dims(ypix.flatten(),axis=1)
    xpix = np.expand_dims(xpix.flatten(),axis=1)
    npix = ypix.shape[0]
    if npix>0:
        msk = np.zeros((np.ptp(ypix)+6, np.ptp(xpix)+6), bool) 
        msk[ypix-ypix.min()+3, xpix-xpix.min()+3] = True
        msk = binary_dilation(msk)
        msk = binary_fill_holes(msk)
        k = np.ones((3,3),dtype=int) # for 4-connected
        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        out = binary_dilation(msk==0, k) & msk

        yext, xext = np.nonzero(out)
        yext, xext = yext+ypix.min()-3, xext+xpix.min()-3
    else:
        yext = np.zeros((0,))
        xext = np.zeros((0,))
    return yext, xext

#gets neuronal indices
def getStats(stat, frame_shape, output_df):
    """Accesses suite2p stats on ROIs and filters ROIs based on cascade spike probability being >= 1 into nid2idx and nid2idx_rejected (respectively)"""
    MIN_PROB = 1.0 
    pixel2neuron = np.full(frame_shape, fill_value=np.nan, dtype=float)
    scatters = dict(x=[], y=[], color=[], text=[])
    nid2idx = {}
    nid2idx_rejected = {}
    print(f"Number of detected ROIs: {stat.shape[0]}")
    for n in range(stat.shape[0]):
        estimated_spikes = output_df.iloc[n]["EstimatedSpikes"]

        if estimated_spikes >= MIN_PROB:
            nid2idx[n] = len(scatters["x"]) # Assign new idx
        else:
            nid2idx_rejected[n] = len(scatters["x"])

        ypix = stat.iloc[n]['ypix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1
        xpix = stat.iloc[n]['xpix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1

        valid_idx = (xpix>=0) & (xpix < frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
        ypix = ypix[valid_idx]
        xpix = xpix[valid_idx]
        yext, xext = boundary(ypix, xpix)
        scatters['x'] += [xext]
        scatters['y'] += [yext]
        pixel2neuron[ypix, xpix] = n

    return scatters, nid2idx, nid2idx_rejected, pixel2neuron


def dispPlot(MaxImg, scatters, nid2idx, nid2idx_rejected,
             pixel2neuron, F, Fneu, save_path, axs=None):
             if axs is None:
                fig = plt.figure(constrained_layout=True)
                NUM_GRIDS=12
                gs = fig.add_gridspec(NUM_GRIDS, 1)
                ax1 = fig.add_subplot(gs[:NUM_GRIDS-2])
                fig.set_size_inches(12,14)
             else:
                 ax1 = axs
                 ax1.set_xlim(0, MaxImg.shape[0])
                 ax1.set_ylim(MaxImg.shape[1], 0)
             ax1.imshow(MaxImg, cmap='gist_gray')
             ax1.tick_params(axis='both', which='both', bottom=False, top=False, 
                             labelbottom=False, left=False, right=False, labelleft=False)
             print("Neurons count:", len(nid2idx))
            #  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True) 
            #  mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_rainbow) 

             def plotDict(n2d2idx_dict, override_color = None):
                 for neuron_id, idx in n2d2idx_dict.items():
                     color = override_color if override_color else mapper.to_rgba(scatters['color'][idx])
                            # print(f"{idx}: {scatters['x']} - {scatters['y'][idx]}")
                            
                     sc = ax1.scatter(scatters["x"][idx], scatters['y'][idx], color = color, 
                                      marker='.', s=1)
             plotDict(nid2idx, 'g')
             plotDict(nid2idx_rejected, 'm')
             ax1.set_title(f"{len(nid2idx)} neurons used (green) out of {len(nid2idx)+len(nid2idx_rejected)} neurons detected (magenta - rejected)") 

             plt.savefig(save_path)
             plt.close(fig)

