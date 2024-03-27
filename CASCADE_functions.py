## CASCADE functions ##

import os, warnings
import sys
import numpy as np
from configurations import *
import matplotlib.pyplot as plt

sys.path.insert(0,cascade_file_path) # cascade2p packages, imported from the downloaded Github repository
from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth, calculate_noise_levels


def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    if file_path.endswith('.npy'):
      traces = np.load(file_path, allow_pickle=True)
      # if saved data was a dictionary packed into a numpy array (MATLAB style): unpack
      if traces.shape == ():
        traces = traces.item()['dF_traces']

    else:
      raise Exception('This function only supports .npy files.') ## technically also exists for matlab files but dropped here, can be added back in if necessary (see original CASCADE code)

    print('Traces standard deviation:', np.nanmean(np.nanstd(traces,axis=1)))
    if np.nanmedian(np.nanstd(traces,axis=1)) > 2:
      print('Fluctuations in dF/F are very large, probably dF/F is given in percent. Traces are divided by 100.')
      return traces/100
    else:
        return traces

def plots_and_basic_info(deltaF_file): ## maybe make into one function with cascade_this, comment what part does what and which can be commented out if not needed

    ROI_number = len(np.load(deltaF_file))

    try:

      print(deltaF_file)
      traces = load_neurons_x_time(rf'{deltaF_file}')
      print('Number of neurons in dataset:', traces.shape[0])
      print('Number of timepoints in dataset:', traces.shape[1])

      ## histogram noise level across neurons
      warnings.filterwarnings('ignore')
      plt.rcParams['figure.figsize'] = [12, 5]
      plt.show()
      noise_levels = plot_noise_level_distribution(traces,frame_rate)

      ## df/f plots
      plt.rcParams['figure.figsize'] = [13, 13]
      #np.random.seed(3952)
      ## plot size calculation, plot 5% of ROIs, minimum 4 (code doesnt work for size < 3), can be removed or replaced by fixed number
      plot_number = 6 ## or if fixed percentage plot_number = int(0.05*ROI_number)
      if plot_number <4: plot_number = 4 ## can be removed
      neuron_indices = np.random.randint(traces.shape[0], size=plot_number)  ## if removed set number here or add plot_number = n at top
      time_axis = plot_dFF_traces(traces,neuron_indices,frame_rate)
      plt.show()

    except Exception as e:

      print('\nSomething went wrong!\nEither the target deltaF_file is missing, in this case please provide the correct location.\nOr your deltaF_file is not yet completely uploaded, in this case wait until the upload is completed.\n')
      print('Error message: '+str(e))

def cascade_this(deltaF_file, nb_neurons):

  # try:

    print(f"{deltaF_file}")
    traces = load_neurons_x_time(rf'{deltaF_file}')
    noise_levels = calculate_noise_levels(traces, frame_rate)

    #@markdown If this takes too long, make sure that the GPU runtime is activated (*Menu > Runtime > Change Runtime Type*).

    total_array_size = traces.itemsize*traces.size*64/1e9

    # If the expected array size is too large for the Colab Notebook, split up for processing
    if total_array_size < 10:

      spike_prob = cascade.predict(model_name, traces, model_folder = cascade_file_path+r"\Pretrained_models", verbosity=1)

    # Will only be use for large input arrays (long recordings or many neurons)
    else:

      print("Split analysis into chunks in order to fit into Colab memory.")

      # pre-allocate array for results
      spike_prob = np.zeros((traces.shape))
      # nb of neurons and nb of chuncks
      nb_neurons = traces.shape[0]
      nb_chunks = int(np.ceil(total_array_size/10))

      chunks = np.array_split(range(nb_neurons), nb_chunks)
      # infer spike rates independently for each chunk
      for part_array in range(nb_chunks):
        spike_prob[chunks[part_array],:] = cascade.predict(model_name, traces[chunks[part_array],:])

  ## The dF/F traces are shown in blue, the inferred spike probability is plotted in orange (shifted downwards by 1 for better visibility).
    print(f"\ncurrent file: {deltaF_file}")
    neuron_indices = np.random.randint(traces.shape[0], size=nb_neurons)
    time_axis = plot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob,y_range=(-1.5, 3))
    plt.show()


    ## Plots randomly drawn excerpts from the ground truth, re-sampled at the same frame rate and noise level as a typical recording of the test dataset.
    ## The resampled dF/F signal is shown in blue. The true spike rate convolved with a smoothing kernel is shown in orange (shifted downward by 1 for better visibility).
    ## This allows to directly compare **data quality** and **possible artifacts** of training dataset (ground truth) and test dataset (your calcium imaging data).

    ## Repeatedly execute this cell to plot new examples.

    median_noise = np.round(np.maximum(2,np.median(noise_levels)))
    nb_traces = 16
    duration = max(time_axis) - 64/frame_rate # seconds
    plot_noise_matched_ground_truth(model_name, median_noise, frame_rate, nb_traces, duration, cascade_file_path)
    plt.show()

    #@markdown By default saves as variable **`spike_prob`** both to a *.mat-file and a *.npy-file. You can uncomment the file format that you do not need or leave it as it is.

    folder = os.path.dirname(deltaF_file)
    file_name = 'predictions_' + os.path.splitext( os.path.basename(deltaF_file))[0]
    save_path = os.path.join(folder, file_name)

    # save as numpy file
    np.save(save_path, spike_prob)
    print(f"saved under {save_path} \n")

  # except Exception as e:

  #   print('\nSomething went wrong!\nEither the target file is missing, in this case please provide the correct location.\nOr your file is not yet completely uploaded, in this case wait until the upload is completed.\n')
  #   print('Error message: '+str(e))
