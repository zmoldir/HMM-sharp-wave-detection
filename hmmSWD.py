import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import remez, filtfilt
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

from SWD.extraction_logger import ExtractionLogger
from hmmlearn import hmm
import matplotlib.pyplot as plt
import pickle


def extract_swr_coordinates(input_array, swr_state_hits=2, state_chain_length=150,
                            swr_length=200, swr_distance=1000, max_length=10000,
                            extraction_logger=None):
    """
    :param input_array: array of state-indices, e.g. [0,0,1,1,1,2,2,3,2,2,1,1,1 ...]:

    :param swr_state_hits: number of "higher" (i.e. 2 or 3) state indices required in a chain for it to be declared a SWR:

    :param state_chain_length: number of data points considered following a non-0 state occurrence as potentially SWR,
                        the potential slice of interest is extended for each 2,3-state within this interval:

    :param swr_length: minimum length requirement for a state chain to be declared  SWR:

    :param swr_distance: refractory period given in data points, in which no SWR following another is allowed:

    :param max_length: maximum length permitted for a SWR, otherwise the event is ignored:

    :param extraction_logger: logger object to generate messages for skipped events

    Outline:
    Iterate over the array, once state 1 is encountered, check if there is a state-chain
    if yes: is the min_length met and >= swr_state_hits found?
        if yes: get coordinates, move loop forward by swr_distance + event end (no SWR possible in this slice)
        if no: move loop forward by state_chain_length + current pos (no SWR possible in this slice)

    :return: list of swr coordinate tuples [start, end]
    """

    # TODO make param defaults depenent on sampling freq, re-implement high freq noise thing
    swr_coordinate_list = []
    perform_length_check_bool = False
    perform_amplitude_check = False

    position_int = 0
    input_length = len(input_array) - 1
    input_array = input_array-3
    while position_int < input_length - swr_length:
        # set all noise states to 0
        if input_array[position_int] < 0:
            input_array[position_int] = 0
        # state 1 (post reduction) or higher encountered, go to the inner loop for state-chain checking
        else:
            state_chain_start = position_int
            state_chain_limit = position_int + state_chain_length
            # inner loop that checks whether SWR-requirements are met by the current data chunk
            while position_int < state_chain_limit and position_int < input_length:
                # noise state encountered, break off event chain here
                if input_array[position_int] < 1:
                    break
                # keep only state changes, set everything else to 1
                # example: 1,1,0,0,2,2,1,1,2,2 -> 1,1,1,1,2,1,1,1,2,1
                elif input_array[position_int] == input_array[position_int - 1] or input_array[position_int] == 2:
                    input_array[position_int] = 1
                # if state is not 1, extend potential chain region
                elif input_array[position_int] > 1:
                    state_chain_limit = position_int + state_chain_length
                position_int += 1
            # if the chain is too long (gamma wave?), ignore this event
            if perform_length_check_bool and state_chain_limit - state_chain_start > max_length:
                if extraction_logger:
                    extraction_logger.log_length_exception(state_chain_start, state_chain_limit)
                continue
            # if a high-amplitude state was fitted, ignore this event
            elif perform_amplitude_check and 4 in input_array[state_chain_start:state_chain_limit]:
                if extraction_logger:
                    extraction_logger.log_amplitude_exception(state_chain_start, state_chain_limit)
                continue
            # if state train is long enough and has >= (min swr_state_hits > 1), add to SWR-list
            elif state_chain_limit - state_chain_start > swr_length \
                    and np.sum(input_array[state_chain_start:state_chain_limit] - 1) >= swr_state_hits:
                swr_coordinate_list.append([state_chain_start, position_int])
                # forward iterator to respect refractory period between SWRs
                position_int += swr_distance
        position_int += 1

    return swr_coordinate_list


def filter_ripple_band(data, sampling_frequency=1500):
    # 150-250Hz bandpass filter
    band = [0, 100 - 25, 100, 250, 250 + 25, 0.5*sampling_frequency]
    filter_numerator = remez(101, band, [0, 1, 0], Hz=sampling_frequency)
    is_nan = np.isnan(data)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(filter_numerator, 1.0, data[~is_nan], axis=0)
    return filtered_data


def normalize_2d_df(input_matrix, sampling_rate, downsampling_factor=1, baseline_window_size=5000, moving_average_window=20, sigma=2):
    """
    :param input_matrix: Pandas.DataFrame to be normalized
    :param sampling_rate: sampling rate w.r.t. the downsampling factor (e.g. 10kHz raw sampling, down-sampled by 10 -> 1kHz)
    :param downsampling_factor: factor by which we sample down the input
    :param baseline_window_size: to be used for the baseline region
    :param moving_average_window: size of the window used for the moving average
    :param sigma: sigma used for gaussian filtering

    Performs, in this sequence:
    sample down to 1 out of samp_factor,
    select a baseline region of lowest variance and divide the data by absolute(mean(baselineRegion)),
    filter to the ripple band frequency,
    take moving average

    :returns: [array with all of the above, array with just baseline normalization, coordinates of baseline region]
    """
    values = []
    if downsampling_factor > 1:
        for row in input_matrix:

            downsampledList = [np.mean(row[counter:counter + downsampling_factor])
                               for counter in range(len(row) - downsampling_factor)
                               if counter % downsampling_factor == 0]

            values.append(downsampledList)

        input_matrix = np.array(values)

    baseline_array, baseline_coordinate_tuple = _pushToBaseline(input_matrix, bucketSize=baseline_window_size)
    input_matrix = filter_ripple_band(baseline_array, sampling_rate)

    if moving_average_window > 1:
        for row in input_matrix:
            tempSeries = pd.Series(row)
            input_matrix = tempSeries.rolling(moving_average_window).mean()
    if sigma > 1:
        filtered_array = gaussian_filter(input_matrix, sigma=sigma)
        return [filtered_array, baseline_array, baseline_coordinate_tuple]
    return [input_matrix, baseline_array, baseline_coordinate_tuple]


def _pushToBaseline(dataMatrix, bucketSize=300):
    """
    @param dataMatrix: matrix of the data whose baseline we're looking for
    @param bucketSize: size of the baseline bins
    @return: baseline corrected version of the data:
            coordList: tuple with start / end coordinate of the region determined as baseline activity, for visualization purposes
    """
    coordList = []
    for column in dataMatrix:
        baseLineArray, coordinates = _detectBaseline(column, bucketSize)
        meanVal = abs(np.mean(baseLineArray))
        coordList.append(coordinates)
        column[...] = column / meanVal - np.mean(baseLineArray / meanVal)  # baseline correction

    return dataMatrix, coordList


def _detectBaseline(data, bucketSize):
    """
    @param data: the array out of which the region with the lowest noise is to be identified
    @param bucketSize: size of the bins to be checked
    @return: bin with the lowest noise and its starting coordinate, in a tuple
    """
    data = np.trim_zeros(data, 'b')
    numOfEntries = len(data)
    lowestSigma = sys.maxsize  # for size comparison
    coordinate = []
    for j in range(0, int(numOfEntries - bucketSize), int(numOfEntries / (bucketSize * 2))):
        thisStd = np.std(data[j:j + bucketSize])  # current deviation
        if thisStd < lowestSigma:  # new min deviation found
            lowestSigma = thisStd
            coordinate = (j, j + bucketSize)
        baselineArray = data[coordinate[0]:coordinate[1]]
    return baselineArray, coordinate


start = time.time()

print("Loading data and normalizing ...")
# rawData = loadmat('./NewData/MaxTrace')['MaxTrace']
rawData = pd.read_csv("18jul19_slice1ExTraces.txt", sep="\t").to_numpy(dtype=float).T
moving_average_array, baseline_data, baseline_coordinate_list = normalize_2d_df(
    rawData, downsampling_factor=20, sampling_rate=1000, moving_average_window=100, sigma=2)
np.savetxt('movingAverageArray.txt', moving_average_array, delimiter=",", fmt="%2.5f")
np.savetxt('baselineCoordinates', baseline_coordinate_list)
np.savetxt('baselineData', baseline_data, delimiter=",", fmt="%2.5f")
'''
print("Loading normalized data")
moving_average_array = np.loadtxt('movingAverageArray.txt', dtype=float)
plt.plot(moving_average_array)
baseline_data = np.loadtxt('baselineData', dtype=float, delimiter=",")
baseline_coordinate_list = np.loadtxt('baselineCoordinates', dtype=float)
baseline_coordinate_list = [[int(baseline_coordinate_list[0]), int(baseline_coordinate_list[1])]]
'''
'''
rawAnnotations = loadmat('./NewData/RipPeakSec')['RipPeakSec']
np.savetxt('annotated_ripples.txt', rawAnnotations)
'''
annotated_ripples = np.floor(np.loadtxt('annotated_ripples.txt')*1000).astype(int)

nine_state_model = hmm.GaussianHMM(n_components=9, covariance_type='full', init_params="stc", n_iter=500, tol=0.1, verbose=True)
# TODO idea: fit PDF (multinomial, GMM?) to SWR data slices, use that as SWR state emission
# TODO cont.: fit PDF to noise, after-ripple event, gamma, as well
nine_state_model.transmat_ = np.array([[0.5, 0.25 - 0.000005, 0.25 - 0.000005, 0.000005, 0.000004, 0.000001, 0, 0, 0],
                                       [0.25-0.000005, 0.5, 0.25-0.000005, 0.000005, 0.000004, 0.000001, 0, 0, 0],
                                       [0.25-0.000005, 0.25-0.000005, 0.5, 0.000005, 0.000004, 0.000001, 0, 0, 0],
                                       [0.05, 0.05, 0.05, 0.83, 0.02, 0, 0, 0, 0],
                                       [0.05, 0.05, 0.05, 0.05, 0.77, 0.03, 0, 0, 0],
                                       [0.01, 0.01, 0.015, 0.05, 0.02, 0.85, 0.05, 0.005, 0],
                                       [0.01, 0.015, 0.01, 0, 0.07, 0.1, 0.8, 0, 0.005],
                                       [0, 0, 0, 0.2, 0, 0.5, 0, 0.3, 0],
                                       [0, 0, 0, 0, 0.2, 0, 0.5, 0, 0.3]])

nine_state_model.startprob_ = np.array([[1 / 3], [1 / 3], [1 / 3], [0], [0], [0], [0], [0], [0]])

moving_average_array = np.nan_to_num(moving_average_array)
# for 1d arrays -> code assumes 2d
lazycopy = moving_average_array.copy()
if moving_average_array.ndim <= 1:
    moving_average_array = np.asmatrix(moving_average_array)

for columnIndex in range(len(moving_average_array[0])):

    i, j = baseline_coordinate_list[columnIndex]
    baseline_deviance = np.std(moving_average_array[columnIndex, i:j])
    nine_state_model.means_ = [[0], [baseline_deviance * -2], [-baseline_deviance * 2],
                               [baseline_deviance * -20],
                               [baseline_deviance * 20],
                               [baseline_deviance * -40],
                               [baseline_deviance * 40],
                               [baseline_deviance * -60],
                               [baseline_deviance * 60]]
    '''
    print("Training new model - should you want to load one instead, comment in the according lines")
    training_limit = int(np.floor(moving_average_array.shape[1]/10))
    nine_state_model.fit(moving_average_array[columnIndex, 0:training_limit].reshape(-1, 1))
    model_file = open('nine_state_model_equal_transmat.pkl', "wb")
    pickle.dump(nine_state_model, model_file)
    '''
    print("Loading previously trained model - should you want to train one instead, comment in the according lines")
    model_file = open('nine_state_model_equal_transmat.pkl', "rb")
    nine_state_model = pickle.load(model_file)

    print('Training finished, predicting ...')
    test = nine_state_model.predict(moving_average_array[columnIndex].reshape(-1, 1))
    logger = ExtractionLogger()
    print('Prediction finished, evaluating ...')
    ripples = extract_swr_coordinates(test, state_chain_length=40, swr_distance=10000,
                                      swr_length=100, swr_state_hits=2, extraction_logger=logger)
    print('Evaluation finished, plotting ...')
    print("num of ripps: {0}".format(len(ripples)))
    for index, x in enumerate(ripples):
        raw_ripp = baseline_data[x[0]-500:x[1]+500]
        plt.plot(range(x[0]-500, x[1]+500), raw_ripp, 'r', alpha=0.7)
        plt.plot(range(x[0]-500, x[1]+500), test[x[0]-500:x[1]+500] * 1000, 'g*', alpha=0.5)
        plt.plot(range(x[0]-500, x[1]+500), lazycopy[x[0]-500:x[1]+500] * 1000, 'b-', alpha=0.5)
        plt.savefig('./imagedump/{0}'.format(index))
        plt.close()
print(time.time() - start)
