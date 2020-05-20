import time
import numpy as np
import pandas as pd
from scipy.signal import remez, filtfilt
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

from SWD.extraction_logger import ExtractionLogger
from cal_neuroim.cal_neuroIm import pushToBaseline
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

    :return list of coordinate start-end-tuples:
    """
    swr_coordinate_list = []

    position_int = 0
    input_length = len(input_array) - 1
    input_array = input_array - 3
    while position_int < input_length - swr_length:
        # state 1 or higher encountered, go to the inner loop for state-chain checking
        if input_array[position_int] > 0:
            state_chain_start = position_int
            state_chain_limit = position_int + state_chain_length
            # inner loop that checks whether SWR-requirements are met by the current data chunk
            while position_int < state_chain_limit and position_int < input_length:
                # set all values in current chunk to 1, except first occurrences of >1
                # example: 1,1,0,0,2,2,1,1,2,2 -> 1,1,1,1,2,1,1,1,2,1
                if input_array[position_int] == input_array[position_int - 1] or input_array[position_int] == 0:
                    input_array[position_int] = 1
                # if state is not 1, extend potential chain region
                if input_array[position_int + 1] > 1:
                    state_chain_limit = position_int + state_chain_length
                position_int += 1
            # if the chain is too long (gamma wave?), ignore this event
            if state_chain_limit - state_chain_start > max_length:
                if extraction_logger:
                    extraction_logger.log_length_exception(state_chain_start, state_chain_limit)
                continue
            # if a high-amplitude state was fitted, ignore this event
            elif False and 4 in input_array[state_chain_start:state_chain_limit]:
                if extraction_logger:
                    extraction_logger.log_amplitude_exception(state_chain_start, state_chain_limit)
                    continue
            # if state train is long enough and has >= (min swr_state_hits > 1), add to SWR-list
            elif state_chain_limit - state_chain_start > swr_length \
                    and np.sum(input_array[state_chain_start:state_chain_limit] - 1) >= swr_state_hits:
                swr_coordinate_list.append([state_chain_start, position_int])
                # forward iterator to respect refractory period between SWRs
                position_int += swr_distance
        else:
            input_array[position_int] = 0
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
                               if counter % (downsampling_factor+1) == 0]

            values.append(downsampledList)

        input_matrix = np.array(values)

    baseline_array, baseline_coordinate_tuple = pushToBaseline(input_matrix, bucketSize=baseline_window_size)
    input_matrix = filter_ripple_band(baseline_array, sampling_rate)

    if moving_average_window > 1:
        for row in input_matrix:
            tempSeries = pd.Series(row)
            input_matrix = tempSeries.rolling(moving_average_window).mean()
    filtered_array = gaussian_filter(input_matrix, sigma=sigma)
    return [input_matrix, baseline_array, baseline_coordinate_tuple]


start = time.time()

'''
rawData = loadmat('myRipTrace.txt')['myRipTrace']

moving_average_array, baseline_data, baseline_coordinate_list = normalize_2d_df(
    rawData, downsampling_factor=15, sampling_rate=1333, moving_average_window=50, sigma=3)
np.savetxt('movingAverageArray.txt', moving_average_array, delimiter=",", fmt="%2.5f")
np.savetxt('baselineCoordinates', baseline_coordinate_list)
'''

moving_average_array = np.loadtxt('movingAverageArray.txt', dtype=float)
baseline_coordinate_list = np.loadtxt('baselineCoordinates')
baseline_coordinate_list = [[int(baseline_coordinate_list[0]), int(baseline_coordinate_list[1])]]

six_state_model = hmm.GaussianHMM(n_components=6, covariance_type='full', init_params="stc", n_iter=500, tol=0.1, verbose=True)
six_state_model.transmat_ = np.array([[0.5, 0.25 - 0.000005, 0.25 - 0.000005, 0.000005, 0.000005, 0],
                                      [0.25-0.000005, 0.5, 0.25-0.000005, 0.000005, 0.000005, 0],
                                      [0.25-0.000005, 0.25-0.000005, 0.5, 0.000005, 0.000005, 0],
                                      [0.01, 0.01, 0.01, 0.95, 0.02, 0],
                                      [0, 0, 0, 0.03, 0.94, 0.03],
                                      [0, 0, 0, 0.02, 0.08, 0.9]])

six_state_model.means_ = np.array([[0], [1.5], [-1.5], [6], [-9], [15]])
six_state_model.startprob_ = np.array([[1 / 3], [1 / 3], [1 / 3], [0], [0], [0]])

moving_average_array = np.nan_to_num(moving_average_array)
# for 1d arrays -> code assumes 2d
if moving_average_array.ndim <= 1:
    moving_average_array = np.asmatrix(moving_average_array)

for columnIndex in range(len(moving_average_array[0])):
    i, j = baseline_coordinate_list[columnIndex]
    print(moving_average_array[columnIndex, i:j])
    baseline_deviance = np.std(moving_average_array[columnIndex, i:j])
    baseline_mean = np.mean(moving_average_array[columnIndex, i:j])
    print(baseline_deviance, baseline_mean)
    '''model.means_ = [[0], [baseline_mean + baseline_deviance * 4],
                    [baseline_mean - baseline_deviance * -6],
                    [baseline_mean + baseline_deviance * 9],
                    [baseline_deviance * 100]]'''
    '''
    training_limit = int(np.floor(moving_average_array.shape[1]/50))
    six_state_model.fit(moving_average_array[columnIndex, 0:training_limit].reshape(-1, 1))

    model_file = open('six_state_model.pkl', "wb")
    pickle.dump(six_state_model, model_file)
    '''
    model_file = open('six_state_model.pkl', "rb")
    six_state_model = pickle.load(model_file)

    print('Training finished, predicting ...')
    test = six_state_model.predict(moving_average_array[columnIndex].reshape(-1, 1))
    logger = ExtractionLogger()
    print('Prediction finished, evaluating ...')
    ripples = extract_swr_coordinates(test, state_chain_length=100, swr_distance=100000, swr_state_hits=3, extraction_logger=logger)
    print('Evaluation finished, plotting ...')
    '''plt.plot(baseline_data[columnIndex], 'g', alpha=0.7)model.means_ = [[0], [baseline_mean+baselineDeviance * 4],
                    [baseline_mean-baselineDeviance * -6],
                    [baseline_mean+baselineDeviance * 9],
                    [baseline_mean * 1000]]
    '''
    plt.plot(moving_average_array[columnIndex, 0:5000], alpha=0.8)
    plt.plot(test[0:5000])
    plt.show()
    '''for x in ripples:
        plt.plot(range(x[0], x[1]), moving_average_array[columnIndex, x[0]:x[1]], 'r')
    plt.savefig('result.png')'''
    logger.print()
    for x in ripples:
        print(x)
print(time.time() - start)

