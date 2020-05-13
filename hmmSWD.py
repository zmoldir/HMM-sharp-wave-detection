import time
import numpy as np
import pandas as pd
from scipy.signal import remez, filtfilt

from cal_neuroim.cal_neuroIm import pushToBaseline
from hmmlearn import hmm
import matplotlib.pyplot as plt


def extract_swr_coordinates(inputArray, swr_state_hits=2, state_chain_length=150,
                            swr_length=200, swr_distance=1000, max_length=10000):
    """
    inputArray: array of state-indices, e.g. [0,0,1,1,1,2,2,3,2,2,1,1,1 ...]

    swr_state_hits: number of "higher" (i.e. 2 or 3) state indices required in a chain for it to be declared a SWR

    state_chain_length: number of data points considered following a non-0 state occurrence as potentially SWR,
                        the potential slice of interest is extended for each 2,3-state within this interval

    swr_length: minimum length requirement for a state chain to be declared  SWR

    swr_distance: refractory period given in data points, in which no SWR following another is allowed

    max_length: maximum length permitted for a SWR, otherwise the event is ignored

    Outline:
    Iterate over the array, once state 1 is encountered, check if there is a state-chain
    if yes: is the min_length met and >= swr_state_hits found?
        if yes: get coordinates, move loop forward by swr_distance + event end (no SWR possible in this slice)
        if no: move loop forward by state_chain_length + current pos (no SWR possible in this slice)

    return list of coordinate start-end-tuples
    """
    swr_coordinate_list = []

    i = 1
    inputLength = len(inputArray) - 1
    while i < inputLength - swr_length:
        # state 1 or higher encountered, go to the inner loop for state-chain checking
        if inputArray[i] > 0:
            state_chain_start = i
            state_chain_limit = i + state_chain_length
            # inner loop that checks whether SWR-requirements are met by the current data chunk
            while i < state_chain_limit and i < inputLength:
                # set all values in current chunk to 1, except first occurrences of >1
                # example: 1,1,0,0,2,2,1,1,2,2 -> 1,1,1,1,2,1,1,1,2,1
                if inputArray[i] == inputArray[i - 1] or inputArray[i] == 0:
                    inputArray[i] = 1
                # if state is not 1, extend potential chain region
                if inputArray[i + 1] > 1:
                    state_chain_limit = i + state_chain_length
                i += 1
            # if a high-amplitude state was fitted or the chain is too long (gamma wave?), ignore this event
            if 4 in inputArray[state_chain_start:state_chain_limit] or state_chain_limit - state_chain_start > max_length:
                continue
            # if state train is long enough and has >= (min swr_state_hits > 1), add to SWR-list
            elif state_chain_limit - state_chain_start > swr_length \
                    and np.sum(inputArray[state_chain_start:state_chain_limit] - 1) >= swr_state_hits:
                swr_coordinate_list.append([state_chain_start, i])
                # forward iterator to respect refractory period between SWRs
                i += swr_distance
        i += 1

    return swr_coordinate_list


def filter_ripple_band(data, sampling_frequency=1500):
    # 150-250Hz bandpass filter
    band = [0, 150 - 25, 150, 250, 250 + 25, 0.5*sampling_frequency]
    filter_numerator = remez(101, band, [0, 1, 0], Hz=sampling_frequency)
    is_nan = np.isnan(data)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(filter_numerator, 1.0, data[~is_nan], axis=0)
    return filtered_data


df = pd.read_csv('/home/meop/Python/Projects/SWD/DetectSeg_transposed.txt', sep=",", header=None, dtype={'val': float}).values.T
values = []
start = time.time()

for row in df:
    downsampledList = [np.mean(row[i:i + 4]) for i in range(len(row) - 4) if i % 5 == 0]
    values.append(downsampledList)

valueArray = np.array(values)

print(time.time() - start)

baselineArray, baselineCoordinates = pushToBaseline(valueArray, bucketSize=1000)
filteredArray = filter_ripple_band(baselineArray, 20000)

movingAverageArray = [pd.DataFrame(filteredArray[i]).rolling(20).apply(lambda y: np.mean(y)).values for i in range(1)]
movingAverageArray = np.nan_to_num(np.array(movingAverageArray))
print(time.time() - start)
            
model = hmm.GaussianHMM(n_components=5, covariance_type='full', init_params='c', params="tc", n_iter=100)

model.startprob_ = np.array([1.0, 0, 0, 0, 0])

model.transmat_ = np.array([[0.9999, 0.00005, 0, 0, 0.00005],
                            [1 / 50, 48 / 50, 1 / 50, 0, 0
                             ],
                            [0, 1 / 50, 48 / 50, 1 / 50, 0],
                            [0, 0, 2 / 50, 48 / 50, 0],
                            [0.01, 0, 0, 0, 0.99]])

model.means_ = np.array([[0], [1], [-2], [4], [50]])

model.fit(movingAverageArray[0].reshape(-1, 1))

test = model.predict(movingAverageArray[0].reshape(-1, 1))

print(time.time() - start)

for columnIndex in [0, 1]:

    i, j = baselineCoordinates[columnIndex]
    baselineDeviance = np.std(movingAverageArray[columnIndex, i:j])

    model.means_ = [[0], [baselineDeviance * 8], [baselineDeviance * -14], [baselineDeviance * 20], [baselineDeviance * 200]]
    model.fit(movingAverageArray[columnIndex].reshape(-1, 1))
    # test = model.predict(movingAverageArray[columnIndex].reshape(-1, 1))

    ripples = extract_swr_coordinates(test, state_chain_length=100, swr_distance=500, swr_state_hits=15)
    plt.plot(baselineArray[columnIndex], 'g', alpha=0.7)
    plt.plot(movingAverageArray[columnIndex], alpha=0.8)
    plt.plot(test, alpha=0.5)
    for x in ripples:
        plt.plot(range(x[0], x[1]), movingAverageArray[columnIndex, x[0]:x[1]], 'r')
    plt.title(str(len(ripples)) + " " + str(baselineDeviance))
    plt.show()

print(time.time() - start)
