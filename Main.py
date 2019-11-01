import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ripple_detection import ripple_detection as rd
from cal_neuroim import cal_neuroIm as cni


def rootmeansquare(array):
    return np.sqrt(np.mean(array**2))


df = pd.read_csv('/home/meop/Python/Projects/SWD/18jul19_slice1ExTraces.txt', sep="\t", dtype="float32")

filteredDF = pd.DataFrame(rd.filter_ripple_band(df.values, 20000))

for column in filteredDF:
    currentSweep = filteredDF[column]
    currentSweep = currentSweep.rolling(30).apply(lambda x: rootmeansquare(x))
    filteredDF[column] = currentSweep
    print("still alive")

rmsArray = np.nan_to_num(filteredDF.values).T
transientListOfLists = cni.thresholdEventDetect(rmsArray, minEventLength=30, sigma=7, thresholdDeviance=2.5)

i = 0
for column, transients in zip(df.values.T, transientListOfLists):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(column)
    for transient in transients:
        ax.plot(range(transient.startTime, transient.endTime), column[transient.startTime:transient.endTime], 'r')
        ax.annotate(transient.numOfPeaks, xy=(transient.startTime, transient.data[0]), xytext=(transient.startTime, transient.data[0]))
    plt.title(str(len(transients)) + " number: " + str(i))
    i += 1
    plt.show()
