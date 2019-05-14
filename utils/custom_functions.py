# Functions to support the analysis of LFP
# Hernando Martinez Vergara
# December 2018

import numpy as np
import math
import re
from scipy.signal import butter
from IPython.display import clear_output
import itertools
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pylab as plt


def getFirstPulses(pulsevector, timedif):
    """
    From a vector of time events, get those that are separated
    from the preceiding one by at least some time
    :param pulsevector: 1D array
    :param timedif: float or int defining the time separation condition
    :return: 1D array of time stamps.
    """
    # calculate the difference between elements
    difvector = [j-i for i, j in zip(pulsevector[:-1], pulsevector[1:])]
    # add a 'fake' one at the beginning to compensate for the reduction of elements
    # and include the first element
    difvector = [timedif+1] + difvector
    
    # get the indeces of those bigger than the condition
    idx = [i for i,v in enumerate(difvector) if v > timedif]
    
    # return the vector
    return pulsevector[idx]



def alignTrace(trace, times, aligntimes, interval, samplingfreq):
    """
    Get a matrix of rows of events aligned to some time,
    spanning an interval
    :param trace: list of 1D arrays of x size (multiple traces of events)    
    :param times: 1D array of x size (times associated to traces)
    :param aligntimes: time events to align to
    :param interval: list of two elements with limits defining
    the length of the aligned traces. In seconds.
    :param samplingfreq: in Hz, the frequency of sampling
    :return: list of 2D arrays of time stamps. In rows the traces, 
    in columns the aligned times.
    """
    # find indexes in times of aligntimes values
    timeindexes = np.zeros(len(aligntimes))
    for i in range(len(timeindexes)):
        timeindexes[i] = np.where(times == aligntimes[i])[0]
    #convert it to integers
    timeindexes = timeindexes.astype(int)
    
    # calculate the number of events to include in each trace
    # convert the interval in seconds to bins using the sampling rate
    ev_limits = samplingfreq * interval
    ev_vector =  range(int(ev_limits[0]), int(ev_limits[1]))
    
    # create a list that will contain one array for each channel
    aligneddata = [None] * len(trace)
    
    for j in range(len(aligneddata)):
        # create a matrix of the desired size
        alignedmatrix = np.zeros((len(aligntimes), len(ev_vector)))

        # populate the matrix with the trace parts
        for i in range(alignedmatrix.shape[0]):
            eventsToSave = ev_vector + timeindexes[i]
            alignedmatrix[i,] = trace[j][eventsToSave]

        # append it to the list
        aligneddata[j] = alignedmatrix
    
    return aligneddata



def normalizeTrace(trace, traceFrames, samplingfreq, sToZero, normwindow):
    '''
    Normalize a trace by two things:
    First, subtract the mean of some time before time = 0 (to make it 0)
    Second, normalize the amplitude to the mean of a defined time window
    :param trace: 1D array of x size, values of the trace to normalize
    :param traceFrames: 1D array of x size, range from -a to b
    :param samplingfreq: in Hz, the frequency of sampling
    :param sToZero: in seconds, how much time before time 0 to use for subtraction
    :param normwindow: in seconds, a list of size 2 defining the window for normalization
    :return: 1D array of x size, normalized
    '''
    # subtract the baseline
    subTrace = translateTrace(trace, traceFrames, samplingfreq, sToZero)
    
    # normalize
    # divide by the minimum of the values within a time window
    norFrames = np.logical_and(traceFrames>normwindow[0], traceFrames<normwindow[1])
    normTrace = subTrace/abs(min(subTrace[norFrames]))
    
    return normTrace


def translateTrace(trace, traceFrames, samplingfreq, sToZero):
    '''
    Translate a trace to zero:
    subtract the mean of some time before time = 0 (to make it 0)
    :param trace: 1D array of x size, values of the trace to normalize
    :param traceFrames: 1D array of x size, range from -a to b
    :param samplingfreq: in Hz, the frequency of sampling
    :param sToZero: in seconds, how much time before time 0 to use for subtraction
    :return: 1D array of x size, translated
    '''
    # subtract the baseline
    # calculate how many time events (frames) that is
    subFrames = int(math.floor(sToZero * samplingfreq.base))
    subTrace = trace - np.mean(trace[traceFrames<0][-subFrames:])
    
    return subTrace


def parseTitlesForDates(expTitles):
    '''
    Parse the titles of open ephys to get only the dates
    :param expTitles: list of titles of size X
    :return: list of dates of size X
    '''
    dates = []
    for title in expTitles:
        match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', title)
        dates.append(match.group())
        
    return dates


def parseTitlesForProtocol(expTitles):
    '''
    Parse the titles of open ephys to get only the protocol used
    Titles require format: X_X_X_X_protocol_X
    :param expTitles: list of titles of size s
    :return: list of dates of size s
    '''
    protocols = []
    for title in expTitles:
        # split by underscore and select fifth element
        protocol = title.split('_')[4]
        protocols.append(protocol)
    return protocols


def timeDifferences(listOfDates):
    '''
    Return the absolute time, in days, of elements in a list of dates, related to the first
    :param listOfDates: list of size X of dates. Format: YYYY-MM-DD_HH-MM-SS
    :return: array of size X of absolute time 
    '''
    abstimeList = []
    for date in listOfDates:
        strList = re.split('_|-', date)
        intList = list(map(int, strList))
        # Calculate absolute time in days
        
        multipliers = [365, 30 ,1 ,1/24 ,1/(24*60), 1/(24*60*60)]
        mulList = [a*b for a,b in zip(intList,multipliers)]
        abstime = sum(mulList)
        abstimeList.append(abstime)
        
    diftime = np.array(abstimeList) - abstimeList[0]
    
    return diftime
               
       
def butter_bandpass(lowcut, highcut, fs, order=5, btype='band'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(17,17))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)
            
            # Create linear regression object
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(pd.DataFrame(data[x]), pd.DataFrame(data[y]))
            # Make predictions using the testing set
            #x_pred = pd.DataFrame(np.arange(min(data[x]), max(data[x]), (max(data[x])-min(data[x]))/100))
            x_pred = pd.DataFrame(data[x])
            y_pred = regr.predict(x_pred)            
            axes[x,y].plot(x_pred, y_pred, '-')
            # The r2 score
            axes[x,y].text(0.1, 0.2, np.corrcoef(data[x], data[y])[1][0], transform=axes[x,y].transAxes, color = 'blue')
            axes[x,y].text(0.1, 0.1, r2_score(data[y], y_pred), transform=axes[x,y].transAxes, color = 'blue')
    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig