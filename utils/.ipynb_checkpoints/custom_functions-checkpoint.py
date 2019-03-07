# Functions to support the analysis of LFP
# Hernando Martinez Vergara
# December 2018

import numpy as np
import math
import re

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
    # calculate how many time events (frames) that is
    subFrames = int(math.floor(sToZero * samplingfreq.base))
    subTrace = trace - np.mean(trace[traceFrames<0][-subFrames:])
    
    # normalize
    # divide by the mean of the values within a time window
    norFrames = np.logical_and(traceFrames>normwindow[0], traceFrames<normwindow[1])
    normTrace = subTrace/abs(np.mean(subTrace[norFrames]))
    
    return normTrace



def parseTitlesForDates(expTitles):
    '''
    Parse the titles of open ephys to get only the data
    :param expTitles: list of titles of size X
    :return: list of dates of size X
    '''
    dates = []
    for title in expTitles:
        match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', title)
        dates.append(match.group())
        
    return dates



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
                      
