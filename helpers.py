import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from qpsolvers import solve_ls

# Some necessary functions.
def next_multiple(number, dividend):
    '''
    next_multiple
    Returns the next multiple of a given number, given a dividend.
    
    Parameters
    number: The number to obtain the closest multiple.
    dividend: The number used to obtain the next multiple.
    '''
    
    return int(np.ceil(number / dividend) * dividend)

def define_reliable_support(audio, clippinglevel):
    '''
    define_reliable_support
    Returns the reliable samples support vector. 
    The support vector of reliable samples is comprised by the samples of the audio signal that are not clipped.
     
    Parameters
    audio: The audio signal.
    clippinglevel: The clipping level.
    '''
    
    samples = []
    
    for i in np.arange(len(audio)):
        if audio[i] != clippinglevel and audio[i] != -clippinglevel:
            samples.append(i)
            
    return np.array(samples)