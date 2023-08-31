
import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image
import scipy.io.wavfile as wav
import pandas as pd

"""This script creates spectrogram matrices from wav files that can be passed 
to the CNN.
"""

prefix = '/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/Dataset/'
df_train = pd.read_csv('train_split_Depression_AVEC2017.csv')

df_test = pd.read_csv('dev_split_Depression_AVEC2017.csv')

df_dev = pd.concat([df_train, df_test], axis=0)

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """
    Short-time Fourier transform of audio signal.
    """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros((np.floor(frameSize/2.0).astype(int))), sig)
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    cols=cols.astype(int)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

def logscale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically.
    """
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  
    timebins, freqbins = np.shape(ims)
    ims = np.transpose(ims)
    ims = np.flipud(ims) 
    #if save_png:
        #create_png(ims, png_name)

    return ims

"""# Spectrogram_Dictionaries

This script builds dictionaries for the depressed and non-depressed classes
with each participant id as the key, and the associated segmented matrix
spectrogram representation as the value. Said values can than be cropped and
randomly sampled as input to the CNN.
"""

def build_class_dictionaries(dir_name):
    """
    Builds a dictionary of depressed participants and non-depressed
    participants with the participant id as the key and the matrix
    representation of the no_silence wav file as the value. These
    values of this dictionary are then randomly cropped and sampled
    from to create balanced class and speaker inputs to the CNN.
    Parameters
    ----------
    dir_name : filepath
        directory containing participant's folders (which contains the
        no_silence.wav)
    Returns
    -------
    depressed_dict : dictionary
        dictionary of depressed individuals with keys of participant id
        and values of with the matrix spectrogram representation
    normal_dict : dictionary
        dictionary of non-depressed individuals with keys of participant id
        and values of with the matrix spectrogram representation
    """
    depressed_dict = dict()
    normal_dict = dict()
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                partic_id = int(file.split('_')[0][1:])
                if in_dev_split(partic_id):
                    wav_file = os.path.join(subdir, file)
                    mat = stft_matrix(wav_file)
                    depressed = get_depression_label(partic_id) 
                    if depressed:
                        depressed_dict[partic_id] = mat
                    elif not depressed:
                        normal_dict[partic_id] = mat
    return depressed_dict, normal_dict

def in_dev_split(partic_id):
    """
    Returns True if the participant is in the AVEC development split
    (aka participant's we have depression labels for)
    """
    return partic_id in set(df_dev['Participant_ID'].values)

def get_depression_label(partic_id):
    """
    Returns participant's PHQ8 Binary label. 1 representing depression;
    0 representing no depression.
    """
    return df_dev.loc[df_dev['Participant_ID'] ==
                      partic_id]['PHQ8_Binary'].item()

if __name__ == '__main__':
    dir_name = os.path.dirname(os.path.realpath("segmented_audio"))
    depressed_dict, normal_dict = build_class_dictionaries(dir_name)

print("depressed_dict: ",len(depressed_dict))

print("normal_dict:",len(normal_dict))

"""Random_Sampling"""

import boto
import numpy as np
import os
import random

"""There exists a large data imbalance between positive and negative samples,
which incurs a large bias in classification. The number of non-depressed
subjects is about four times bigger than that of depressed ones. If these
samples for learning, the model will have a strong bias to the non-depressed
class. Moreover, regarding the length of each sample, a much longer signal of
an individual may emphasize some characteristics that are person specific.
To solve the problem, I perform random cropping on each of the participant's
spectrograms of a specified width (time) and constant height (frequency), to
ensure the CNN has an equal proportion for every subject and each class.
"""

np.random.seed(15)

def determine_num_crops(depressed_dict, normal_dict, crop_width=125):
    """
    Finds the shortest clip in the entire dataset which, according to our
    random sampling strategy, will limit the number of samples we take from
    each clip to make sure our classes are balanced.
    Parameters
    ----------
    depressed_dict : dictionary
        a dictionary of depressed participants with the participant id as the
        key and the segmented and concatenated matrix representation of
        their spectrograms as the values.
    crop_width : integer
        the desired pixel width of the crop samples
        (125 pixels = 4 seconds of audio)
    Returns
    -------
    num_samples_from_clips : int
        the maximum number of samples that should be sampled from each clip
        to ensure balanced classes can be built.
    """
    merged_dict = dict(normal_dict, **{str(k): v for k, v in depressed_dict.items()})
    shortest_clip = min(merged_dict.items(), key=lambda x: x[1].shape[1])
    shortest_pixel_width = shortest_clip[1].shape[1]
    num_samples_from_clips = shortest_pixel_width / crop_width
    return num_samples_from_clips

def build_class_sample_dict(segmented_audio_dict, n_samples, crop_width):
    """
    Get N (num_samples) pseudo random non-overlapping samples from the all
    the depressed participants.
    Parameters
    ----------
    segmented_audio_dict : dictionary
        a dictionary of a class of participants with keys of participant ids
        and values of the segmented audio matrix spectrogram representation
    n_samples : integer
        number of random non-overlapping samples to extract from each
        segmented audio matrix spectrogram
    crop_width : integer
        the desired pixel width of the crop samples
        (125 pixels = 4 seconds of audio)
    Returns
    -------
    class sample dict : dictionary
        a dictionary of a class of participants with keys of participant ids
        and values of a list of the cropped samples from the spectrogram
        matrices. The lists are n_samples long and the entries within the
        list have dimension (numFrequencyBins * crop_width)
    """
    class_samples_dict = dict()
    for partic_id, clip_mat in segmented_audio_dict.items():
            samples = get_random_samples(clip_mat, n_samples, crop_width)
            class_samples_dict[partic_id] = samples
    return class_samples_dict

def get_random_samples(matrix, n_samples, crop_width):
    """
    Get N random samples with width of crop_width from the numpy matrix
    representing the participant's audio spectrogram.
    """
    clipped_mat = matrix[:, (matrix.shape[1] % crop_width):]
    n_splits = clipped_mat.shape[1] / crop_width
    #print("clipped_mat",type(clipped_mat))
    #print("n_splits",type(n_splits))
    cropped_sample_ls = np.split(clipped_mat, n_splits, axis=1)
    #print("cropped_sample_ls",type(cropped_sample_ls))
    #print("n_samples",type(n_samples))
    samples = random.sample(cropped_sample_ls, int(n_samples))
    return samples

def create_sample_dicts(crop_width):
    """
    Utilizes the above function to return two dictionaries, depressed
    and normal. Each dictionary has only participants in the specific class,
    with participant ids as key, a values of a list of the cropped samples
    from the spectrogram matrices. The lists are vary in length depending
    on the length of the interview clip. The entries within the list are
    numpy arrays with dimennsion (513, 125).
    """
    # build dictionaries of participants and segmented audio matrix
    #dir_name = os.path.dirname(os.path.realpath("segmented_audio"))
    #depressed_dict, normal_dict = build_class_dictionaries(dir_name)
    n_samples = determine_num_crops(depressed_dict, normal_dict,
                                    crop_width=crop_width)
    depressed_samples = build_class_sample_dict(depressed_dict, n_samples,
                                                crop_width)
    normal_samples = build_class_sample_dict(normal_dict, n_samples,
                                             crop_width)
    for key, _ in depressed_samples.items():
        path = 'Randomly_Sampled_Data/'
        filename = 'D{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *depressed_samples[key])
    for key, _ in normal_samples.items():
        path = 'Randomly_Sampled_Data/'
        filename = '/N{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *normal_samples[key])

def rand_samp_train_test_split(npz_file_dir):
    """
    Given the cropped segments from each class and particpant, this fucntion
    determines how many samples we can draw from each particpant and how many
    participants we can draw from each class.
    Parameters
    ----------
    npz_file_dir : directory
        directory contain the
    crop_width : integer
        the desired pixel width of the crop samples
        (125 pixels = 4 seconds of audio)
    Returns
    -------
    num_samples_from_clips : int
        the maximum number of samples that should be sampled from each clip
        to ensure balanced classes can be built.
    """
    npz_files = os.listdir(npz_file_dir)

    dep_samps = [f for f in npz_files if f.startswith('D')]
    norm_samps = [f for f in npz_files if f.startswith('N')]
    max_samples = min(len(dep_samps), len(norm_samps))
    dep_select_samps = np.random.choice(dep_samps, size=max_samples,
                                        replace=False)
    norm_select_samps = np.random.choice(norm_samps, size=max_samples,
                                         replace=False)
    test_size = 0.2
    num_test_samples = int(len(dep_select_samps) * test_size)

    train_samples = []
    for sample in dep_select_samps[:-num_test_samples]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                train_samples.append(data[key])
    for sample in norm_select_samps[:-num_test_samples]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                train_samples.append(data[key])
    #y=(np.ones(len(train_samples)//2),np.zeros(len(train_samples)//2))
    #print("y:",y)
    train_labels = np.concatenate((np.ones(len(train_samples)//2),
                                   np.zeros(len(train_samples)//2)))
    test_samples = []
    for sample in dep_select_samps[-num_test_samples:]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                test_samples.append(data[key])
    for sample in norm_select_samps[-num_test_samples:]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                test_samples.append(data[key])
    test_labels = np.concatenate((np.ones(len(test_samples)//2),
                                  np.zeros(len(test_samples)//2)))

    return np.array(train_samples), train_labels, np.array(test_samples), \
        test_labels

if __name__ == '__main__':
    create_sample_dicts(crop_width=125)
    train_samples, train_labels, test_samples, \
        test_labels = rand_samp_train_test_split('Randomly_Sampled_Data')

    # save as npz locally
    print("Saving npz file locally...")
    np.savez('Randomly_Sampled_Data/train_samples.npz', train_samples)
    np.savez('Randomly_Sampled_Data/train_labels.npz', train_labels)
    np.savez('Randomly_Sampled_Data/test_samples.npz', test_samples)
    np.savez('Randomly_Sampled_Data/test_labels.npz', test_labels)
    print("Saved Locally")



