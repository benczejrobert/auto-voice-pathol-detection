from utils import *
from imports import *


def iterate_dataset(db_path, window_length, window_type, overlap, feature_extractor, feature_dict, feature_list,
                    window=True, normalize=False, standardize=False, tfrecord=False, variance_type='var', min_sig_len = None, overwrite = False, identity = False, featspath = os.path.join("..", "Features"), raw_features = True, keep_feature_dims = False):
    """
    Function that iterates through the files of the dataset and:
    Reads each signal (in our case, a voicerec)
    Transforms each signal to single channel
    Normalizes / standardizes each signal
    Applies the windowing technique over each signal    
    Extracts features from each array of windows of each signal  
    Saves the extracted features in a .txt file for each signal

    Args    
    + db_path: the path of the dataset relative to the source file's folder
    + window_length: length of a signal window (in samples)
    + window_type: the type of window used to split the signal, e.g. 'blackman', 'hamming', 'hanning', etc.
    + overlap: the percentage of overlapping between windows

    + feature_extractor: the FeatureExtractor object to be used for extracting the features.

    + feature_dict: the dictionary containing all the parameters needed for the extraction of features.
    Parameters can be: 'sr': 22050 [samples] , 'n_fft': 2048 [samples] , 'n_mfcc': 13 [int], 'hop_length': 512 [samples], 'margin': 1.0 [float]

    + feature_list: the list of features to be extracted. can contain: fft, dwt, autocorrelation, cepstrum, zcr, rms, spect, mel_spect, spectral_centroid, spectral_bandwidth, harmonics,
    chroma_stft, spectral_rolloff, tempo, mfcc

    + window: [boolean] If True, the features will be extracted window by window, otherwise, directly from the array of signals.

    + the feature to be extracted from a signal window.
    Can be:
        - normalize: boolean, decides if data will be normalized default False
        - standardize:  boolean, decides if data will be standardized default False
        - tfrecord: a boolean that indicates whether the .tfrecord conversion should apply.

    + variance_type: the variance to be used, either 'var' (the normal variance) or 'smad' (square median absolute deviation).

    + identity: if True, no features will be extracted and the windowed signal will be saved as is

    + raw_features: if True, skip the extraction of mean and var from the features
    """

    if overwrite and os.path.exists(featspath):
        rmtree(featspath)

    if not os.path.exists(db_path):
        raise Exception("Specified path does not exist! "+db_path)

    if tfrecord and not os.path.exists(os.path.join('..', 'TFRecord')):
        os.mkdir(os.path.join('..', 'TFRecord'))
    writer = None
    if tfrecord:
        writer = tf.io.TFRecordWriter(os.path.join("..", "TFRecord", "Features.tfrecord"))

    maximum = []
    features_cardinality = 0
    num_of_windows = None

    if min_sig_len != None:

        non_ov = int(np.round((1- overlap/100)*window_length))
        num_of_windows = int((min_sig_len-(window_length))//non_ov+1)


    for subdir, dirs, files in sorted(os.walk(db_path)):
        i = 0
        for file in sorted(files):
            i+=1
            voicerec = os.path.join(subdir, file)
            pathology = os.path.split(subdir)[1]
            print("iterate_dataset reached file ", i, "/", len(files), " from "+pathology )
            if (not voicerec.endswith('.wav')) and (not voicerec.endswith('.egg')) and (not voicerec.endswith('.npy') ):
                continue
            if db_path == os.path.join('..','Windowed_signals'):
                w_list = np.load(voicerec)
                pass
            else:
                x,rate = librosa.load(voicerec, sr = None)

                x = convert_to_sg_ch(x)

                if normalize == True:
                    x = normalization(x)
                if standardize == True:
                    x = standardization(x)

                w_list = sigwin(x[:min_sig_len], window_length, window_type, overlap)
                # print("iterate_dataset len wlist = ", len(w_list))

            if not identity:
                W_feats = extract_features(w_list, feature_extractor, feature_dict, feature_list, window, variance_type, raw_features, keep_feature_dims)
            else:
                W_feats = w_list
                featspath = os.path.join('..','Windowed_signals')
            if maximum == []:
                maximum = np.zeros(W_feats[0].shape) #np zeros of shape
            maximum = np.maximum(np.max(np.asanyarray(W_feats), axis=0), maximum)
            if num_of_windows != None:
                if W_feats.shape[0] > num_of_windows:
                    W_feats = W_feats[0:num_of_windows]
                else:
                    while (W_feats.shape[0] < num_of_windows):
                        non_overlap_dim = int((1 - overlap / 100) * W_feats.shape[1])
                        new_wind = W_feats[-1, non_overlap_dim:W_feats.shape[1]]
                        zeros = np.zeros(non_overlap_dim)
                        new_wind = np.concatenate((new_wind, zeros))
                        new_wind = np.reshape(new_wind, (1, new_wind.shape[0]))
                        W_feats = np.append(W_feats, new_wind, axis=0)
            save_features(featspath, pathology, file, W_feats, tfrecord, writer, db_path, overwrite)
            features_cardinality += W_feats.shape[0]



    if tfrecord:
        path_maximum = os.path.join('..', 'Cardinality', 'Maximum.txt')
        if not os.path.exists(os.path.join('..', 'Cardinality')):
            os.mkdir(os.path.join('..', 'Cardinality'))
        np.savetxt(path_maximum, maximum)
        card_path = os.path.join("..", "Cardinality", "Features.txt")
        write_cardinality(card_path, features_cardinality)