from imports import *

class MultiDim_MinMaxScaler():
    def __init__(self):
        self.min = None
        self.max = None
    def fit(self, x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        self.min = np.min(x)
        self.max = np.max(x)
    def transform(self,x): # scales its input to whatever min/max was extracted via fit()
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        x = x+np.abs(self.min)
        self.max+=+np.abs(self.min)
        return x/self.max

class MultiDim_MaxAbsScaler(): #if not in use modify name to end with _vertical
    #this scales everything vertically
    def __init__(self):
        self.maxabs = None
    def fit(self,x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        y = np.concatenate(x,axis=0)
        self.maxabs = np.max(np.abs(y),axis=0)
    def transform(self,x): # scales its input to whatever min/max was extracted via fit()
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        return x/self.maxabs

class MultiDim_MaxAbsScaler_orig():  #if not in use modify name to end with _original
    def __init__(self):
        self.maxabs = None
    def fit(self,x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        self.maxabs = np.max(np.abs(x))
    def transform(self,x): # scales its input to whatever min/max was extracted via fit()
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        return x/self.maxabs

def convert_to_sg_ch(x):
    if x.ndim==1:
        return x
    else:
        return np.mean(x, axis=1, dtype='float32')

def normalization(x):
    
    x_norm = x/max(np.abs(x))
    
    return x_norm #norm [-1,1]

def standardization(x):
    return (x-np.mean(x))/np.std(x)

def amplitude(W_signal):
    return np.abs(W_signal)

def phase(W_signal):
    return np.angle(W_signal)

def sigwin(x, l, w_type, overlap):
    """
    w_type[string] can be:  -rect
                            -boxcar
                            -triang
                            -blackman
                            -hamming
                            -hann
                            -bartlett
                            -flattop
                            -parzen
                            -bohman
                            -blackmanharris
                            -nuttall
                            -barthann

    overlap [percentage]
    l[sample number]
    x[list or np.array]
    """
    if l>len(x):
        raise Exception("sigwin ERROR: window length > signal length")

    overlap=overlap/100
    if type(x)==np.ndarray:
        x=x.tolist()
    w = []
    delay = int(np.round((1- overlap)*l))    #inainte era int simplu, fara round
    if delay == 0:
        raise Exception("sigwin ERROR: non_overlapping section must be greater than 0! Please check overlap and l parameters. ")
    if( w_type !='rect'):
        win = windows.get_window(w_type,l).tolist()

    for i in range(0, len(x), delay):
        if i+l<=len(x):
            if (w_type == 'rect'):
                w.append(x[i:i+l])
            else:
                w.append(np.multiply(win,x[i:i+l]))

    return np.array(w)


def sigrec(w_signal, overlap, mode='MEAN'):
    """
    Arguments:

        w_signal: an array with the windows of size #windows x window_length

        overlap: the percentage of overlapping between windows
        
        mode: method to reconstruct the signal:
		'OLA' for overlap and addition
		'MEAN' for overlap and mean (default if not 'OLA')

    Outputs:


        x: the reconstructed signal of size signal_length


    """

    n = len(w_signal)  # number of windows
    overlap = overlap / 100  # calc percentage
    l = len(w_signal[0])  # window len

    non_ov = int((1 - overlap) * l)  # non overlapping section of 2 windows
    lenx = (n - 1) * non_ov + l  # len of signal to reconstruct. formula might be wrong.
    delay = non_ov  # used to delay i'th window when creating the matrix that will be averaged

    w_frm_aux = np.zeros((n, lenx), dtype ='float32')  # size = windows x signal_length
    # dtype='float32' to reduce memory usage
    for i in range(0, len(w_signal)):
        crt = np.zeros(i * delay).tolist()
        crt.extend(w_signal[i])
        crt.extend(np.zeros(lenx - i * (delay) - l).tolist())

        w_frm_aux[i] += crt

    summ = np.sum(w_frm_aux, axis=0)
    if mode == 'OLA': return summ
    
    nonzero = w_frm_aux != 0
    divvect = np.sum(nonzero, axis=0)   
    divvect[divvect==0]=1   #avoid division by zero
    x = summ / divvect

    return x
    
def DWT(w_signal, wavelet_type):
    """
    wavelet_type can be:
        dbX, where X is a number in [1;38]
        symX, where X is a number in [2;20]
        coifX, where X is a number in [1;17]
        biorX, where X can be: (1.1, 1.3, 1.5,
                                2.2, 2.4, 2.6, 2.8,
                                3.1, 3.3, 3.5, 3.7, 3.9,
                                4.4, 5.5, 6.8)
        rbioX where X can be: (1.1, 1.3, 1.5,
                                2.2, 2.4, 2.6, 2.8,
                                3.1, 3.3, 3.5, 3.7, 3.9,
                                4.4, 5.5, 6.8)
        haar,
        dmey.
    """
    W_signal, _ = pywt.dwt(w_signal, wavelet_type, axis=-1)
    return np.asanyarray(W_signal)

def FFT(w_signal, N_fft):
    
    """
    Arguments:


        w_signal: an array of windows from a signal, of size #windows x window_length


        N_fft:  #of points to perform FFT

    Outputs:


        W_signal: an array containing the FFT of each window, of size #windows x N_fft

    """


    W_signal=np.fft.fft(w_signal,N_fft, axis = -1)

  
    return np.array(W_signal)

def IFFT(W_signal, l):
    
    """
    Arguments:

    W_signal: an array containing the FFT of each window, of size #windows x N_fft
    l: length of each window in the output array of windows
    Outputs:

    w_signal: an array of windows from a signal, of size #windows x window_length
    """
    try:
        w_signal=np.fft.ifft(W_signal, W_signal.shape[-1], axis = -1)[:, 0:l]
    except:
        w_signal = np.fft.ifft(W_signal, W_signal.shape[-1], axis=-1)[0:l]
    return w_signal

def autocorrelation(w_signal, N_fft):
    window_length = np.shape(w_signal)[1]
    autocorrel = IFFT(np.square(amplitude(FFT(w_signal, N_fft))), N_fft)[:, 0:window_length]
    return autocorrel.real

def utils_cepstrum(w_signal, N_fft):
    """
    Uses FFT, IFFT and log functions to calculate the Cepstrum
    """

    window_length = np.shape(w_signal)[-1]
    try:
        interm = amplitude(FFT(w_signal, N_fft))
        interm = 0.001 * np.float64(interm == 0) + interm
        C = IFFT(np.log(interm), N_fft)[:, 0:window_length]
    except:
        interm = amplitude(FFT(w_signal, N_fft))
        interm = 0.001*np.float64(interm==0) + interm
        C = IFFT(np.log(interm), N_fft)[0:window_length]

    return C.real
    
def next_pow_of_2(x):
    return int(np.power(2, np.ceil(np.log(x)/np.log(2))))

def MSE(voice_recording, reconstructed):
    """
    MSE returns a tuple of 2 elements:
    	1. MSE value of the entire signal
    	2. MSE value in each point
    """
    diff = voice_recording[0:len(reconstructed)] - reconstructed
    return (np.mean(diff**2), (diff**2))

def extract_features(w_signal, feature_extractor, feature_dict, feature_list, window=True, variance_type='var', raw_features = True, keep_feature_dims = False):
    """
    Arguments:
        - w_signal [2D-array], windowed signal
        - feature_extractor [FeatureExtractor object], used to extract the feature
        - feature_dict [dict], parameters needed for the extraction of features, e.g. features_dict['n_fft'] = 1024
        - feature_list [list], list of features to be extracted
        - window [boolean], if true extract features window by window
        - variance_type [string], either var for variance or smad for square median absolute deviation
    Output:
        - W_feature [2D-array]
    """

    W_feats = []
    if window:
        for win in w_signal:
            _ = feature_extractor.extract_features(win, ['spect', 'mel_spect'], feature_dict)
            features = feature_extractor.extract_features(win, feature_list, feature_dict, variance_type,raw_features,keep_feature_dims)
            W_feats.append(features)

    else:
        features = feature_extractor.extract_features(w_signal, feature_list, feature_dict, variance_type, raw_features)
        W_feats.extend(features)

    W_feats = np.asanyarray(W_feats)
    return W_feats

def save_features(featspath, pathology, voicerec, W_features, tfrecord, writer, db_path, overwrite = False):
    """
    If tfrecord is set to True:
    	The script adds the requested features in the Feature.tfrecord file
    Else:
    	The script creates (or overwrites) a voicerec.npy file which contains the requested features
    	of the selected .wav file.

    Arguments:
        - featspath [string], relative path of the Features folder to save the features to
        - pathology [string], name of the pathology
                                must be one of the pathologies in the database!
        - voicerec [string], name of the voice recording
        - W_features [2D-array], array of features to be writen, size #windows x #features
        - tfrecord [bool], if true saves W_features in the '.tfrecord' file
                           else it saves it as voicerec.npy file.
        - writer [tf.io.TFRecordWriter], the writer used to write records to a .tfrecord file.
    """

    pathologies = get_class_list(db_path)


    if not os.path.exists(featspath):
        os.mkdir(featspath) #create Features folder if does not already exist


    if pathology not in pathologies:
        raise Exception(pathology + ' is not a pathology of the Saarbrucken database')

    if tfrecord:
        windows_label = pathologies.index(pathology) * np.ones(W_features.shape[0], dtype=np.int64)
        labels = np.zeros([W_features.shape[0], len(pathologies)], dtype=np.int64)
        labels[np.arange(W_features.shape[0]), windows_label] = 1
        convert_to_tfrecord(writer, W_features, labels)
    else:
        pathol_path = os.path.join(featspath, pathology)
        if not os.path.exists(pathol_path):
            os.mkdir(pathol_path) #create voice_recording pathology folder if does not already exist
        if voicerec[-4:len(voicerec)] in ['.wav', '.npy', '.egg']:
            voicerec = voicerec[0:-4]
        file = os.path.join(pathol_path, voicerec)
        np.save(file, W_features)

def max_freq(W_signal, sample_rate):
    """
    This function returns the maximum frequency of a windowed Fourier transformed signal.
    
    W_signal: an array containing the FFT of each window, of size #windows x N_fft // 2


    sampling_rate: the frequency used to sample the voice_recording
    
    """
   
    f_max = (sample_rate/(2*W_signal.shape[1]))*np.max(np.argmax(W_signal,axis=1))
    
    return f_max


def pathology_maximum_frequency(path, window_length, window_type, overlap):
    """
    Function that finds the maximum frequency for each musical pathology in the GTZAN dataset, as well as the overall
    maximum frequency.

    Arguments:
        -path: the relative path to the GTZAN dataset
        -window_length: the length of the window (in samples) e.g. 11025 samples = 0.5s
        -window_type: the type of window used to split the signal, e.g. 'rect'
        -overlap: the percentage of overlapping between windows, e.g. 50%

    Outputs:
        -pathology_freq: a dictionary containing all the maximum frequencies for each musical pathology.
    """
    pathology_freq = {}

    for pathology in os.listdir(path):
        pathology_max = 0
        for file in os.listdir(os.path.join(path, pathology)):
            sample_rate, voice_recording = wavfile.read(os.path.join(path, pathology, file))
            voice_recording = convert_to_sg_ch(voice_recording)
            voice_recording = normalization(voice_recording)
            w_signal = sigwin(voice_recording, window_length, window_type, overlap)
            W_FFT = FFT(w_signal, next_pow_of_2(window_length))
            W_FFT = amplitude(W_FFT)[:, 0:W_FFT.shape[1]//2]

            f_max = max_freq(W_FFT, sample_rate)
            if pathology_max < f_max:
                pathology_max = f_max

        pathology_freq[pathology] = pathology_max

    highest_freq = max(pathology_freq.values())
    pathology_freq["Highest frequency"] = highest_freq

    return pathology_freq

def get_callbacks(path_model):
    logdir = '../Log/log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model_checkpoint = ModelCheckpoint(path_model, monitor='val_loss', verbose=1, save_best_only=True)
    return [model_checkpoint, tensorboard_callback]



def create_model(data, no_classes, optimizer, dropout_rate=0.5, summary=True): #_initial
    inshape = list(data.shape)[1::]
    input = Input(shape=inshape)
    input2 = input
    if len(inshape)>2:
        #dropout = 0.4
        inshape.append(1)
        inshape = tuple(inshape)
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = Activation('relu')(input2) #how many filters and what kernel size?
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = Activation('relu')(input2)
        # nl (macro layer nonlinearity lookup), nf = no filters, (c1,c2)kernel size. (3,5), (5,3) or (3,3) provided best accuracy
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = BatchNormalization()(input2)   # end macro layer 1
        input2 = MaxPooling2D(pool_size=(4, 4), padding='same')(input2)
        #todo add maxpool2D -> modified last dimension to 50


        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = Activation('relu')(input2) #unspecified by Radu Dogaru
        input2 = Conv2D(input_shape=inshape, filters=1, kernel_size=(3, 3), padding='same',data_format="channels_last")(input2)  # unspecified
        input2 = BatchNormalization()(input2) #end macro layer 2
        input2 = MaxPooling2D(pool_size=(4,4), padding='same')(input2)

        input2 = Conv2D(input_shape=inshape, filters=1, kernel_size=(3, 3), padding='same',data_format="channels_last")(input2)  # unspecified
        input2 = BatchNormalization()(input2)   #end macro layer 3
        input2 = MaxPooling2D(pool_size=(4,4), padding='same')(input2)
        input2 = GlobalAveragePooling2D()(input2)
        # todo add global average pooling and remove flatten
        input2 = Flatten()(input2)  #RDT processing might be the key. this is some sort of spectral feature thing
        # todo add dense
    else:
        print("in create model inshape = ", inshape)
        input2 = Conv1D(input_shape=inshape,filters=1,kernel_size=1,padding='same',data_format="channels_last")(input2)
        input2 = Flatten()(input2)
    # hdn1 = Dense(512, name='layer1')(input2)
    # act1 = Activation('relu')(hdn1)
    # act1 = BatchNormalization()(act1)
    # dp1 = Dropout(dropout_rate)(act1)
    #
    # hdn2 = Dense(256, name='layer2')(dp1)
    # act2 = Activation('relu')(hdn2)
    # # bn2 = BatchNormalization()(act2)
    # dp2 = Dropout(dropout_rate)(act2)
    #
    # hdn3 = Dense(128, name='layer3')(dp2)
    # act3 = Activation('relu')(hdn3)
    # # bn3 = BatchNormalization()(act3)
    # dp3 = Dropout(dropout_rate)(act3)
    #
    # hdn4 = Dense(64, name='layer4')(dp3)
    # act4 = Activation('relu')(hdn4)
    # # bn4 = BatchNormalization()(act4)
    # dp4 = Dropout(dropout_rate)(act4)
    #
    # hdn5 = Dense(32, name='layer5')(dp4)
    # act5 = Activation('relu')(hdn5)
    # # bn5 = BatchNormalization()(act5)
    # dp5 = Dropout(dropout_rate)(act5)
    hdn6 = Dense(no_classes)(input2)
    output = Activation('softmax')(hdn6)

    model = Model(inputs=input, outputs=output)

    if summary:
        print(model.summary())

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def create_model_original(data, no_classes, optimizer, dropout_rate=0.5, summary=True): #_initial
    inshape = list(data.shape)[1::]
    input = Input(shape=inshape)
    input2 = input
    if len(inshape)>2:
        inshape.append(1)
        inshape = tuple(inshape)
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2)
        input2 = Activation('relu')(input2)
        input2 = Flatten()(input2)
    else:
        print("in create model inshape = ", inshape)
        input2 = Conv1D(input_shape=inshape,filters=1,kernel_size=1,padding='same',data_format="channels_last")(input2)
        input2 = Flatten()(input2)
    hdn1 = Dense(512, name='layer1')(input2)
    act1 = Activation('relu')(hdn1)
    act1 = BatchNormalization()(act1)
    dp1 = Dropout(dropout_rate)(act1)

    hdn2 = Dense(256, name='layer2')(dp1)
    act2 = Activation('relu')(hdn2)
    # bn2 = BatchNormalization()(act2)
    dp2 = Dropout(dropout_rate)(act2)

    hdn3 = Dense(128, name='layer3')(dp2)
    act3 = Activation('relu')(hdn3)
    # bn3 = BatchNormalization()(act3)
    dp3 = Dropout(dropout_rate)(act3)

    hdn4 = Dense(64, name='layer4')(dp3)
    act4 = Activation('relu')(hdn4)
    # bn4 = BatchNormalization()(act4)
    dp4 = Dropout(dropout_rate)(act4)

    hdn5 = Dense(32, name='layer5')(dp4)
    act5 = Activation('relu')(hdn5)
    # bn5 = BatchNormalization()(act5)
    dp5 = Dropout(dropout_rate)(act5)
    hdn6 = Dense(no_classes)(dp5)
    output = Activation('softmax')(hdn6)

    model = Model(inputs=input, outputs=output)

    if summary:
        print(model.summary())

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


def create_model_deepCNN(data, no_classes, optimizer, dropout_rate=0.5, summary=True): #_deepCNN

    inshape = list(data.shape)[1::]
    input = Input(shape=inshape)
    input2 = input

    if len(inshape)>2:
        inshape.append(1)
        inshape = tuple(inshape)

        input2 = BatchNormalization()(input2)
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(4,2),padding='same',data_format="channels_last")(input2)
        input2 = MaxPooling2D(pool_size=(2,2), padding='same')(input2)
        ###############
        input2 = Conv2D(input_shape=inshape, filters=1, kernel_size=(2, 2), padding='same',
                        data_format="channels_last")(input2)  # (4,2) kernel size. was 1 initially

        input2 = MaxPooling2D(pool_size=(2, 2), padding='same')(input2)
        ###############
        input2 = Conv2D(input_shape=inshape, filters=1, kernel_size=(2, 2), padding='same',
                        data_format="channels_last")(input2)  # (4,2) kernel size. was 1 initially

        input2 = MaxPooling2D(pool_size=(2, 2), padding='same')(input2)
        input2 = Flatten()(input2)
    else:
        print("in create model inshape = ", inshape)
        input2 = Conv1D(input_shape=inshape,filters=1,kernel_size=1,padding='same',data_format="channels_last")(input2)
        input2 = Flatten()(input2)

    # hdn2 = Dense(256, name='layer2')(dp1)
    hdn2 = Dense(256, name='layer2',kernel_regularizer='l1_l2')(input2)
    act2 = Activation('relu')(hdn2)
    # bn2 = BatchNormalization()(act2)
    dp2 = Dropout(dropout_rate)(act2)

    hdn3 = Dense(128, name='layer3')(dp2)
    act3 = Activation('relu')(hdn3)
    # bn3 = BatchNormalization()(act3)
    dp3 = Dropout(dropout_rate)(act3)

    hdn4 = Dense(64, name='layer4')(dp3)
    act4 = Activation('relu')(hdn4)
    # bn4 = BatchNormalization()(act4)
    dp4 = Dropout(dropout_rate)(act4)

    hdn5 = Dense(32, name='layer5')(dp4)
    act5 = Activation('relu')(hdn5)
    # bn5 = BatchNormalization()(act5)
    dp5 = Dropout(dropout_rate)(act5)
    hdn6 = Dense(no_classes)(dp5)
    output = Activation('softmax')(hdn6)

    model = Model(inputs=input, outputs=output)

    if summary:
        print(model.summary())

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def train_model(model, tfrecord, train_dataset, val_dataset, batch_size, epochs, path_model):
    """
    If tfrecord is True:
        train_dataset and val_dataset must be a tensorflow.data.Dataset.Batch
    Else:
        train_dataset and val_dataset must be a tuple of (features, labels)
    """
    if tfrecord:
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2, callbacks=get_callbacks(path_model=path_model))
    else:
        model.fit(x=train_dataset[0], y=train_dataset[1], validation_data=(val_dataset[0], val_dataset[1]), 
                  batch_size=batch_size, epochs=epochs, verbose=2, callbacks=get_callbacks(path_model=path_model))
            

def test_model(model, dataset):
    return model.predict(dataset)
    
def write_cardinality(path, cardinality):
    """
    Write the cardinality of a TFRecordDataset in a ".txt" file.

    Arguments:
        - path [string], relative path to the ".txt" file
        - cardinality [int], the length of the TFRecordDataset

    Raises:
        - TypeError if cardinality is not an integer
    """
    if type(cardinality)!=int:
        raise TypeError("Cardinality is not an integer")
    else:
        dir_path = os.path.join(*path.split(os.sep)[0:-1])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        f = open(path, 'w')
        f.write(str(cardinality))
        f.close()

def load_cardinality(path):
    """
    Arguments:
        - path [string], relative path to the ".txt" file

    Output:
        - cardinality [int], the length of the TFRecordDataset
    """
    f = open(path, 'r')
    cardinality = f.read()
    f.close()
    return int(cardinality)

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_window_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    #example_proto needs to be a string scalar tensor

    voice_recording_feature_description = {
    'window': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    }
    
    return tf.io.parse_single_example(example_proto, voice_recording_feature_description)

def parse_and_decode_function(example_proto):
    feature_description = {
        'window': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_label = tf.io.decode_raw(element["label"], "int64")
    decoded_window = tf.io.decode_raw(element["window"], "float64")
    return (decoded_window, decoded_label)

def convert_to_tfrecord(writer, W_features, labels ):

    window_dict = {}
    for i in range(W_features.shape[0]):
        window_dict["window"] = bytes_feature(W_features[i].tobytes())
        window_dict["label"] = bytes_feature(labels[i].tobytes())

        ftexample = tf.train.Example(features=tf.train.Features(feature=window_dict) )
        ftserialized = ftexample.SerializeToString()
        writer.write(ftserialized)

def prep_dataset(dataset, batch_size,shuffle_buffer, shuffle_seed = 5):
    """
    A function that prepares the TFRecordDataset for the neural network model.
    This function should implement the mapping of each element in the dataset, followed by a shuffle and a group by batch.

    Arguments:
        - dataset: the dataset on which to perform the dataset.shuffle() and dataset.batch() methods
        - batch_size: the size of a batch used to train the model.
        - shuffle_buffer: the size of the buffer used to shuffle the data.
        - shuffle_seed: the value of the random seed that will be passed to the dataset.shuffle() function

    Outputs:
        - dataset: the TFRecordDataset to be used in the neural network.
    """
    decoded_batch_dataset = dataset.shuffle(shuffle_buffer, shuffle_seed, False).batch(batch_size, False)
    return decoded_batch_dataset
        
def squared_median_abs_dev(x):
    if len(x.shape) == 1:
        return scipy.stats.median_absolute_deviation(x)**2
    elif len(x.shape) == 2:
        return np.mean(scipy.stats.median_absolute_deviation(x, axis=1)**2)
    else:
        raise TypeError("Input must be a vector or a matrix")
    
def plot_confusion_matrix(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
    pathologies = load_classlist()
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=pathologies).plot()
    
    
def compute_scaler(tfrecord, with_mean = True, scaler_type = 'maxabs'):
    """
    Computes the scaler on the entire database.

    Arguments:
        - tfrecord [boolean], if true, reads data from TFRecord files, else from .npy files
        - scaler_type [string] can be 'standard', 'minmax', 'maxabs'
    Output:
        - scaler [a fitted sklearn.preprocessing scaler]
        Can be Standard -> [mean - 3*sigma, mean + 3*sigma] , MinMax -> default [0,1]  or MaxAbs -> [-1,1]
    """
    X = []
    if scaler_type not in ['standard', 'minmax', 'maxabs']:
        print("Please select scaler_type from: 'standard', 'minmax', 'maxabs' ")
        sys.exit()
    if scaler_type == 'standard':
        scaler = StandardScaler(with_mean=with_mean) # -> not just a maxabs +/-3, also modifies the distribution to Gauss
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()   #-> does indeed [0,1]. BUT requires individual feature vectors of max. 1D shape
    if scaler_type == 'maxabs':
        scaler = MaxAbsScaler()     # -> does indeed [-1,1]

    paths = [os.path.join('..', 'Train'), os.path.join('..', 'Test')]
    for path in paths:
        pathology_folders = sorted(os.listdir(path))
        for pathology in pathology_folders:
            files = sorted(os.listdir(os.path.join(path, pathology)))
            for file in files:
                npy = np.load(os.path.join(path, pathology, file))
                try:
                    if len(npy[0].shape)>=2:    #if 1st element is not a vector or a scalar
                        if scaler_type == 'minmax':
                            scaler = MultiDim_MinMaxScaler()
                        else:
                            scaler = MultiDim_MaxAbsScaler()   #if standard scale or maxabs [-1,1]
                except: #1st element is a scalar, scalars have no len()
                    pass
                X.extend(npy)

    scaler.fit(X)

    return scaler
        
def create_test_npy(path, window_length, scaler,overlap, db_path):
    """Creates x_test and y_test (true labels) from .npy files"""
    min_sig_len = get_min_len(db_path)
    non_ov = int(np.round((1 - overlap / 100) * window_length))
    no_of_windows = int((min_sig_len - (window_length)) // non_ov + 1)
    pathology_labels = {}
    classes = get_class_list(path)
    for i in range(len(classes)):
        pathology_labels[classes[i]] = i

    pathology_folders = sorted(os.listdir(path))
    x_test, y_true = [], []
    for pathology in pathology_folders:
        files = sorted(os.listdir(os.path.join(path, pathology)))
        for file in files:
            npy = np.load(os.path.join(path, pathology, file))
            x_test.extend(npy)
            for _ in range(no_of_windows):
                y_true.append(pathology_labels[pathology])
    x_test = scaler.transform(x_test)
    return (np.asanyarray(x_test), np.asanyarray(y_true))
   
def create_test_tfrecord(path, scaler):
    """Gets x_test and y_true (true labels) for test from """
    raw_dataset = tf.data.TFRecordDataset(os.path.join(path))
    parsed_dataset = raw_dataset.map(parse_and_decode_function)
    x_test, y_test , original_labels = [], [], []
    for element in parsed_dataset:
        x_test.append(element[0].numpy())
        y_test.append(np.argmax(element[1].numpy()))
        original_labels.append(element[1])
    x_test = scaler.transform(x_test)
    return (np.asanyarray(x_test), np.asanyarray(y_test))

def get_per_class_accuracy(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
    pathologies = load_classlist()
    print("\nPer class accuracy:")
    for line, (i,pathology) in zip(cm, enumerate(pathologies)):
        print("\tAccuracy of class ", pathology, "is: ", format(100*line[i]/np.sum(line), ".2f"), "%")
    print('\n')

def plot_roc_curve(y_true, y_score):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    pathologies = load_classlist()
    colors = ['purple','red', 'black', 'green', 'blue','chocolate', 'lime', 'orange', 'magenta','cyan'][0:len(pathologies)]
    plt.figure()
    y_test = to_categorical(np.asanyarray(y_true), num_classes=10)
    for i in range(len(pathologies)):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve for ' + pathologies[i] + ' (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def spectrogram(x, n_fft = 2048, hop_len = None, win_len = None, win_type = "hann", center = True):
    #todo could use boxcar window tho

    """
    This function extracts the spectrogram with the librosa STFT function.

    :param x:
    :return:
    """
    # default settings
    if win_len == None:
        win_len = n_fft
    if hop_len == None:
        hop_len = win_len//4

    return np.abs(librosa.stft(x,n_fft,hop_len,win_len,win_type,center))

def getline():
    callerframerecord = inspect.stack()[1]    # 0 represents this line
    # 1 represents line at caller
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    print("in file ", info.filename)                      # __FILE__     -> Test.py
    print("in function ", info.function)                  # __FUNCTION__ -> Main
    print("at line ", info.lineno)                        # __LINE__     -> 13

def get_min_files(path):

    """
        :param path: path to look for the folder containing the minimum number of files
        :return: min = no_files in minfiles_path, minfiles_path = path to the folder containing min no files
    """
    min = None
    minfiles_path = None
    for subdir, dirs, files in sorted(os.walk(path)):
        if min == None and len(files):
            min = len(files)
            minfiles_path = subdir
        elif min != None:
            if min>len(files) and len(files):
                min = len(files)
                minfiles_path = subdir

    return min, minfiles_path

def get_short_files(path, win_length = None, EGG=''):
    """
    This function indicates all the recordings < win_length from a given folder for further deletion.
    :param path: path to folder to delete from
    :return: list containing the paths of recordings to delete
    """

    deletlist = []
    for subdir, dirs, files in sorted(os.walk(path)):
        for file in files:
            if not file.endswith('.wav') and not file.endswith('.egg'):
                continue
            voicerec = os.path.join(subdir, file)
            x, sr = librosa.load(voicerec, sr = None)

            if win_length == None:
                win_length = sr
            if win_length>len(x):
                deletlist.append(voicerec)

    return deletlist

def get_min_len(path):
    """
    Returns the no_samples of the shortest recording from path
    :param path: string -> path to the root folder of the dataset

    """

    # print("path in get_min_len = ", path)
    min_len = None
    for subdir, dirs, files in sorted(os.walk(path)):
        i=0
        for file in files:
            i+=1
            voicerec = os.path.join(subdir, file)
            pathology = os.path.split(subdir)[1]
            # print("get_min_len reached file ", i, "/", len(files), " from " + pathology)
            if not file.endswith('.wav') and not file.endswith('.egg'):
                continue
            x,sr = librosa.load(voicerec, sr=None)
            if min_len == None:
                min_len = len(x)
            elif min_len > len(x):
                min_len = len(x)
    return min_len

def delete_speaker(path,EGG='',getshort = False):
    print("getshort = ", getshort)
    print("path = ", path)
    ending = '.wav'
    if(len(EGG)):
        ending = '.'+EGG[1:]
    if getshort:
        EGG = ''    #1st delete_speaker directly receives the correct speaker


    vowels = ['-a','-i','-u']
    intonations = ['_h','_l','_n']
    root = os.path.split(path)[0]
    speaker = os.path.split(path)[-1].split('-')[0] #select the speaker ID number only
    no_deleted = 0
    print("del path = ", path)
    for v in vowels:
        for i in intonations:
            if os.path.exists(os.path.join(root,speaker+v+i+EGG+ending)):
                os.remove(os.path.join(root,speaker+v+i+EGG+ending))
                no_deleted+=1
    return no_deleted
    return vowin

def count_speakers(path):
    """
        counts the speakers from a given folder of the Saarbrucken Voice Dataset
    """
    speakers = {}
    for s in os.listdir(path):
        if s.endswith('.wav') or s.endswith('.npy'):
            speakers[s.split('-')[0]] = 1
    return len(speakers)

def get_class_list(path):
    if not os.path.exists(path):
        raise Exception("The path: "+path+" does not exist!")
    for subdir, dirs, files in sorted(os.walk(path)):
        if len(dirs):
            return dirs

def count_files(path):
    """
        Counts the occurences of unique filenames
        in all the folders from "path", recursively.
        returns: dict {filename: no_occurences}
    """
    occ = {}
    for subdir, dirs, files in sorted(os.walk(path)):
        for file in files:
            if not file.endswith('.wav'):
                continue
            if not file in occ.keys():
                occ[file] = 1 #str(1)+'-'+os.path.split(subdir)[-1]
            else:
                occ[file] += 1 #str(int(occ[file][0]) + 1)+'-'+os.path.split(subdir)[-1]    #also add the pathology this belongs to
    return occ

def save_classlist(db_path):
    classes = get_class_list(db_path)
    if os.path.exists('classlist.npy'):
        os.remove('classlist.npy')
    np.save('classlist.npy', classes)

def load_classlist():
    if not os.path.exists('classlist.npy'):
        raise Exception('load_classlist() ERROR: no file "classlist.npy" found. This is because no features have been extracted. Try running main with iterate = True ')
    return np.load('classlist.npy')


def saveresults(respath, balance_classes, perspeaker, vowels, inton, shuffle_mode, variance_type):
    """
        Creates a folder to save the results of the current program run.
        The folder name will contain the program parameters and values.

    """
    if not os.path.exists(respath):
        os.mkdir(respath)
    dirname = "balance_classes_["+str(balance_classes)+"]-shuffle_mode_["+str(shuffle_mode)+"]-perspeaker_["+str(perspeaker)+"]-vowels_"+str(vowels)+"-inton_"+str(inton)+"-variance_type_["+variance_type+"]"
    print("dirname = ", dirname)
    if not os.path.exists(os.path.join(respath,dirname)):
        os.mkdir(os.path.join(respath,dirname))
    cls = str(load_classlist().tolist())
    if not os.path.exists(os.path.join(respath,dirname,cls)):
        print(os.path.join(respath, dirname, cls))
    os.startfile(os.path.join(respath,dirname))


def check_db(db_path):
    uniq = count_files(db_path)
    noniq = {}
    for file in uniq.keys():
        if uniq[file] > 1:
            noniq[file] = uniq[file]
    print("non unique = ", noniq)
    noniqfiles = {}
    for i in noniq:
        noniqfiles[i] = ''

    for subdir, dirs, files in sorted(os.walk(db_path)):
        if subdir == db_path:
            continue
        # print("subdir,dirs,files = ", subdir,"; ",dirs,"; ",files)

        for i in noniq.keys():
            # print("i = ", i)
            if i in files:
                # print("WTF")
                noniqfiles[i] = noniqfiles[i] + os.path.split(subdir)[-1] + ';'
    print(' ')
    print("non unique files = ", noniqfiles)  # laryngitis contains files both in dysphonie and pocketpuckered
