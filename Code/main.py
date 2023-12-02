from utils import *
from iterate_dataset import iterate_dataset
from clean_dataset import clean_dataset
from split_dataset import split_dataset
from k_fold_cross_validation import k_fold_cross_validation
from feature_extractor import *

def main():
    EGG = '' #'-egg' or empty. if empty string no EGG will be used. else runs for EGG
    clean = False #[boolean], eliminate all recordings < win_length & downsample the data to match the cardinality of the min cardinality class
    trim_to_shortest = True #[boolean], operate with the first min_sig_len [int] samples of each voice recording
    iterate = True  #[boolean], extract features by parsing the dataset
    split = True #[boolean], split the data according to split_perc
    perspeaker = True #[boolean], if True, Test folder will contain all the voicerecs from split_perc speakers. otherwise will contain voicerecs from (possibly) all the speakers
    tfrecord = False  #[boolean], save features in a TFRecord file
    train = False  #[boolean], train a new model, otherwise use the model from path_model
    model_folder = os.path.join('..','Model','model'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) #this folder will be made at training
    path_model = os.path.join(model_folder, 'model_1.h5')  #[path], relative path to save/load model to/from
    test_trained_model = True #if true tests the model that has been created. else, tests the model at pred_path_model
    pred_path_model = os.path.join('..','Model','testmodel','model.h5')
    predict = True  #[boolean], predict using a model that is loaded from path_model
    save_results = True
    shuffle_mode = False  # [boolean], if True shuffles train and validation datasets as one dataset, else individually
    check = True # [boolean] checks for speakers having more than one of the downloaded pathologies
    respath = os.path.join('..','Results')
    featspath = os.path.join('..','Features'+EGG)   #path to save the Features to

    #Arguments for clean dataset
    balance_classes = True #[boolean], if True downsample the dataset to the number of voicerecs contained in the lowest cardinality folder
    vowels = None #list of vowels to be kept from the SVD dataset; None to keep them all. can contain only "i", "a", "u"
    # vowels = ['a'] #list of vowels to be kept from the SVD dataset; None to keep them all. can contain only "i", "a", "u"
    intonations = None #list of intonations to be kept from the SVD dataset. can contain only "l", "h", "n"
    # intonations = ['h','l','n'] #list of intonations to be kept from the SVD dataset. can contain only "l", "h", "n"

    # Arguments for iterate dataset
    # db_path = os.path.join('..', 'Datasets'+EGG)  #[path], relative path to the pathology database
    db_path = os.path.join('..', 'Live-Exp/demo'+EGG)  #[path], relative path to the pathology database
    non_windowed_db = db_path
    if os.path.exists(os.path.join('..', 'Windowed_signals'+EGG)) and EGG == '':
        db_path = os.path.join('..', 'Windowed_signals'+EGG)

    overwrite = False #[boolean] if True delete the previously existing Features folder for a clean extraction
    window_length = 50000//50  #[int], length of a window in samples (1s = 50000 samples)
    window_type = 'rect'  #[string], type of window e.g: rect, blackman, etc
    overlap = 50  #[int] window overlap percentage (between 0 and 100)
    feature_extractor = FeatureExtractor()

    feature_list = ['mfcc']
    identity = False # if True, save the windowed signals to the disk to os.path.join('..','Windowed_signals')
    feature_dict = {'sr': 50000, 'n_fft': 2048, 'n_mfcc': 26, 'hop_length': 512//4, 'margin': 3.0, 'n_lvls':5,'wavelet_type':'db1'}  #[dict] of args for feature extractor (parameters for different functions)
    window = True  # [boolean], True = the features will be extracted window by window, False = directly from the array of signals.
    normalize = False  #[boolean], normalize the signal (song) in range [-1;1]
    standardize = False  #[bool], standardize the signal (song). For each sample x: x = (x - mean(signal)) / std(signal)
    variance_type = 'smad'  #[string], type of variance, either 'var' or 'smad'
    min_sig_len = None #[int], no of samples to trim the other voice recordings to; None if no trimming will be performed
    raw_features = True #[bool] if True, skips mean and var extraction from the audio features in the feature list
    keep_feature_dims = True #[bool] if True, do not reduce individual features' dimensions to 1D shape. Only useful if raw_features is True
    scaler_type = 'maxabs' # [str] can be one of: 'standard', 'minmax', 'maxabs'

    # Arguments for split dataset
    split_path = os.path.join('..', 'Features')+EGG  #[path], relative path to Features folder
    # split_perc = 90  #[int], percentage of Train files for splitting between Test and Train
    split_perc = 84  # test = 17% of 6 speakers = 1; 17% of 54 voicerecs = 9

    # Arguments for k_fold_cross_validation (train)
    k_fold_path = os.path.join('..', 'Train')  #[path], relative path to Train folder
    k = len(get_class_list(db_path))  #[int], number of folds to be performed 
    batch_size = 1024  #[int], size of batch in examples (windows)
    shuffle_buffer = 3 * batch_size  #[int], size of the buffer used to shuffle the data
    epochs = 530 #[int], number of epochs to be performed during training

    optimizer = 'adam'  #[string or tensorflow.keras.optimizers], optimizer to be used
    dropout = 0.5  #[float], between 0 and 1. Fraction of the input units to drop

    # Arguments for predict (test)
    per_class_accuracy = True  #[boolean], prints the accuracy of each class
    confusion_matrix = True  #[boolean], plots the confusion matrix
    roc_curve = True  #[boolean], plots the ROC curve with Area Under Curve calculated

    # database preprocessing
    if clean:
        clean_dataset(db_path, balance_classes, window_length, vowels, intonations, EGG)

    if check:
        check_db(db_path)

    # Start of program
    if iterate:
        if trim_to_shortest:
            min_sig_len = get_min_len(non_windowed_db)
            print("min_sig_len = ", min_sig_len)
        iterate_dataset(db_path, window_length, window_type, overlap, feature_extractor, feature_dict, feature_list,
                        window, normalize, standardize, tfrecord, variance_type, min_sig_len, overwrite, identity, featspath, raw_features, keep_feature_dims)
        save_classlist(db_path)

    if split:
        if tfrecord:
            features_cardinality = load_cardinality(os.path.join('..', 'Cardinality', 'Features.txt'))
        else:
            features_cardinality = None
        split_dataset(split_path, split_perc, features_cardinality, tfrecord, perspeaker)

    if not os.path.exists(os.path.join('..', 'Train')):
        raise Exception('No folder named Train. Please rerun with split parameter = True')

    scaler = None
    if train:
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        if scaler == None:
            scaler = compute_scaler(tfrecord, scaler_type=scaler_type)
        if tfrecord:
            train_cardinality = load_cardinality(os.path.join('..', 'Cardinality', 'Train_Features.txt'))
        else:
            train_cardinality = None
        k_fold_cross_validation(k_fold_path, k, path_model, tfrecord, train_cardinality, batch_size, shuffle_buffer,
                                epochs, optimizer, dropout, scaler, shuffle_mode)

    if predict:
        if scaler == None:
            scaler = compute_scaler(tfrecord, scaler_type=scaler_type)
        if test_trained_model:
            try:
                model = tf.keras.models.load_model(path_model)
            except:
                print("No model was trained to test at current execution. Testing the model at "+pred_path_model)
                model = tf.keras.models.load_model(pred_path_model)
                model.summary()
        else:
            model = tf.keras.models.load_model(pred_path_model)
        if tfrecord:
            x_test, y_true = create_test_tfrecord(os.path.join('..', 'TFRecord', 'Test_Features.tfrecord'), scaler)
        else:
            x_test, y_true = create_test_npy(os.path.join('..', 'Test'), window_length, scaler, overlap, non_windowed_db)

        y_score = test_model(model, x_test)
        y_pred = np.argmax(y_score, axis=1)
        if confusion_matrix:
            plot_confusion_matrix(y_true, y_pred)
        if per_class_accuracy:
            get_per_class_accuracy(y_true, y_pred)
        if roc_curve:
            plot_roc_curve(y_true, y_score)
    if save_results:
        saveresults(respath, balance_classes, perspeaker, vowels, intonations, shuffle_mode, variance_type)



if __name__ == '__main__':
    main()








