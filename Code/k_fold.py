from utils import *

def k_fold(path, k, fold_nr, tfrecord, train_cardinality):
    '''
    Splits the Train set using k-fold principle.

    Input:
        - path [string], relative path to 'Train' folder
        - k [int], number of sub-sets to be generated
        - fold_nr [int], the sub-set to be used as validation data
                        ! must be between 1 and k !
        - tfrecord: a boolean that indicates to split the .tfrecord file.
        - train_cardinality: the length of the Train.tfrecord file.
    Output:
        - x_train [2D-array], features used as training data,
                            size: #pathologies * (k - 1)folds * #windows x #features
        - x_val [2D-array], features used as validation data,
                            size: #pathologies * folds * #windows x #features
        - y_train [1D-array], labels for the training data,
                            size: #pathologies * (k - 1)folds * #windows
        - y_val [1D-array], labels used for validation data,
                            size: #pathologies * folds * #windows
        - train_dataset: the Train TFReloadDataset used to train the neural network.
        - val_dataset: the Val TFReloadDataset used for validation in the neural network.
    '''
    if fold_nr > k or fold_nr <= 0:
        raise Exception("Incorect value for fold_nr")

    if tfrecord:
        songs_per_pathology = len(os.listdir('..' + os.sep + 'GTZAN' + os.sep + 'Blues'))
        no_of_pathologies = len(os.listdir('..' + os.sep + 'GTZAN'))
        features_cardinality = load_cardinality(os.path.join("..", "Cardinality", "Features.txt"))
        train_song_number = songs_per_pathology * no_of_pathologies * train_cardinality // features_cardinality
        split = train_cardinality / features_cardinality
        windows_per_song = train_cardinality // train_song_number
        if k == 1: # consider the first 10% songs (of each pathology) as validation data
            no_of_songs_for_val_per_pathology = int(split * songs_per_pathology // 10)
        else:
            no_of_songs_for_val_per_pathology = int(split * songs_per_pathology // k)
        range_min = (fold_nr - 1) * no_of_songs_for_val_per_pathology
        if k == 1:
            range_max = range_min + int(split * (songs_per_pathology // 10)) - 1
        else:
            range_max = range_min + int(split * songs_per_pathology // k) - 1
        val_files_index = np.zeros((no_of_pathologies, no_of_songs_for_val_per_pathology))
        for i in range(no_of_pathologies):
            val_files_index[i] = i * int(split * songs_per_pathology) + np.linspace(range_min, range_max, range_max - range_min + 1)
        raw_dataset = tf.data.TFRecordDataset(os.path.join('..', 'TFRecord', 'Train_Features.tfrecord'))
        parsed_dataset = raw_dataset.map(parse_and_decode_function)
        train_dataset = []
        val_dataset = []
        batch_dataset = parsed_dataset.batch(windows_per_song)
        last_index = 0
        for line in val_files_index:
            if train_dataset == []:
                train_dataset = batch_dataset.take(line[0] - last_index)
            else:
                train_dataset = train_dataset.concatenate(batch_dataset.take(line[0] - last_index))
            batch_dataset = batch_dataset.skip(line[0] - last_index)
            last_index = line[-1] + 1
            if val_dataset == []:
                val_dataset = batch_dataset.take(line[-1] - line[0] + 1)
            else:
                val_dataset = val_dataset.concatenate(batch_dataset.take(line[-1] - line[0] + 1))
            batch_dataset = batch_dataset.skip(line[-1] - line[0] + 1)
        train_dataset = train_dataset.concatenate(batch_dataset.take(-1))  # take the remaining tensors
        return (train_dataset.unbatch(), val_dataset.unbatch())

    else:
        pathology_labels = {}
        classes = get_class_list(path)
        for i in range(len(classes)):
            pathology_labels[classes[i]] = i

        pathology_folders = sorted(os.listdir(path))
        x_train, x_val, y_train, y_val = [], [], [], []
        for pathology in pathology_folders:
            files = sorted(os.listdir(os.path.join(path, pathology)))
            random.shuffle(files)
            if k == 1:
                nr_of_files_to_load = len(files) // 5  #one speaker val hardcoded
                #todo un-hardcode this //5, make it user-specified^
            else:
                nr_of_files_to_load = len(files) // k
            range_min = (fold_nr - 1) * nr_of_files_to_load
            range_max = range_min + nr_of_files_to_load
            eval_files = files[range_min:range_max]
            for file in files:
                npy = np.load(os.path.join(path, pathology, file))
                if file in eval_files:
                    x_val.extend(npy)
                    y_val.extend(pathology_labels[pathology] * np.ones(npy.shape[0]))   #count no_windows and add labels to all of them
                else:
                    x_train.extend(npy)
                    y_train.extend(pathology_labels[pathology] * np.ones(npy.shape[0]))
        y_train = to_categorical(np.asanyarray(y_train), num_classes=len(classes))
        y_val = to_categorical(np.asanyarray(y_val), num_classes=len(classes))
        return (np.asanyarray(x_train), np.asanyarray(x_val),
                np.asanyarray(y_train), np.asanyarray(y_val))
