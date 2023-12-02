from utils import *


def split_dataset(split_path, split_perc, features_cardinality, tfrecord=False, perspeaker = False):
    '''
    Script that splits the dataset in Train and Test features.

    Arguments:
        - path [string], relative path to the 'Features' folder
        - split_perc [int], percentage used to split Features in (split_perc) Train
        			and (1 - split_perc) Test
        - tfrecord: a boolean that indicates to split the .tfrecord file.

        - features_cardinality: the length of the Features TFRecordDataset.

    Outputs (if tfrecord = True):
        - train_cardinality.txt: Text file containing the length of the Train TFRecordDataset.
        - test_cardinality.txt: Text file containing the length of the Test TFRecordDataset.
    '''
    ordinal = ['st','nd','rd']
    # pathologies = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop',
    #           'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

    pathologies = load_classlist()

    if not tfrecord:
        test_path = os.path.join(''.join(split_path.split(os.sep)[0:-1]), 'Test')
        train_path = os.path.join(''.join(split_path.split(os.sep)[0:-1]), 'Train')
        for t_path in [test_path, train_path]:
            if os.path.exists(t_path):
                rmtree(t_path)  # delete Test/Train folder if it exists
            os.mkdir(t_path)  # creates Test/Train folder and subfolder for each pathology
            [os.mkdir(os.path.join(t_path, folder)) for folder in pathologies]

        for subdir, dirs, files in os.walk(split_path):
            if subdir == split_path:
                print("in split_dataset subdir = ", subdir)
                continue  # skip first iteration (parent folder), get to subfolders

            no_of_test_files = len(files) - int(split_perc / 100 * len(files))

            if not perspeaker:
                test_file_indices = np.random.choice(np.linspace(0, len(files) - 1, len(files), dtype='int'),
                                                 replace=False, size=no_of_test_files)  # make a list of indices
            else:
                files = sorted(files)
                no_speakers = count_speakers(subdir)
                test_speaker_indices = np.random.choice(np.linspace(0, no_speakers - 1, no_speakers, dtype='int'),
                                                     replace=False, size=no_speakers - int(no_speakers*(split_perc/100)))  # make a list of indices
                files_per_speaker = len(files)//no_speakers     #assuming every speaker contains the same number of voicerecs
                test_file_indices = []
                for i in test_speaker_indices:
                    test_file_indices.extend(np.asanyarray(range(i*files_per_speaker,i*files_per_speaker+files_per_speaker)).tolist())

            dst_test = os.path.join(test_path, subdir.split(os.sep)[-1])
            dst_train = os.path.join(train_path, subdir.split(os.sep)[-1])
            i = 0
            for file in sorted(files):
                if i%10<3 and i not in [10,11,12]:
                    print("split_dataset reached ",i+1, ordinal[i%10]+" file of "+ subdir)
                else:
                    print("split_dataset reached ",i+1,"th file of "+subdir)
                # if int(file.split('.')[-2]) in test_file_indices:
                if int(i) in test_file_indices:
                    copy(os.path.join(subdir, file), dst_test)
                else:
                    copy(os.path.join(subdir, file), dst_train)
                i+=1
    else:
        if not os.path.exists('..' + os.sep + 'TFRecord'):
            os.mkdir('..' + os.sep + 'TFRecord')  # create the TFrecord folder if it does not exist
        # initialize the writers for the test&train TFRecords
        trainwriter = tf.io.TFRecordWriter('..' + os.sep + 'TFRecord' + os.sep + 'Train_Features.tfrecord')
        testwriter = tf.io.TFRecordWriter('..' + os.sep + 'TFRecord' + os.sep + 'Test_Features.tfrecord')
        # open the feature TFRecord that will be split into test and train
        raw_dataset = tf.data.TFRecordDataset(os.path.join('..', 'TFRecord', 'Features.tfrecord'))
        parsed_dataset = raw_dataset.map(parse_window_function)

        window_counter = 0
        voicerecs_per_pathology = len(os.listdir('..' + os.sep + 'GTZAN' + os.sep + 'Blues'))
        no_of_pathologies = len(os.listdir('..' + os.sep + 'GTZAN'))
        no_of_voicerecs = voicerecs_per_pathology * no_of_pathologies
        windows_per_voicerec = features_cardinality // no_of_voicerecs
        no_of_test_files = voicerecs_per_pathology - int((split_perc / 100) * voicerecs_per_pathology)

        train_cardinality, test_cardinality = 0, 0  # will count the train&test window numbers
        windows_per_pathology = windows_per_voicerec*voicerecs_per_pathology

        for window_data in parsed_dataset:
            window_data['window'] = bytes_feature(window_data['window'])
            window_data['label'] = bytes_feature(window_data['label'] )
            ftfeature = tf.train.Features(feature=window_data)
            ftexample = tf.train.Example(features = ftfeature)
            ftserialized = ftexample.SerializeToString()

            voicerec_counter = window_counter // windows_per_voicerec
            voicerec_in_pathology = voicerec_counter % voicerecs_per_pathology

            # when the pathology changes randomize the voicerecs that will be added to the test dataset from every pathology
            if window_counter%windows_per_pathology ==0:
                test_file_indices = np.random.choice(np.linspace(0, voicerecs_per_pathology - 1, voicerecs_per_pathology, dtype='int'),
                                                 replace=False, size=no_of_test_files)

            if voicerec_in_pathology in test_file_indices:
                testwriter.write(ftserialized)
                test_cardinality += 1
            else:
                trainwriter.write(ftserialized)
                train_cardinality += 1
            window_counter += 1

        if not os.path.exists(os.path.join('..', 'Cardinality')):
            os.mkdir(os.path.join('..', 'Cardinality'))
        path_train = os.path.join('..', 'Cardinality', 'Train_Features.txt')
        path_test = os.path.join('..', 'Cardinality', 'Test_Features.txt')
        write_cardinality(path_train, train_cardinality)
        write_cardinality(path_test, test_cardinality)




