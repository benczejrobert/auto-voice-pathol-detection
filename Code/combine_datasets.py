from utils import *

# iterate the 2 folders

def combine_datasets(db_path, db_path_egg, featspath, tfrecord, writer, overwrite):
    """
        If tfrecord is set to True:
        	The script adds the requested features in the Feature.tfrecord file
        Else:
        	The script creates (or overwrites) a voicerec.npy file which contains the concatenation of the requested
        	features extracted from the selected .wav and .egg files.

        Arguments:
            - db_path [string], relative path to the Features folder containing the .npy files of the .wav recordings
            - db_path_egg [string], relative path to the Features folder containing the .npy files of the .egg recordings
            - featspath [string], relative path of the Features folder to save the combined features to
            - pathology [string], name of the pathology
                                    must be one of the pathologies in the database!
            - voicerec [string], name of the voice recording
            - W_features [2D-array], array of features to be writen, size #windows x #features
            - tfrecord [bool], if true saves W_features in the '.tfrecord' file
                               else it saves it as voicerec.npy file.
            - writer [tf.io.TFRecordWriter], the writer used to write records to a .tfrecord file.
        """

    for pack in zip(sorted(os.walk(db_path)), sorted(os.walk(db_path_egg))):
        (subdir, dirs, files) = pack[0]
        (subdir_egg, dirs_egg, files_egg) = pack[1]
        i=0
        for filepack in zip(sorted(files),sorted(files_egg)):
            (file,file_egg) = filepack
            i+=1
            voicefeat = os.path.join(subdir, file)
            eggfeat = os.path.join(subdir_egg, file_egg)
            pathology = os.path.split(subdir)[1]
            # print("voicefeat =",voicefeat)
            # print("voicefeat =",eggfeat)
            npy_vox = np.load(voicefeat)
            npy_egg = np.load(eggfeat)
            combined_npy = np.concatenate([npy_vox,npy_egg], axis = 2)
            save_features(featspath, pathology, file, combined_npy, tfrecord, writer, db_path, overwrite)

db_path = os.path.join('..', 'Features')
db_path_egg = os.path.join('..', 'Features-egg')
featspath = os.path.join('..', 'Features-combined')
combine_datasets(db_path, db_path_egg, featspath, False, None, False)
