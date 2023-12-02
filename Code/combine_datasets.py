from utils import *

# iterate the 2 folders

#todo norm per time bin -> parcurg db-ul, aflu cate 26 max-uri prin concatenarea matricilor una langa alta
# si apoi impart fiecare coloana a matricelor la coloana de 26 max-uri.
# apoi bag eventual batch norm la retea si compar rezultatele
# https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739

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
        #todo skip 1st iter

        # print(subdir, dirs)
        # print(subdir_egg, dirs_egg)
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
            # print("npy_vox.shape =",npy_vox.shape)
            # print("npy_egg.shape =",npy_egg.shape)
            # combined_npy = np.concatenate([npy_vox,npy_egg], axis = 0) #sau axis -1 nuj, crek am axis 0,1,2. vreau ca fiecare matrice sa contina mfcc-ii celor 2 matrice initiale.
            # print("axis = 0", combined_npy.shape)   #todo axis 0 nu vrem
            # combined_npy = np.concatenate([npy_vox,npy_egg], axis = 1) #sau axis -1 nuj, crek am axis 0,1,2. vreau ca fiecare matrice sa contina mfcc-ii celor 2 matrice initiale.
            # print("axis = 1", combined_npy.shape) #todo advised "una sub alta"
            combined_npy = np.concatenate([npy_vox,npy_egg], axis = 2) #sau axis -1 nuj, crek am axis 0,1,2. vreau ca fiecare matrice sa contina mfcc-ii celor 2 matrice initiale.

            print("axis = 2", combined_npy.shape) #todo - try and see
            asdf
            #todo featspath should look like "folder_name-combined"
            save_features(featspath, pathology, file, combined_npy, tfrecord, writer, db_path, overwrite)

db_path = os.path.join('..', 'Features')
db_path_egg = os.path.join('..', 'Features-egg')
featspath = os.path.join('..', 'Features-combined')
combine_datasets(db_path, db_path_egg, featspath, False, None, False)
