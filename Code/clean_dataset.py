from utils import *
from imports import *


'''

In the introduction of SVD:
Depending on the quality of the speech signal, not all vowels in all forms were occasionally found in the recordings of the vowels. 
This means that some recording sessions do not have all the files.

=> todo function that ensures speakers contain all the vowels+intonations(?)
-> was not necessary because this excrept didn't suffer from this drawback

'''

def clean_dataset(db_path, balance_classes = True, win_length = None, vowels = None, intonations = None, EGG=''):
    # todo this function can not yet ensure equal gender distribution. might want to do this as well
    '''

        Deletes all the recordings less than win_length(defaults to 1 s) samples from db_path then deletes the speakers from all classes of db_path
        that contain more speakers than the no_spearkers of the lowest cardinality class.

    '''

    #gets dir list
    dirs = get_class_list(db_path)

    if balance_classes:
        # clean: delete short files in every dir (shorter than the requested time window)
        for dir in dirs:
            print("in balance_classes, get_short_files dir = ", dir)
            cleanme = os.path.join(db_path,dir)
            deletlist = get_short_files(cleanme, win_length)
            for deletme in deletlist:
                if os.path.exists(deletme):
                    no_deleted = delete_speaker(deletme,EGG,True)

        # get min folder: gets the pathology with least no of files
        min_no_files, minfiles_path = get_min_files(db_path)
        # deletes speakers from big folders to match no_speakers in min_folder
        for dir in dirs:
            cleanme = os.path.join(db_path,dir)
            if cleanme == minfiles_path:
                continue    #continue because deletion was already performed
            #gets the files list to delete from
            for subdir, dirs, files in sorted(os.walk(cleanme)):
                break
            while len(files)>min_no_files:
                no_del = delete_speaker(os.path.join(cleanme,files[-1]),EGG)
                # files.pop()
                del files[-no_del:] #remove the deleted files from the list

    # possible vowels = ['-a', '-i', '-u']
    # possible intonations = ['_h', '_l', '_n']

    # if len(keep_vowels_intonations):
    if vowels != None and intonations != None: #if no vowels or intonations
        # have been chosen to be kept then dataset remains unchanged

        # fix the lists
        for v in range(len(vowels)):
            if vowels[v][0] != '-':
                vowels[v] = '-' + vowels[v][0]
        for i in range(len(intonations)):
            if intonations[i][0] != '_':
                intonations[i] = '_' + intonations[i][0]
        print("vowels, intonations = ", vowels, intonations)
        keepus = []
        for subdir, dirs, files in sorted(os.walk(db_path)):
            for file in files:
                voicerec = os.path.join(subdir, file)

                vis = []
                for v in vowels:
                    for i in intonations:
                        vis.append(v+i)
                for vi in vis:
                    # print("voicerec split = ", voicerec.split('.')[-2].split('-')[-1])

                    if vi == '-'+voicerec.split('.')[-2].split('-')[-1]: # if the voice recording does not contain the given vi then delete it. #todo THIS WORKS ONLY IF VOWELS&INTONATIONS CONTAIN 1 COMBO
                        # print("voicerec = ", voicerec)
                        if os.path.exists(voicerec):
                            keepus.append(voicerec)
                            # os.remove(voicerec)

        for subdir, dirs, files in sorted(os.walk(db_path)):
            for file in files:
                voicerec = os.path.join(subdir, file)
                if voicerec not in keepus:
                    os.remove(voicerec)