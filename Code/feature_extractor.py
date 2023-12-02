from utils import *

class FeatureExtractor:
    def __init__(self, location='Features_functions', extension='py', feature_base_name='feature'):
        """The constructor should set the location and extension of the features, and load all the features."""
        self.__location = str(location)
        self.__extension = str(extension)
        self.signal = None
        self.__feature_base_name = str(feature_base_name)
        self.__load_features()


    def __get_feature_name(self, feature_id):
        # print("in get feature name feature_id = ", feature_id)
        return self.__feature_base_name + '_' + feature_id

    def __set_params(self, signal, features_dict=None):
        '''This method should set the signal and all the parameters to be used for extracting features, e.g. self.n_fft = features_dict['n_fft]'''
        setattr(self, 'signal', np.asanyarray(signal, dtype=np.float64))
        for key, value in features_dict.items():
            setattr(self, key, value)
        # print("self.n_lvls = ", self.n_lvls)
        # print("features_dict = ", features_dict)

    def __load_features(self):
        '''This method should load all the existing features and save them as atributes of this object, e.g. self.fft'''
        for file in os.listdir(self.__location):
            if not file.endswith("." + self.__extension):
               continue
            # print("in load features file = ", file)
            function = open(os.path.join(self.__location, file), 'r').read()
            dic = {}
            exec(function, None, dic)
            reference = list(dic.values())[0]
            feature_name = self.__get_feature_name(file.split('.')[0])
            # print("feature_name = ", feature_name)
            # print("reference = ", reference)
            setattr(self, feature_name, reference)

    def __extract_feature_by_id(self, id):
        '''This method should extract only the feature with the given unique id'''
        feature_id = self.__get_feature_name(str(id))
        if hasattr(self, feature_id):
            try:
                return getattr(self, feature_id)(self)
            except:
                self.__load_features()
                try:
                    return getattr(self, feature_id)(self)
                except:
                    raise Exception('Attribute ' + feature_id + ' not found after reloading.')
        else:
            self.__load_features()
            try:
                return getattr(self, feature_id)(self)
            except:
                raise Exception('Attribute ' + feature_id + ' not found after reloading.')


    def extract_features(self, signal, feature_list, features_dict, variance_type='var', raw_features = True, keep_feature_dims = False):
        '''This method should return a list of features for a given signal. If the signal is 1D (a window), the mean and variance should be extracted from each feature.'''
        #returns the list of features or matrix of features if raw dims is set. e.g. mfcc = matrix & you output the matrix as such

        features = []
        self.__set_params(signal, features_dict)
        for id in feature_list:

            if raw_features:   #extracts features only without calculating mean & var.
                feature = self.__extract_feature_by_id(id)
                try:
                    fshapelen = (len(feature.shape) != 2)
                except:
                    fshapelen = True    # if there is no shape it's different from 2
                if fshapelen:   #shape is not 2 i.e. not mfcc or something
                    try:
                        features.extend(feature)  # if feature is an iterable
                    except:
                        features.append(feature)  # if feature is not an iterable i.e. tempo
                if id == 'mfcc' and not keep_feature_dims:
                    feature = np.mean(feature, axis=-1)  #todo sau fa-le flatten idk -> reduces time dimension
                    # feature = np.flatten(feature)
                    # feature.T.flatten()
                    # feature = np.mean(feature, axis=0)  #-> reduces MFCC dimension
                    # feature = feature[0]
                try:
                    features.extend(feature)  # if feature is an iterable
                except:
                    features.append(feature)  # if feature is not an iterable
            if not raw_features:
                if len(signal.shape) == 1:
                    feature = self.__extract_feature_by_id(id)
                    if id == 'tempo':
                        features.append(float(feature))
                    # else:
                    #     try:
                    #         features.extend(feature)    #if feature is an iterable
                    #     except:
                    #         features.append(feature)    #if feature is not an iterable
                    elif id == 'mfcc':
                        features.extend(np.mean(feature, axis=1))
                        if variance_type == 'var':
                            features.extend(np.var(feature, axis=1) ** 2)
                        else:
                            features.extend(scipy.stats.median_absolute_deviation(feature, axis=1) ** 2)
                    else:
                        features.append(np.mean(feature))
                        if variance_type == 'var':
                            features.append(np.var(feature))
                        else:
                            features.append(squared_median_abs_dev(feature))
            if len(signal.shape) == 2:
                feature = self.__extract_feature_by_id(id)
                features.extend(feature.tolist())

        return features
