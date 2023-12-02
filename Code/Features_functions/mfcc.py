def mfcc(self):
    retme = librosa.feature.mfcc(S=librosa.power_to_db(self.mel_spect), n_mfcc=self.n_mfcc)
    # print(retme.shape)
    return retme
#todo what if no self.mel_spect?