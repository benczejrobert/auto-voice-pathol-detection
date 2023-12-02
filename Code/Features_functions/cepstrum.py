def cepstrum(self):
    C = utils_cepstrum(self.signal, self.n_fft)
    try:
        retme = C[:, 0:C.shape[-1] // 2 + 1]
    except:
        retme = C[0:C.shape[-1] // 2 + 1]
    return retme