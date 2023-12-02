#https://dtcwt.readthedocs.io/en/0.12.0/
# actually performs dtcwt, not dtcwpt
# https://journals.utm.my/jurnalteknologi/article/view/14748/7551
# https://iopscience.iop.org/article/10.1088/1742-6596/1372/1/012029/meta

def dtcwpt(self):

    '''
    Perform a *n*-level DTCWT decompostion on a 1D column vector *X* (or on
        the columns of a matrix *X*).

    vecs_t pyramid attributes:

    .. py:attribute:: lowpass

        A NumPy-compatible array containing the coarsest scale lowpass signal.

    .. py:attribute:: highpasses

        A tuple where each element is the complex subband coefficients for
        corresponding scales finest to coarsest.

    .. py:attribute:: scales

        *(optional)* A tuple where each element is a NumPy-compatible array
        containing the lowpass signal for corresponding scales finest to
        coarsest. This is not required for the inverse and may be *None*.


    '''

    transform = dtcwt.Transform1d() #todo maybe make this output row vector not col vect.
    #todo vezi cum faci cu n_levels ca sa poti parametriza scula
    vecs_t = transform.forward(self.signal, nlevels=self.n_lvls)

    # asdf
    # print("dtcwpt")
    # print("self.signal.shape = ", self.signal.shape)
    # print("lowpasas shape vecs_t = ", (vecs_t.lowpass[1].shape)) #-> complecsi, ndarray
    # print("lowpass shape vecs_t = ", (vecs_t.lowpass.shape)) #-> complecsi, ndarray
    # print("highpasses0 shape vecs_t = ", (vecs_t.highpasses[0].shape)) #-> complecsi, ndarray
    # print("highpasses1 shape vecs_t = ", (vecs_t.highpasses[1].shape)) #-> complecsi, ndarray
    # print("highpasses2 shape vecs_t = ", (vecs_t.highpasses[2].shape)) #-> complecsi, ndarray
    # print("highpasses type vecs_t = ", np.array(vecs_t.highpasses).shape) #-> original tuplu de 5 np.arrays de dim (x,1), (x*2^(-k),1), cu k pana la 5.

    #TODO lookup howto process these features
    # for coherent results -> extract energy from the lowpass & highpasses then concatenate

    #features need shape of (len,) or -> tolist

    #medii si variante din abs-uri? dar din faze?
    return np.reshape(vecs_t.lowpass,(vecs_t.lowpass.shape[0],)).tolist()
    return vecs_t