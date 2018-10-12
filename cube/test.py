import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def _normalize(mgc, mean, stdev):
    for x in xrange(mgc.shape[0]):
        mgc[x] = (mgc[x] - mean) / stdev
    return mgc


if __name__ == '__main__':
    # import dynet
    import cube_runtime
    from io_modules.dataset import DatasetIO
    import numpy as np

    cube_runtime.print_version()
    cube_runtime.load_vocoder('../data/models/rnn_vocoder_sparse')
    mgc = np.load('../test.mgc.npy')
    #mgc=np.zeros((390, 60), dtype=np.double)
    mgc = mgc.copy(order='C')
    x=cube_runtime.vocode(mgc, 0.8)
    dio = DatasetIO()
    #zz=
    #enc = dio.b16_to_float(x, discreete=True)
    enc=np.array(x, dtype='int16')

    output_file = 'test.wav'
    dio.write_wave(output_file, enc, 16000, dtype=np.int16)
    print (x);
