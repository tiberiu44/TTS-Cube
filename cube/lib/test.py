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
    import numpy as np

    cube_runtime.print_version()
    cube_runtime.load_vocoder('../data/models/rnn_vocoder')
    mean = np.load('../data/models/mean.npy')
    stdev = np.load('../data/models/stdev.npy')
    mgc = np.load('../data/processed/dev/anca_dcnews_0127.orig.mgc.npy')
    #mgc=np.zeros((390, 60), dtype=np.double)
    mgc = _normalize(mgc, mean, stdev)
    mgc = mgc.copy(order='C')
    x=cube_runtime.vocode(mgc, stdev, mean, 0.8)
    from io_modules.dataset import DatasetIO
    dio = DatasetIO()
    enc = dio.b16_to_float(x, discreete=True)
    output_file = 'test.wav'
    dio.write_wave(output_file, enc, 16000)
    print x;
