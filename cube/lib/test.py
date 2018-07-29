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
    cube_runtime.load_vocoder('../data/models/rnn')
    mean = np.load('../data/models/mean.npy')
    stdev = np.load('../data/models/stdev.npy')
    mgc = np.load('../data/processed/dev/anca_dcnews_0952.mgc.npy')
    mgc = _normalize(mgc, mean, stdev)
    mgc = mgc.copy(order='C')
    cube_runtime.vocode(mgc, stdev, mean, 0.8)
