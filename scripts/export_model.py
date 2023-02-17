import os
import sys
import optparse
import tarfile
import yaml

sys.path.append('')
from cube.networks.cubegan import Cubegan
from cube.io_utils.io_cubegan import CubeganEncodings


def _export_model(params):
    tar = tarfile.open('{0}.tar.gz'.format(params.output_model), 'w:gz')
    base_path = params.input_model
    # remove unnecessary network modules for a smaller model
    sys.stdout.write('Loading model and removing discriminator... ')
    sys.stdout.flush()
    encodings = CubeganEncodings('{0}.encodings'.format(params.input_model))
    conf = yaml.load(open('{0}.yaml'.format(params.input_model)), yaml.Loader)
    cond_type = conf['conditioning']
    model = Cubegan(encodings, conditioning=cond_type, train=True)
    del model._mpd
    del model._msd
    model.save('{0}.model'.format(params.input_model))
    sys.stdout.write('done\n')
    sys.stdout.write('Creating archive...\n')
    in_file_list = ['{0}.{1}'.format(base_path, ext) for ext in ['model', 'yaml', 'encodings']]
    out_file_list = ['cubegan.{0}'.format(ext) for ext in ['model', 'yaml', 'encodings']]
    for in_file, out_file in zip(in_file_list, out_file_list):
        sys.stdout.write('\t{0}\n'.format(in_file))
        tar.add(in_file, out_file)
    tar.close()
    sys.stdout.write('Splitting the model into multiple volumes...')
    sys.stdout.flush()
    f_in = open('{0}.tar.gz'.format(params.output_model), 'rb')
    CHUNK_SIZE = 49 * 1024 * 1024
    counter = 0
    while True:
        chunk = f_in.read(CHUNK_SIZE)
        if not chunk:
            break
        f_out = open('{0}-{1:02d}'.format(params.output_model, counter), 'wb')
        counter += 1
        sys.stdout.write(' {0}'.format(counter))
        sys.stdout.flush()
        f_out.write(chunk)
        f_out.close()

    sys.stdout.write(' done\n')
    os.unlink('{0}.tar.gz'.format(params.output_model))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-model', action='store', dest='input_model',
                      help='Cleanup temporary training files and start from fresh')
    parser.add_option('--output-model', action='store', dest='output_model',
                      help='Location of the training files')

    (params, _) = parser.parse_args(sys.argv)

    _export_model(params)
