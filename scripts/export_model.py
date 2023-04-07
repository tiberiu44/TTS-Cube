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
    model.load('{0}.last'.format(params.input_model))
    del model._mpd
    del model._msd
    if hasattr(model, '_dummy'):
        del model._dummy
    model.save('{0}.model'.format(params.input_model))
    sys.stdout.write('done\n')
    sys.stdout.write('Creating archive...\n')
    in_file_list = ['{0}.{1}'.format(base_path, ext) for ext in ['model', 'yaml', 'encodings']]
    out_file_list = ['cubegan.{0}'.format(ext) for ext in ['model', 'yaml', 'encodings']]
    for in_file, out_file in zip(in_file_list, out_file_list):
        sys.stdout.write('\t{0}\n'.format(in_file))
        tar.add(in_file, out_file)

    in_file_list = ['{0}.{1}'.format(params.input_phonemizer, ext) for ext in ['sacc.best', 'encodings']]
    out_file_list = ['phonemizer.{0}'.format(ext) for ext in ['model', 'encodings']]
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
    model_desc = {'version': params.version,
                  'phonemizer': 'sentence',
                  'synthesis': 'cubegan',
                  'language': params.language,
                  'description': params.description}
    yaml.safe_dump(model_desc, open('{0}.yaml'.format(params.output_model), 'w'))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-model', action='store', dest='input_model',
                      help='What model to export (should be a cubegan)')
    parser.add_option('--input-phonemizer', action='store', dest='input_phonemizer',
                      help='What phonemizer to export with the model')
    parser.add_option('--output-model', action='store', dest='output_model',
                      help='Location of the training files')
    parser.add_option('--version', dest='version', default='1.0.0',
                      help='What version to set for the exported model')
    parser.add_option('--language', dest='language', default='multi',
                      help='What is the LC for this model (use 2 letter LC - you can pass a comma separated list)')
    parser.add_option('--description', dest='description', default='',
                      help='Short description of the model')

    (params, _) = parser.parse_args(sys.argv)

    _export_model(params)
