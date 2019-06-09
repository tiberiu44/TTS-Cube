import json
from flask import Flask, request, send_file
import os
from synthesis import load_encoder, load_vocoder, synthesize_text, write_signal_to_file
import dynet_config
import sys
import optparse
from os.path import exists

encoders = {}
vocoders = {}


def load_all_models(base_path):
    global encoders, vocoders

    for lang in os.listdir(base_path):
        new_path = '%s/%s' % (base_path, lang)
        if os.path.isdir(new_path):
            try:
                encoder = load_encoder(params, new_path)
                vocoder = load_vocoder(params, new_path)

                encoders[lang] = encoder
                vocoders[lang] = vocoder
            except:
                print('Language %s does not have all models' % lang)


app = Flask(__name__)


@app.route('/healthcheck', methods=['GET'])
def health_check():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/synthesis', methods=['GET'])
def get_wav():
    out_file = 'out.wav'
    data = json.loads(request.data.decode('utf-8'), encoding='utf-8')

    try:
        language = data['language']
    except:
        return json.dumps({'error': 'language not set'}), 400, {'ContentType': 'application/json'}

    if language not in encoders:
        return json.dumps({'error': 'language not found'}), 400, {'ContentType': 'application/json'}

    try:
        text = data['text']
    except:
        return json.dumps({'error': 'text not set'}), 400, {'ContentType': 'application/json'}

    try:
        speaker_identity = data['speaker']
    except:
        return json.dumps({'error': 'speaker not set'}), 400, {'ContentType': 'application/json'}

    print(language, text, speaker_identity)
    global g2p

    signal = synthesize_text(text, encoders[language], vocoders[language], speaker_identity, g2p=g2p)
    write_signal_to_file(signal, out_file, params)

    return send_file('../%s' % out_file, mimetype='audio/wav')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--host', action='store', dest='host', default='0.0.0.0',
                      help='Host IP of the WebService (default="0.0.0.0")')
    parser.add_option('--port', action='store', dest='port', type='int', default=8080,
                      help='Port IP of the WebService (default=8080)')
    parser.add_option('--learning_rate', action='store', dest='learning_rate', type='float',
                      help='Learning rate; Used for compatibility issues (default=0.0001)', default=0.0001)
    parser.add_option('--mgc-order', action='store', dest='mgc_order', type='int',
                      help='Order of MGC parameters (default=80)', default=80)
    parser.add_option('--temperature', action='store', dest='temperature', type='float',
                      help='Exploration parameter (max 1.0, default 0.7)', default=0.7)
    parser.add_option('--target-sample-rate', action='store', dest='target_sample_rate',
                      help='Resample input files at this rate (default=24000)', type='int', default=24000)
    parser.add_option("--set-mem", action='store', dest='memory', default='2048', type='int',
                      help='preallocate memory for batch training (default 2048)')
    parser.add_option('--g2p-model', dest='g2p', action='store',
                      help='Use this G2P model for processing')
    params, _ = parser.parse_args(sys.argv)

    dynet_config.set(mem=params.memory, random_seed=9)

    models_base_path = 'data/models'
    load_all_models(models_base_path)
    if params.g2p is not None:
        from models.g2p import G2P
        from io_modules.encodings import Encodings

        g2p_encodings = Encodings()
        g2p_encodings.load(params.g2p + '.encodings')
        g2p = G2P(g2p_encodings)
        g2p.load(params.g2p + '-bestAcc.network')
        if exists(params.g2p + '.lexicon'):
            g2p.load_lexicon(params.g2p + '.lexicon')
    else:
        g2p = None

    app.run(host=params.host, port=params.port)
