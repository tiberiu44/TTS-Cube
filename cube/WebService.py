import json
from flask import Flask, request, send_file
import os
from synthesis import load_encoder, load_vocoder, synthesize_text, write_signal_to_file
import dynet_config


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


params = {'mgc_order': 80, 'temperature': 0.7, 'target_sample_rate': 24000, 'learning_rate': 0.0001}
params = dotdict(params)

encoders = {}
vocoders = {}

def load_all_encoders(base_path):
    global encoders

    for file in os.listdir(base_path):
        new_file = '%s/%s' % (base_path, file)
        if os.path.isdir(new_file):
            encoders[file] = load_encoder(params, new_file)


def load_all_vocoders(base_path):
    global vocoders

    for file in os.listdir(base_path):
        new_file = '%s/%s' % (base_path, file)
        if os.path.isdir(new_file):
            vocoders[file] = load_vocoder(params, new_file)


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

    signal = synthesize_text(text, encoders[language], vocoders[language], speaker_identity)
    write_signal_to_file(signal, out_file, params)

    return send_file('../%s' % out_file, mimetype='audio/wav')


if __name__ == '__main__':
    dynet_config.set(mem=2048, random_seed=9)

    model_path = 'data/models'
    load_all_encoders(model_path)
    load_all_vocoders(model_path)

    app.run(host='0.0.0.0', port=8080)
