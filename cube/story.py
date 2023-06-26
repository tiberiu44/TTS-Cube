from cube.api import TTSCube
import librosa
import numpy as np
from pathlib import Path


class StoryCube:
    def __init__(self, base_model):
        self._cube = TTSCube.load(base_model)
        self._default_music, _ = librosa.load('{0}/.ttscube/models/{1}/music.wav'.format(str(Path.home()), base_model),
                                              sr=24000)

    def __call__(self, text, speaker=None, background_music_path: str = None):

        parts = text.split('\n\n')
        buffer = [0 for _ in range(24000 * 5)]

        metadata = [
            {
                'name': 'intro',
                'start': 0,
                'end:': 5,
                'text': ''
            }
        ]
        start = 5
        for part in parts:
            print(part)
            audio = self._cube(part, speaker=speaker)
            print("audio:", len(audio))
            for x in audio:
                buffer.append(x)
            for _ in range(24000):
                buffer.append(0)
            metadata.append({
                'name': 'paragraph',
                'text': part,
                'start': start,
                'end': start + (len(audio) / 24000) + 1  # + 24000
            })
            start += (len(audio) / 24000) + 1
        music = self._default_music
        if background_music_path is not None:
            music, _ = librosa.load(background_music_path, sr=24000)
        # add 5 more seconds of music
        for ii in range(24000 * 5):
            buffer.append(0)

        # add music
        for ii in range(len(buffer)):
            buffer[ii] = (music[ii % len(music)] * 0.30) * 32700 + buffer[ii]
        buffer = np.array(buffer, dtype='int16')
        return {
            'audio': buffer,
            'meta': metadata
        }
