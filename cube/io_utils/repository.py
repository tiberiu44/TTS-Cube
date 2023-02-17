import requests
import string
import tarfile
import os
from tqdm import tqdm


def _download_file(url: str, filename: str, description=None):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Error getting {url}, received status_code {r.status_code}")
    file_size = int(r.headers['Content-Length'])
    chunk_size = 1024

    with open(filename, 'wb') as fp:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=description, unit_divisor=1024,
                  disable=True if description is None else False, leave=False) as progressbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk is not None:
                    fp.write(chunk)
                    fp.flush()
                    progressbar.update(len(chunk))

    return r.status_code


def download_model(local_path, model):
    download_base = "https://github.com/tiberiu44/TTS-Cube-Models/raw/main/models/{0}/model".format(model)
    file_base = 'model'
    terminations = ['{0:02d}'.format(ii) for ii in range(20)]
    file_list = []
    for t in terminations:
        download_url = '{0}-{1}'.format(download_base, t)
        target_file = str(os.path.join(local_path, file_base))
        target_file = '{0}-{1}'.format(target_file, t)
        try:
            if _download_file(download_url, target_file, description='Part {0}'.format(t)) != 200:
                break
        except:
            break
        file_list.append(target_file)

    target_file = os.path.join(local_path, file_base)

    f_out = open(target_file, 'wb')
    for file in file_list:
        f_in = open(file, 'rb')
        while True:
            buffer = f_in.read(1024 * 1024)
            if not buffer:
                break
            f_out.write(buffer)
    f_out.close()

    tar = tarfile.open(target_file, 'r:gz')
    tar.extractall(local_path)
    tar.close()

    for file in file_list:
        os.unlink(file)
    os.unlink(target_file)
