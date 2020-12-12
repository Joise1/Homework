import hashlib
import json
import os

import stanza


def get_md5(path):
    """
    Get the MD5 value of a path.
    """
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()


if __name__ == '__main__':
    # stanza.download('en')  # This downloads the English models for the neural pipeline
    nlp = stanza.Pipeline('en')  # This sets up a default neural pipeline in English

    # get plain text
    input_dir = 'data/SW100'
    for dirpath, dirs, files in os.walk(input_dir):
        for f in files:
            txt_file = os.path.join(dirpath, f)
            if not txt_file.endswith('.txt'):
                continue
            if not os.path.exists(txt_file[:-4] + '.ann'):
                continue
            with open(txt_file, encoding='utf-8') as f:
                sentences = f.read()
                doc = nlp(sentences)
                dict = doc.to_dict()
                with open(txt_file+'.json', "w") as f:
                    json.dump(dict, f)