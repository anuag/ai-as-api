""" aiModelTest.py """

import pathlib
import json

from api.AIModel import AIModel


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'training' / 'trained_model'
SPAM_HD_PATH = MODEL_DIR / 'spam_model.hdf5'
SPAM_TOKENIZER_PATH = MODEL_DIR / 'spam_tokenizer.json'
SPAM_METADATA_PATH = MODEL_DIR / 'spam_metadata.json'


if __name__ == '__main__':
    spamClassifier = AIModel(
        modelPath=SPAM_HD_PATH,
        tokenizerPath=SPAM_TOKENIZER_PATH,
        metadataPath=SPAM_METADATA_PATH,
    )
    print(str(spamClassifier))
    #
    sampleTexts = [
        'This is a nice touch, adding a sense of belonging and coziness. Thank you so much.',
        'Click here to WIN A FREE IPHONE and this and that.',
    ]
    result = spamClassifier.predict(sampleTexts)
    #
    print(result)
    #
    print(json.dumps(result))
