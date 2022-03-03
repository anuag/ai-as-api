""" loadTestModel.py
    
        Check that one can start predicting with just the files
        found in the "trained model" directory.
"""

import sys

import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import models

# in
trainedModelFile = 'training/trained_model_v1/clickbait_model.h5'
trainedMetadataFile = 'training/trained_model_v1/clickbait_metadata.json'
trainedTokenizerFile = 'training/trained_model_v1/clickbait_tokenizer.json'


if __name__ == '__main__':
    # Load tokenizer and metadata:
    #   (in metadata, we'll need keys 'label_legend_inverted' and 'max_seq_length')
    tokenizer = tokenizer_from_json(open(trainedTokenizerFile).read())
    metadata = json.load(open(trainedMetadataFile))
    # Load the model:
    model = models.load_model(trainedModelFile)

    # a function for testing:
    def predictSpamStatus(text, clickbaitModel, pMaxSequence, pLabelLegendInverted, pTokenizer):
        sequences = pTokenizer.texts_to_sequences([text])
        xInput = pad_sequences(sequences, maxlen=pMaxSequence)
        yOutput = clickbaitModel.predict(xInput)
        preds = yOutput[0]
        labeledPredictions = {pLabelLegendInverted[str(i)]: x for i, x in enumerate(preds)}
        return labeledPredictions

    if sys.argv[1:] == []:
        # texts for the test
        sampleTexts = [
            'Alleged drug kingpin arrested in Mexico',
            'Cute Things To Distract From Your Awkward Thanksgiving',
        ]
    else:
        sampleTexts = [
            ' '.join(sys.argv[1:])
        ]

    # simple test:
    print('\n\tMODEL TEST:')
    print('=' * 20)
    for st in sampleTexts:
        preds = predictSpamStatus(st, model, metadata['max_seq_length'], metadata['label_legend_inverted'], tokenizer)
        print('TEXT       = %s' % st)
        print('PREDICTION = %s' % str(preds))
        print('*' * 20)
