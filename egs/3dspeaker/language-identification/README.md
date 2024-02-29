# Language identification

## Introduction
This recipe offers two language identification methods that aims to predict the language category for given utterance. One is the classic method which use encoder(eres2net/cam++) to extract speaker embedddings and predict the language category through the classifier. The other approach involves several steps. Initially, phonetic information is extracted using the speech recognition model, paraformer. Subsequently, speaker embeddings are extracted through the encoder(eres2net/cam++). Finally, language prediction is carried out by the classifier.

## Usage
``` sh
pip install -r requirements.txt
# only use eres2net/cam++ model extract speaker embeddings
bash run.sh
# use paraformer to extract phoneme features and then use eres2net/cam++ model extract speaker embeddings
bash run_paraformer.sh
```

## Additional information
The language identification model using paraformer, exhibits higher accuracy for short-duration utterances. However, one drawback is the model's larger parameter size. In five-language (Chinese, English, Japanese, Cantonese, and Korean) recognition tasks, this model boasts an accuracy rate exceeding 99%.

