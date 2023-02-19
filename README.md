# Semantic Decoding

This repository contains code used in the paper "Semantic reconstruction of continuous language from non-invasive brain recordings" by Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth.  

## Usage

1. Download [language model data](https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip) and extract contents into new `data_lm/` directory. 

2. Download [encoding model data](https://utexas.box.com/shared/static/f52cue02voyqzkjcsb79adulo5lxpyv9.zip) and extract contents into new `data_em/` directory. Stimulus data for `data_stimulus/` and response data for `data_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds003020/).

3. Download [test data]() and extract contents into new `data_test/` directory.

4. Estimate the encoding model. The encoding model predicts brain responses from contextual features of the stimulus extracted using GPT. The `--gpt` parameter determines the GPT checkpoint used. Use `--gpt imagined` when estimating models for imagined speech data, as this will extract features using a GPT checkpoint that was not trained on the imagined speech stories. Use `--gpt perceived` when estimating models for other data. The encoding model will be saved in `MODEL_DIR/[SUBJECT_ID]`

```bash
python3 decoding/train_EM.py --subject [SUBJECT_ID] --gpt perceived
```

5. Estimate the word rate model. The word rate model predicts word times from brain responses. Two word rate models will be saved in `MODEL_DIR/[SUBJECT_ID]`. The `word_rate_model_speech` model uses brain responses in speech regions, and should be used when decoding imagined speech and perceived movie data. The `word_rate_model_auditory` model uses brain responses in auditory cortex, and should be used when decoding perceived speech data.

```bash
python3 decoding/train_WR.py --subject [SUBJECT_ID]
```

6. Test the decoder on brain responses not used in model estimation. The decoder predictions will be saved in `RESULTS_DIR/[SUBJECT_ID]`

```bash
python3 decoding/run_decoder.py --subject [SUBJECT_ID] --scan perceived_speech/wheretheressmoke.hf5 --gpt perceived --word_rate auditory
```
