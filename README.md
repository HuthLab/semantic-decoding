# Semantic Decoding

This repository contains code used in the paper "Semantic reconstruction of continuous language from non-invasive brain recordings" by Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth.  

## Usage

1. Download [language model data](https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip) and extract contents into new `data_lm/` directory. 

2. Download [training data](https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip) and extract contents into new `data_train/` directory. Stimulus data for `train_stimulus/` and response data for `train_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds003020/).

3. Download [test data](https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip) and extract contents into new `data_test/` directory. Stimulus data for `test_stimulus/[EXPERIMENT]` and response data for `test_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds004510/).

4. Estimate the encoding model. The encoding model predicts brain responses from contextual features of the stimulus extracted using GPT. The `--gpt` parameter determines the GPT checkpoint used. Use `--gpt imagined` when estimating models for imagined speech data, as this will extract features using a GPT checkpoint that was not trained on the imagined speech stories. Use `--gpt perceived` when estimating models for other data. The encoding model will be saved in `MODEL_DIR/[SUBJECT_ID]`. Alternatively, download [pre-fit encoding models](https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz).

```bash
python3 decoding/train_EM.py --subject [SUBJECT_ID] --gpt perceived
```

5. Estimate the word rate model. The word rate model predicts word times from brain responses. Two word rate models will be saved in `MODEL_DIR/[SUBJECT_ID]`. The `word_rate_model_speech` model uses brain responses in speech regions, and should be used when decoding imagined speech and perceived movie data. The `word_rate_model_auditory` model uses brain responses in auditory cortex, and should be used when decoding perceived speech data. Alternatively, download [pre-fit word rate models](https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz).

```bash
python3 decoding/train_WR.py --subject [SUBJECT_ID]
```

6. Test the decoder on brain responses not used in model estimation. The decoder predictions will be saved in `RESULTS_DIR/[SUBJECT_ID]/[EXPERIMENT_NAME]`.

```bash
python3 decoding/run_decoder.py --subject [SUBJECT_ID] --experiment [EXPERIMENT_NAME] --task [TASK_NAME]
```

7. Evaluate the decoder predictions against reference transcripts. The evaluation results will be saved in `SCORE_DIR/[SUBJECT_ID]/[EXPERIMENT_NAME]`.

```bash
python3 decoding/evaluate_predictions.py --subject [SUBJECT_ID] --experiment [EXPERIMENT_NAME] --task [TASK_NAME]
```