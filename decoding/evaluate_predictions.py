import os
import numpy as np
import json
import argparse

import config
from utils_eval import generate_null, load_transcript, windows, segment_data, WER, BLEU, METEOR, BERTSCORE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--metrics", nargs = "+", type = str, default = ["WER", "BLEU", "METEOR", "BERT"])
    parser.add_argument("--references", nargs = "+", type = str, default = [])
    parser.add_argument("--null", type = int, default = 10)
    args = parser.parse_args()
    
    if len(args.references) == 0:
        args.references.append(args.task)
        
    with open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)
                
    # load language similarity metrics
    metrics = {}
    if "WER" in args.metrics: metrics["WER"] = WER(use_score = True)
    if "BLEU" in args.metrics: metrics["BLEU"] = BLEU(n = 1)
    if "METEOR" in args.metrics: metrics["METEOR"] = METEOR()
    if "BERT" in args.metrics: metrics["BERT"] = BERTSCORE(
        idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")), 
        rescale = False, 
        score = "recall")

    # load prediction transcript
    pred_path = os.path.join(config.RESULT_DIR, args.subject, args.experiment, args.task + ".npz")
    pred_data = np.load(pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # generate null sequences
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"
    null_word_list = generate_null(pred_times, gpt_checkpoint, args.null)
        
    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}
    for reference in args.references:

        # load reference transcript
        ref_data = load_transcript(args.experiment, reference)
        ref_words, ref_times = ref_data["words"], ref_data["times"]

        # segment prediction and reference words into windows
        window_cutoffs = windows(*eval_segments[args.task], config.WINDOW)
        ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
        pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
        null_window_list = [segment_data(null_words, pred_times, window_cutoffs) for null_words in null_word_list]
        
        for mname, metric in metrics.items():

            # get null score for each window and the entire story
            window_null_scores = np.array([metric.score(ref = ref_windows, pred = null_windows) 
                                           for null_windows in null_window_list])
            story_null_scores = window_null_scores.mean(1)

            # get raw score and normalized score for each window
            window_scores[(reference, mname)] = metric.score(ref = ref_windows, pred = pred_windows)
            window_zscores[(reference, mname)] = (window_scores[(reference, mname)] 
                                                  - window_null_scores.mean(0)) / window_null_scores.std(0)

            # get raw score and normalized score for the entire story
            story_scores[(reference, mname)] = metric.score(ref = ref_windows, pred = pred_windows)
            story_zscores[(reference, mname)] = (story_scores[(reference, mname)].mean()
                                                 - story_null_scores.mean()) / story_null_scores.std()
    
    save_location = os.path.join(config.REPO_DIR, "scores", args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, args.task), 
             window_scores = window_scores, window_zscores = window_zscores, 
             story_scores = story_scores, story_zscores = story_zscores)