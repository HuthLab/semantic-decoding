import os
import numpy as np
import json

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel

from jiwer import wer
from datasets import load_metric
from bert_score import BERTScorer

BAD_WORDS_PERCEIVED_SPEECH = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])
BAD_WORDS_OTHER_TASKS = frozenset(["", "sp", "uh"])

from utils_ridge.textgrid import TextGrid
def load_transcript(experiment, task):
    if experiment in ["perceived_speech", "perceived_multispeaker"]: skip_words = BAD_WORDS_PERCEIVED_SPEECH
    else: skip_words = BAD_WORDS_OTHER_TASKS
    grid_path = os.path.join(config.DATA_TEST_DIR, "test_stimulus", experiment, task.split("_")[0] + ".TextGrid")
    transcript_data = {}
    with open(grid_path) as f: 
        grid = TextGrid(f.read())
        if experiment == "perceived_speech": transcript = grid.tiers[1].make_simple_transcript()
        else: transcript = grid.tiers[0].make_simple_transcript()
        transcript = [(float(s), float(e), w.lower()) for s, e, w in transcript if w.lower().strip("{}").strip() not in skip_words]
    transcript_data["words"] = np.array([x[2] for x in transcript])
    transcript_data["times"] = np.array([(x[0] + x[1]) / 2 for x in transcript])
    return transcript_data

"""windows of [duration] seconds at each time point"""
def windows(start_time, end_time, duration, step = 1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [(center - half, center + half) for center in range(start_time + half, end_time - half + 1) if center % step == 0]

"""divide [data] into list of segments defined by [cutoffs]"""
def segment_data(data, times, cutoffs):
    return [[x for c, x in zip(times, data) if c >= start and c < end] for start, end in cutoffs]

"""generate null sequences with same times as predicted sequence"""
def generate_null(pred_times, gpt_checkpoint, n):
    
    # load language model
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)
    
    # generate null sequences
    null_words = []
    for _count in range(n):
        decoder = Decoder(pred_times, 2 * config.EXTENSIONS)
        for sample_index in range(len(pred_times)):
            ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)
            for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[c]
                if len(nuc) < 1: continue
                extend_words = [hyp.words + [x] for x in nuc]
                likelihoods = np.random.random(len(nuc))
                local_extensions = [Hypothesis(parent = hyp, extension = x) 
                                    for x in zip(nuc, logprobs, [np.zeros(1) for _ in nuc])]
                decoder.add_extensions(local_extensions, likelihoods, nextensions)
            decoder.extend(verbose = False)
        null_words.append(decoder.beam[0].words)
    return null_words

"""
WER
"""
class WER(object):
    def __init__(self, use_score = True):
        self.use_score = use_score
    
    def score(self, ref, pred):
        scores = []
        for ref_seg, pred_seg in zip(ref, pred):
            if len(ref_seg) == 0 : error = 1.0
            else: error = wer(ref_seg, pred_seg)
            if self.use_score: scores.append(1 - error)
            else: use_score.append(error)
        return np.array(scores)
    
"""
BLEU (https://aclanthology.org/P02-1040.pdf)
"""
class BLEU(object):
    def __init__(self, n = 4):
        self.metric = load_metric("bleu", keep_in_memory=True)
        self.n = n
        
    def score(self, ref, pred):
        results = []
        for r, p in zip(ref, pred):
            self.metric.add_batch(predictions=[p], references=[[r]])
            results.append(self.metric.compute(max_order = self.n)["bleu"])
        return np.array(results)
    
"""
METEOR (https://aclanthology.org/W05-0909.pdf)
"""
class METEOR(object):
    def __init__(self):
        self.metric = load_metric("meteor", keep_in_memory=True)

    def score(self, ref, pred):
        results = []
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [" ".join(x) for x in pred]
        for r, p in zip(ref_strings, pred_strings):
            self.metric.add_batch(predictions=[p], references=[r])
            results.append(self.metric.compute()["meteor"])
        return np.array(results)
        
"""
BERTScore (https://arxiv.org/abs/1904.09675)
"""
class BERTSCORE(object):
    def __init__(self, idf_sents=None, rescale = True, score = "f"):
        self.metric = BERTScorer(lang = "en", rescale_with_baseline = rescale, idf = (idf_sents is not None), idf_sents = idf_sents)
        if score == "precision": self.score_id = 0
        elif score == "recall": self.score_id = 1
        else: self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [" ".join(x) for x in pred]
        return self.metric.score(cands = pred_strings, refs = ref_strings)[self.score_id].numpy()