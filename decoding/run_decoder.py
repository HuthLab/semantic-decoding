import os
import numpy as np
import json
import argparse
import h5py
from pathlib import Path

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--scan", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--word_rate", type = str, default = "auditory")
    args = parser.parse_args()
    
    # load responses
    hf = h5py.File(os.path.join(config.TEST_DATA_DIR, args.subject, args.scan), "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    
    # load gpt
    with open(os.path.join(config.LM_DATA_DIR, args.gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.LM_DATA_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.LM_DATA_DIR, args.gpt, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    # load models
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % args.word_rate), allow_pickle = True)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % args.gpt))
    weights = encoding_model["weights"]
    noise_model = encoding_model['noise_model']
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.scan not in encoding_model["stories"]
    
    # predict word times
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # decode responses
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)
    for sample_index in range(len(word_times)):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
    save_location = os.path.join(config.RESULT_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    decoder.save(os.path.join(save_location, Path(args.scan).stem))