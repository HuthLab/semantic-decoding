import os
import numpy as np
import json
import argparse

import config
from utils_stim import get_story_wordseqs
from utils_resp import get_resp
from utils_ridge.DataSequence import DataSequence
from utils_ridge.util import make_delayed
from utils_ridge.ridge import bootstrap_ridge
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    # ROI voxels
    with open(os.path.join(config.DATA_TRAIN_DIR, "ROIs", "%s.json" % args.subject), "r") as f:
        vox = json.load(f)
            
    # estimate word rate model
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    
    wordseqs = get_story_wordseqs(stories)
    rates = {}
    for story in stories:
        ds = wordseqs[story]
        words = DataSequence(np.ones(len(ds.data_times)), ds.split_inds, ds.data_times, ds.tr_times)
        rates[story] = words.chunksums("lanczos", window = 3)
    nz_rate = np.concatenate([rates[story][5+config.TRIM:-config.TRIM] for story in stories], axis = 0)
    nz_rate = np.nan_to_num(nz_rate).reshape([-1, 1])
    mean_rate = np.mean(nz_rate)
    rate = nz_rate - mean_rate
    
    for roi in ["speech", "auditory"]:
        resp = get_resp(args.subject, stories, stack = True, vox = vox[roi])
        delresp = make_delayed(resp, config.RESP_DELAYS)
        nchunks = int(np.ceil(delresp.shape[0] / 5 / config.CHUNKLEN))    
        weights, _, _ = bootstrap_ridge(delresp, rate, use_corr = False,
            alphas = config.ALPHAS, nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)
        np.savez(os.path.join(save_location, "word_rate_model_%s" % roi), 
            weights = weights, mean_rate = mean_rate, voxels = vox[roi])