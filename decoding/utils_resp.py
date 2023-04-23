import os
import numpy as np
import h5py

import config

def get_resp(subject, stories, stack = True, vox = None):
    """loads response data
    """
    subject_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", subject)
    resp = {}
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        resp[story] = np.nan_to_num(hf["data"][:])
        if vox is not None:
            resp[story] = resp[story][:, vox]
        hf.close()
    if stack: return np.vstack([resp[story] for story in stories]) 
    else: return resp