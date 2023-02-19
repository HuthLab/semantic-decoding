import numpy as np
import scipy.stats as ss

class Decoder(object):
    """class for beam search decoding
    """
    def __init__(self, word_times, beam_width, extensions = 5):
        self.word_times = word_times
        self.beam_width, self.extensions = beam_width, extensions
        self.beam = [Hypothesis()] # initialize with empty hypothesis
        self.scored_extensions = [] # global extension pool
        
    def first_difference(self):
        """get first index where hypotheses on the beam differ
        """
        words_arr = np.array([hypothesis.words for hypothesis in self.beam])
        if words_arr.shape[0] == 1: return words_arr.shape[1]
        for index in range(words_arr.shape[1]): 
            if len(set(words_arr[:, index])) > 1: return index
        return 0
    
    def time_window(self, sample_index, seconds, floor = 0):
        """number of prior words within [seconds] of the currently sampled time point"""
        window = [time for time in self.word_times if time < self.word_times[sample_index] 
                  and time > self.word_times[sample_index] - seconds]
        return max(len(window), floor)
        
    def get_hypotheses(self):
        """get the number of permitted extensions for each hypothesis on the beam
        """
        if len(self.beam[0].words) == 0: 
            return zip(self.beam, [self.extensions for hypothesis in self.beam])
        logprobs = [sum(hypothesis.logprobs) for hypothesis in self.beam]
        num_extensions = [int(np.ceil(self.extensions * rank / len(logprobs))) for 
                          rank in ss.rankdata(logprobs)]
        return zip(self.beam, num_extensions)
    
    def add_extensions(self, extensions, likelihoods, num_extensions):
        """add extensions for each hypothesis to global extension pool
        """
        scored_extensions = sorted(zip(extensions, likelihoods), key = lambda x : -x[1])
        self.scored_extensions.extend(scored_extensions[:num_extensions])

    def extend(self, verbose = False):
        """update beam based on global extension pool 
        """
        self.beam = [x[0] for x in sorted(self.scored_extensions, key = lambda x : -x[1])[:self.beam_width]]
        self.scored_extensions = []
        if verbose: print(self.beam[0].words)
        
    def save(self, path):
        """save decoder results
        """
        np.savez(path, words = np.array(self.beam[0].words), times = np.array(self.word_times))
        
class Hypothesis(object):
    """a class for representing word sequence hypotheses
    """
    def __init__(self, parent = None, extension = None):
        if parent is None: 
            self.words, self.logprobs, self.embs = [], [], []
        else:
            word, logprob, emb = extension
            self.words = parent.words + [word]
            self.logprobs = parent.logprobs + [logprob]
            self.embs = parent.embs + [emb]