import numpy as np
import itertools as itools
from utils_ridge.interpdata import sincinterp2D, gabor_xfm2D, lanczosinterp2D

class DataSequence(object):
    """DataSequence class provides a nice interface for handling data that is both continuous
    and discretely chunked. For example, semantic projections of speech stimuli must be
    considered both at the level of single words (which are continuous throughout the stimulus)
    and at the level of TRs (which contain discrete chunks of words).
    """
    def __init__(self, data, split_inds, data_times=None, tr_times=None):
        """Initializes the DataSequence with the given [data] object (which can be any iterable)
        and a collection of [split_inds], which should be the indices where the data is split into
        separate TR chunks.
        """
        self.data = data
        self.split_inds = split_inds
        self.data_times = data_times
        self.tr_times = tr_times

    def mapdata(self, fun):
        """Creates a new DataSequence where each element of [data] is produced by mapping the
        function [fun] onto this DataSequence's [data].
        The [split_inds] are preserved exactly.
        """
        return DataSequence(self, map(fun, self.data), self.split_inds)

    def chunks(self):
        """Splits the stored [data] into the discrete chunks and returns them.
        """
        return np.split(self.data, self.split_inds)

    def data_to_chunk_ind(self, dataind):
        """Returns the index of the chunk containing the data with the given index.
        """
        zc = np.zeros((len(self.data),))
        zc[dataind] = 1.0
        ch = np.array([ch.sum() for ch in np.split(zc, self.split_inds)])
        return np.nonzero(ch)[0][0]

    def chunk_to_data_ind(self, chunkind):
        """Returns the indexes of the data contained in the chunk with the given index.
        """
        return list(np.split(np.arange(len(self.data)), self.split_inds)[chunkind])

    def chunkmeans(self):
        """Splits the stored [data] into the discrete chunks, then takes the mean of each chunk
        (this is assuming that [data] is a numpy array) and returns the resulting matrix with
        one row per chunk.
        """
        dsize = self.data.shape[1]
        outmat = np.zeros((len(self.split_inds)+1, dsize))
        for ci, c in enumerate(self.chunks()):
            if len(c):
                outmat[ci] = np.vstack(c).mean(0)

        return outmat

    def chunksums(self, interp="rect", **kwargs):
        """Splits the stored [data] into the discrete chunks, then takes the sum of each chunk
        (this is assuming that [data] is a numpy array) and returns the resulting matrix with
        one row per chunk.
        If [interp] is "sinc", the signal will be downsampled using a truncated sinc filter
        instead of a rectangular filter.
        if [interp] is "lanczos", the signal will be downsampled using a Lanczos filter.
        [kwargs] are passed to the interpolation function.
        """
        if interp=="sinc":
            ## downsample using sinc filter
            return sincinterp2D(self.data, self.data_times, self.tr_times, **kwargs)
        elif interp=="lanczos":
            ## downsample using Lanczos filter
            return lanczosinterp2D(self.data, self.data_times, self.tr_times, **kwargs)
        elif interp=="gabor":
            ## downsample using Gabor filter
            return np.abs(gabor_xfm2D(self.data.T, self.data_times, self.tr_times, **kwargs)).T
        else:
            dsize = self.data.shape[1]
            outmat = np.zeros((len(self.split_inds)+1, dsize))
            for ci, c in enumerate(self.chunks()):
                if len(c):
                    outmat[ci] = np.vstack(c).sum(0)
                    
            return outmat

    def copy(self):
        """Returns a copy of this DataSequence.
        """
        return DataSequence(list(self.data), self.split_inds.copy(), self.data_times, self.tr_times)
    
    @classmethod
    def from_grid(cls, grid_transcript, trfile):
        """Creates a new DataSequence from a [grid_transript] and a [trfile].
        grid_transcript should be the product of the 'make_simple_transcript' method of TextGrid.
        """
        data_entries = list(zip(*grid_transcript))[2]
        if isinstance(data_entries[0], str):
            data = list(map(str.lower, list(zip(*grid_transcript))[2]))
        else:
            data = data_entries
        word_starts = np.array(list(map(float, list(zip(*grid_transcript))[0])))
        word_ends = np.array(list(map(float, list(zip(*grid_transcript))[1])))
        word_avgtimes = (word_starts + word_ends)/2.0
        
        tr = trfile.avgtr
        trtimes = trfile.get_reltriggertimes()
        
        split_inds = [(word_starts<(t+tr)).sum() for t in trtimes][:-1]
        return cls(data, split_inds, word_avgtimes, trtimes+tr/2.0)

    @classmethod
    def from_chunks(cls, chunks):
        """The inverse operation of DataSequence.chunks(), this function concatenates
        the [chunks] and infers split_inds.
        """
        lens = map(len, chunks)
        split_inds = np.cumsum(lens)[:-1]
        #data = reduce(list.__add__, map(list, chunks)) ## 2.26s for 10k 6-w chunks
        data = list(itools.chain(*map(list, chunks))) ## 19.6ms for 10k 6-w chunks
        return cls(data, split_inds)