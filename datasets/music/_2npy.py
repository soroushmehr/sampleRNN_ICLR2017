import numpy
np = numpy
import scipy.io.wavfile
import scikits.audiolab

import random
import time
import os
import glob

__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return
    #import collections
    #if isinstance(inp_list, (collections.Sequence)):
    if isinstance(inp_list, numpy.ndarray):
        numpy.random.seed(__RAND_SEED)
        numpy.random.shuffle(inp_list)
        return
    # destructive operations; in place; no need to return
    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

data_path = os.path.abspath('./download/parts')
print data_path

paths = sorted(glob.glob(data_path+"/*.flac"))
__fixed_shuffle(paths)

arr = [(scikits.audiolab.flacread(p)[0]).astype('float16') for p in paths]
np_arr = np.array(arr)
np.save('all_music.npy', np_arr)
np.save('music_train.npy', np_arr[:-2*256])
np.save('music_valid.npy', np_arr[-2*256:-256])
np.save('music_test.npy', np_arr[-256:])
