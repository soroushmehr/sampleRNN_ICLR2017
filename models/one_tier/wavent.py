#!/usr/bin/env python
"""
WaveNets Audio Generation Model

How-to-run example:

sampleRNN$
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,lib.cnmem=.95 python models/one_tier/wavent.py --dim 64 --q_levels 256 --q_type linear --which_set MUSIC --batch_size 8 --wavenet_blocks 4 --dilation_layers_per_block 10 --sequence_len_to_train 1600
"""
import time
from datetime import datetime
print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
exp_start = time.time()

import os, sys
sys.path.insert(1, os.getcwd())
import argparse

import numpy
numpy.random.seed(123)
np = numpy
import random
random.seed(123)

import theano
import theano.tensor as T
import theano.ifelse
import lasagne
import scipy.io.wavfile

import lib


### Parsing passed args/hyperparameters ###
def get_args():
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue

    # No default value here. Indicate every single arguement.
    parser = argparse.ArgumentParser(
        description='two_tier.py\nNo default value! Indicate every argument.')

    # Hyperparameter arguements:
    parser.add_argument('--exp', help='Experiment name',
            type=str, required=False, default='_')
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',\
            type=check_positive, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of audio samples. Should be 256 for mu-law.',\
            type=check_positive, required=True)
    parser.add_argument('--q_type', help='Quantization in linear-scale, a-law-companding, or mu-law compandig. With mu-/a-law quantization level shoud be set as 256',\
            choices=['linear', 'a-law', 'mu-law'], required=True)
    #parser.add_argument('--nll_coeff', help='Value of alpha in [0, 1] for cost=alpha*NLL+(1-alpha)*FFT_cost',\
    #        type=check_unit_interval, required=True)
    parser.add_argument('--which_set', help='ONOM, BLIZZ, or MUSIC',
            choices=['ONOM', 'BLIZZ', 'MUSIC', 'HUCK'], required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[8, 16, 32, 64, 128, 256], required=True)
    parser.add_argument('--wavenet_blocks', help='Number of wavnet blocks to use',
            type=check_positive, required=True)
    parser.add_argument('--dilation_layers_per_block', help='number of dilation layers per block',
            type=check_positive, required=True)

    parser.add_argument('--sequence_len_to_train', help='size of output map',
            type=check_positive, required=True)

    parser.add_argument('--debug', help='debug mode', required=False, default=False, action='store_true')

    parser.add_argument('--resume', help='Resume the same model from the last checkpoint. Order of params are important. [for now]',\
            required=False, default=False, action='store_true')

    args = parser.parse_args()

    # Create tag for this experiment based on passed args
    tag = reduce(lambda a, b: a+b, sys.argv).replace('--resume', '').replace('/', '-').replace('--', '-').replace('True', 'T').replace('False', 'F')
    print "Created experiment tag for these args:"
    print tag

    return args, tag

args, tag = get_args()

# N_FRAMES = args.n_frames # How many 'frames' to include in each truncated BPTT pass
OVERLAP = (2**args.dilation_layers_per_block - 1)*args.wavenet_blocks + 1# How many samples per frame
#GLOBAL_NORM = args.global_norm
DIM = args.dim # Model dimensionality.
Q_LEVELS = args.q_levels # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
Q_TYPE = args.q_type # log- or linear-scale
#NLL_COEFF = args.nll_coeff
WHICH_SET = args.which_set
BATCH_SIZE = args.batch_size
#DATA_PATH = args.data_path

if Q_TYPE == 'mu-law' and Q_LEVELS != 256:
    raise ValueError('For mu-law Quantization levels should be exactly 256!')

# Fixed hyperparams
GRAD_CLIP = 1 # Elementwise grad clip threshold
BITRATE = 16000

# Other constants
#TRAIN_MODE = 'iters' # To use PRINT_ITERS and STOP_ITERS
TRAIN_MODE = 'time' # To use PRINT_TIME and STOP_TIME
#TRAIN_MODE = 'time-iters'
# To use PRINT_TIME for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
#TRAIN_MODE = 'iters-time'
# To use PRINT_ITERS for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
PRINT_ITERS = 10000 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 100000 # Stop after this many iterations
PRINT_TIME = 90*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*60 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
N_SEQS = 10  # Number of samples to generate every time monitoring.
FOLDER_PREFIX = os.path.join('results_wavenets', tag)
SEQ_LEN = args.sequence_len_to_train # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

LEARNING_RATE = lib.floatX(numpy.float32(0.0001))
RESUME = args.resume

epoch_str = 'epoch'
iter_str = 'iter'
lowest_valid_str = 'lowest valid cost'
corresp_test_str = 'correponding test cost'
train_nll_str, valid_nll_str, test_nll_str = \
    'train NLL (bits)', 'valid NLL (bits)', 'test NLL (bits)'

if args.debug:
    import warnings
    warnings.warn('----------RUNNING IN DEBUG MODE----------')
    TRAIN_MODE = 'time-iters'
    PRINT_TIME = 100
    STOP_TIME = 300
    STOP_ITERS = 1000

### Create directories ###
#   FOLDER_PREFIX: root, contains:
#       log.txt, __note.txt, train_log.pkl, train_log.png [, model_settings.txt]
#   FOLDER_PREFIX/params: saves all checkpoint params as pkl
#   FOLDER_PREFIX/samples: keeps all checkpoint samples as wav
#   FOLDER_PREFIX/best: keeps the best parameters, samples, ...

if not os.path.exists(FOLDER_PREFIX):
    os.makedirs(FOLDER_PREFIX)

PARAMS_PATH = os.path.join(FOLDER_PREFIX, 'params')

if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)

SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'samples')

if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)

BEST_PATH = os.path.join(FOLDER_PREFIX, 'best')

if not os.path.exists(BEST_PATH):
    os.makedirs(BEST_PATH)

lib.print_model_settings(locals(), path=FOLDER_PREFIX, sys_arg=True)

### Creating computation graph ###

def create_wavenet_block(inp, num_dilation_layer, input_dim, output_dim, name =None):
    assert name is not None
    layer_out = inp
    skip_contrib = []
    skip_weights = lib.param(name+".parametrized_weights", lib.floatX(numpy.ones((num_dilation_layer,))))
    for i in range(num_dilation_layer):
        layer_out, skip_c = lib.ops.dil_conv_1D(
                    layer_out,
                    output_dim,
                    input_dim if i == 0 else output_dim,
                    2,
                    dilation = 2**i,
                    non_linearity = 'gated',
                    name = name+".dilation_{}".format(i+1)
                )
        skip_c = skip_c*skip_weights[i]

        skip_contrib.append(skip_c)

    skip_out =  skip_contrib[-1]

    j = 0
    for i in range(num_dilation_layer-1):
        j += 2**(num_dilation_layer-i-1)
        skip_out = skip_out + skip_contrib[num_dilation_layer-2 - i][:,j:]

    return layer_out, skip_out

def create_model(inp):
    out = (inp.astype(theano.config.floatX)/lib.floatX(Q_LEVELS-1) - lib.floatX(0.5))
    l_out = out.dimshuffle(0,1,'x')

    skips = []
    for i in range(args.wavenet_blocks):
        l_out, skip_out = create_wavenet_block(l_out, args.dilation_layers_per_block, 1 if i == 0 else args.dim, args.dim, name = "block_{}".format(i+1))
        skips.append(skip_out)

    out = skips[-1]

    for i in range(args.wavenet_blocks -  1):
        out = out + skips[args.wavenet_blocks - 2 - i][:,(2**args.dilation_layers_per_block - 1)*(i+1):]

    for i in range(3):
        out = lib.ops.conv1d("out_{}".format(i+1), out, args.dim, args.dim, 1, non_linearity='relu')

    out = lib.ops.conv1d("final", out, args.dim, args.q_levels, 1, non_linearity='identity')

    return out

sequences = T.imatrix('sequences')
h0        = T.tensor3('h0')
reset     = T.iscalar('reset')
mask      = T.matrix('mask')

if args.debug:
    # Solely for debugging purposes.
    # Maybe I should set the compute_test_value=warn from here.
    sequences.tag.test_value = numpy.zeros((BATCH_SIZE, SEQ_LEN), dtype='int32')

input_sequences = sequences[:, :-1]
target_sequences = sequences[:, (2**args.dilation_layers_per_block - 1)*args.wavenet_blocks + 1:]

target_mask = mask[:, (2**args.dilation_layers_per_block - 1)*args.wavenet_blocks + 1:]

output = create_model(input_sequences)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(output.reshape((-1, Q_LEVELS))),
    target_sequences.flatten()
)

cost = cost.reshape(target_sequences.shape)
cost = cost * target_mask
# Don't use these lines; could end up with NaN
# Specially at the end of audio files where mask is
# all zero for some of the shorter files in mini-batch.
#cost = cost.sum(axis=1) / target_mask.sum(axis=1)
#cost = cost.mean(axis=0)

# Use this one instead.
cost = cost.sum()
cost = cost / target_mask.sum()

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
# log_2(e) = 1.44269504089
cost = cost * lib.floatX(numpy.log2(numpy.e))

### Getting the params, grads, updates, and Theano functions ###
params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
lib.print_params_info(params, path=FOLDER_PREFIX)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=LEARNING_RATE)

# Training function
train_fn = theano.function(
    [sequences, mask],
    cost,
    updates=updates,
    on_unused_input='warn'
)

# Validation and Test function
test_fn = theano.function(
    [sequences, mask],
    cost,
    on_unused_input='warn'
)

# Sampling at frame level
generate_fn = theano.function(
    [sequences],
    lib.ops.softmax_and_sample(output),
    on_unused_input='warn'
)


def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(
                    os.path.join(SAMPLES_PATH, name+'.wav'),
                    BITRATE,
                    data)

    total_time = time.time()
    # Generate N_SEQS' sample files, each 5 seconds long
    N_SECS = 5
    LENGTH = N_SECS*BITRATE

    if args.debug:
        LENGTH = 1024

    num_prev_samples_to_use = (2**args.dilation_layers_per_block - 1)*args.wavenet_blocks + 1

    samples = numpy.zeros((N_SEQS, LENGTH + num_prev_samples_to_use), dtype='int32')
    samples[:, :num_prev_samples_to_use] = Q_ZERO

    for t in range(LENGTH):
        samples[:,num_prev_samples_to_use+t:num_prev_samples_to_use+t+1] = generate_fn(samples[:, t:t + num_prev_samples_to_use+1])
        if (t > 2*BITRATE) and( t < 3*BITRATE):
            samples[:,num_prev_samples_to_use+t:num_prev_samples_to_use+t+1] = Q_ZERO

    total_time = time.time() - total_time
    log = "{} samples of {} seconds length generated in {} seconds."
    log = log.format(N_SEQS, N_SECS, total_time)
    print log,

    for i in xrange(N_SEQS):
        samp = samples[i, num_prev_samples_to_use: ]
        if Q_TYPE == 'mu-law':
            from datasets.dataset import mu2linear
            samp = mu2linear(samp)
        elif Q_TYPE == 'a-law':
            raise NotImplementedError('a-law is not implemented')
        write_audio_file("sample_{}_{}".format(tag, i), samp)

### Import the data_feeder ###
# Handling WHICH_SET
if WHICH_SET == 'ONOM':
    from datasets.dataset import onom_train_feed_epoch as train_feeder
    from datasets.dataset import onom_valid_feed_epoch as valid_feeder
    from datasets.dataset import onom_test_feed_epoch  as test_feeder
elif WHICH_SET == 'BLIZZ':
    from datasets.dataset import blizz_train_feed_epoch as train_feeder
    from datasets.dataset import blizz_valid_feed_epoch as valid_feeder
    from datasets.dataset import blizz_test_feed_epoch  as test_feeder
elif WHICH_SET == 'MUSIC':
    from datasets.dataset import music_train_feed_epoch as train_feeder
    from datasets.dataset import music_valid_feed_epoch as valid_feeder
    from datasets.dataset import music_test_feed_epoch  as test_feeder
elif WHICH_SET == 'HUCK':
    from datasets.dataset import huck_train_feed_epoch as train_feeder
    from datasets.dataset import huck_valid_feed_epoch as valid_feeder
    from datasets.dataset import huck_test_feed_epoch  as test_feeder


def monitor(data_feeder):
    """
    Cost and time of test_fn on a given dataset section.
    Pass only one of `valid_feeder` or `test_feeder`.
    Don't pass `train_feed`.

    :returns:
        Mean cost over the input dataset (data_feeder)
        Total time spent
    """
    _total_time = 0.
    _costs = []
    _data_feeder = data_feeder(BATCH_SIZE,
                               SEQ_LEN,
                               OVERLAP,
                               Q_LEVELS,
                               Q_ZERO,
                               Q_TYPE)

    for _seqs, _reset, _mask in _data_feeder:
        _start_time = time.time()
        _cost = test_fn(_seqs, _mask)
        _total_time += time.time() - _start_time

        _costs.append(_cost)

    return numpy.mean(_costs), _total_time


print "Wall clock time spent before training started: {:.2f}h"\
        .format((time.time()-exp_start)/3600.)
print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
costs = []
lowest_valid_cost = numpy.finfo(numpy.float32).max
corresponding_test_cost = numpy.finfo(numpy.float32).max
new_lowest_cost = False
end_of_batch = False
epoch = 0  # Important for mostly other datasets rather than Blizz

# Initial load train dataset
tr_feeder = train_feeder(BATCH_SIZE,
                         SEQ_LEN,
                         OVERLAP,
                         Q_LEVELS,
                         Q_ZERO,
                         Q_TYPE)



if RESUME:
    # Check if checkpoint from previous run is not corrupted.
    # Then overwrite some of the variables above.
    iters_to_consume, res_path, epoch, total_iters,\
        [lowest_valid_cost, corresponding_test_cost, test_cost] = \
        lib.resumable(path=FOLDER_PREFIX,
                      iter_key=iter_str,
                      epoch_key=epoch_str,
                      add_resume_counter=True,
                      other_keys=[lowest_valid_str,
                                  corresp_test_str,
                                  test_nll_str])
    # At this point we saved the pkl file.
    last_print_iters = total_iters
    print "### RESUMING JOB FROM EPOCH {}, ITER {}".format(epoch, total_iters)
    # Consumes this much iters to get to the last point in training data.
    consume_time = time.time()
    for i in xrange(iters_to_consume):
        tr_feeder.next()
    consume_time = time.time() - consume_time
    print "Train data ready in {:.2f}secs after consuming {} minibatches.".\
            format(consume_time, iters_to_consume)

    lib.load_params(res_path)
    print "Parameters from last available checkpoint loaded from path {}".format(res_path)

test_time = 0.0

while True:
    # THIS IS ONE ITERATION
    if total_iters % 500 == 0:
        print total_iters,

    total_iters += 1

    try:
        # Take as many mini-batches as possible from train set
        mini_batch = tr_feeder.next()
    except StopIteration:
        # Mini-batches are finished. Load it again.
        # Basically, one epoch.
        tr_feeder = train_feeder(BATCH_SIZE,
                                 SEQ_LEN,
                                 OVERLAP,
                                 Q_LEVELS,
                                 Q_ZERO,
                                 Q_TYPE)

        # and start taking new mini-batches again.
        mini_batch = tr_feeder.next()
        epoch += 1
        end_of_batch = True
        print "[Another epoch]",

    seqs, reset, mask = mini_batch


    ##Remove this
    # print seqs.shape
    # targ = generate_fn(seqs)
    # print targ.shape
    #####

    start_time = time.time()
    cost = train_fn(seqs, mask)
    total_time += time.time() - start_time
    #print "This cost:", cost, "This h0.mean()", h0.mean()

    costs.append(cost)

    if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
        (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='time-iters' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='iters-time' and total_iters-last_print_iters >= PRINT_ITERS) or \
        end_of_batch:
        print "\nValidation!",
        valid_cost, valid_time = monitor(valid_feeder)
        print "Done!"

        # Only when the validation cost is improved get the cost for test set.
        if valid_cost < lowest_valid_cost:
            lowest_valid_cost = valid_cost
            print "\n>>> Best validation cost of {} reached. Testing!"\
                    .format(valid_cost),
            test_cost, test_time = monitor(test_feeder)
            print "Done!"
            # Report last one which is the lowest on validation set:
            print ">>> test cost:{}\ttotal time:{}".format(test_cost, test_time)
            corresponding_test_cost = test_cost
            new_lowest_cost = True

        # Stdout the training progress
        print_info = "epoch:{}\ttotal iters:{}\twall clock time:{:.2f}h\n"
        print_info += ">>> Lowest valid cost:{}\t Corresponding test cost:{}\n"
        print_info += "\ttrain cost:{:.4f}\ttotal time:{:.2f}h\tper iter:{:.3f}s\n"
        print_info += "\tvalid cost:{:.4f}\ttotal time:{:.2f}h\n"
        print_info += "\ttest  cost:{:.4f}\ttotal time:{:.2f}h"
        print_info = print_info.format(epoch,
                                       total_iters,
                                       (time.time()-exp_start)/3600,
                                       lowest_valid_cost,
                                       corresponding_test_cost,
                                       numpy.mean(costs),
                                       total_time/3600,
                                       total_time/total_iters,
                                       valid_cost,
                                       valid_time/3600,
                                       test_cost,
                                       test_time/3600)
        print print_info

        # Save and graph training progress
        x_axis_str = 'iter'
        train_nll_str, valid_nll_str, test_nll_str = \
            'train NLL (bits)', 'valid NLL (bits)', 'test NLL (bits)'
        training_info = {'epoch' : epoch,
                         x_axis_str : total_iters,
                         train_nll_str : numpy.mean(costs),
                         valid_nll_str : valid_cost,
                         test_nll_str : test_cost,
                         'lowest valid cost' : lowest_valid_cost,
                         'correponding test cost' : corresponding_test_cost,
                         'train time' : total_time,
                         'valid time' : valid_time,
                         'test time' : test_time,
                         'wall clock time' : time.time()-exp_start}
        lib.save_training_info(training_info, FOLDER_PREFIX)
        print "Train info saved!",

        y_axis_strs = [train_nll_str, valid_nll_str, test_nll_str]
        lib.plot_traing_info(x_axis_str, y_axis_strs, FOLDER_PREFIX)
        print "Plotted!"

        # Generate and save samples
        print "Sampling!",
        tag = "e{}_i{}_t{:.2f}_tr{:.4f}_v{:.4f}"
        tag = tag.format(epoch,
                         total_iters,
                         total_time/3600,
                         numpy.mean(cost),
                         valid_cost)
        tag += ("_best" if new_lowest_cost else "")
        # Generate samples
        generate_and_save_samples(tag)
        print "Done!"

        # Save params of model
        lib.save_params(
                os.path.join(PARAMS_PATH, 'params_{}.pkl'.format(tag))
        )
        print "Params saved!"

        if total_iters-last_print_iters == PRINT_ITERS \
            or total_time-last_print_time >= PRINT_TIME:
                # If we are here b/c of onom_end_of_batch, we shouldn't mess
                # with costs and last_print_iters
            costs = []
            last_print_time += PRINT_TIME
            last_print_iters += PRINT_ITERS

        end_of_batch = False
        new_lowest_cost = False

        print "Validation Done!\nBack to Training..."

    if (TRAIN_MODE=='iters' and total_iters == STOP_ITERS) or \
       (TRAIN_MODE=='time' and total_time >= STOP_TIME) or \
       ((TRAIN_MODE=='time-iters' or TRAIN_MODE=='iters-time') and \
            (total_iters == STOP_ITERS or total_time >= STOP_TIME)):

        print "Done! Total iters:", total_iters, "Total time: ", total_time
        print "Experiment ended at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
        print "Wall clock time spent: {:.2f}h"\
                    .format((time.time()-exp_start)/3600)

        sys.exit()
