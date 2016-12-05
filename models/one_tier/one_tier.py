"""
RNN Audio Generation Model

one-tier model, Quantized input
For more info:
$ python one_tier.py -h

How-to-run example:
sampleRNN$ pwd
/u/mehris/sampleRNN

sampleRNN$ \
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u \
models/one_tier/one_tier.py --exp AXIS1 --seq_len 10 --weight_norm True \
--emb_size 64 --skip_conn False --dim 32 --n_rnn 2 --rnn_type LSTM --learn_h0 \
False --q_levels 16 --q_type linear --batch_size 128 --which_set MUSIC

To resume add ` --resume` to the END of the EXACTLY above line. You can run the
resume code as many time as possible, depending on the TRAIN_MODE.
(folder name, file name, flags, their order, and the values are important)
"""
from time import time
from datetime import datetime
print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
exp_start = time()

import os, sys, glob
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

LEARNING_RATE = 0.001

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
        description='one_tier.py\nNo default value! Indicate every argument.')

    # Hyperparameter arguements:
    parser.add_argument('--exp', help='Experiment name',
            type=str, required=False, default='_')
    parser.add_argument('--seq_len', help='How many audio samples to include\
            in each truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--weight_norm', help='Adding learnable weight normalization\
            to all the linear layers (except for the embedding layer)',\
            type=t_or_f, required=True)
    parser.add_argument('--emb_size', help='Size of embedding layer (> 0)',
            type=check_positive, required=True)  # different than two_tier
    parser.add_argument('--skip_conn', help='Add skip connections to RNN', type=t_or_f, required=True)
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',\
            type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
            type=check_positive, choices=xrange(1,6), required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--learn_h0', help='Whether to learn the initial state of RNN',\
            type=t_or_f, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of audio samples. Should be 256 for mu-law.',\
            type=check_positive, required=True)
    parser.add_argument('--q_type', help='Quantization in linear-scale, a-law-companding, or mu-law compandig. With mu-/a-law quantization level shoud be set as 256',\
            choices=['linear', 'a-law', 'mu-law'], required=True)
    parser.add_argument('--which_set', help='ONOM, BLIZZ, or MUSIC',
            choices=['ONOM', 'BLIZZ', 'MUSIC'], required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[64, 128, 256], required=True)

    parser.add_argument('--debug', help='Debug mode', required=False, default=False, action='store_true')
    # NEW
    parser.add_argument('--resume', help='Resume the same model from the last checkpoint. Order of params are important. [for now]',\
            required=False, default=False, action='store_true')

    args = parser.parse_args()

    # Create tag for this experiment based on passed args
    tag = reduce(lambda a, b: a+b, sys.argv).replace('--resume', '').replace('/', '-').replace('--', '-').replace('True', 'T').replace('False', 'F')
    tag += '-lr'+str(LEARNING_RATE)
    print "Created experiment tag for these args:"
    print tag

    return args, tag

args, tag = get_args()

SEQ_LEN = args.seq_len # How many audio samples to include in each truncated BPTT pass
OVERLAP = 1
WEIGHT_NORM = args.weight_norm
EMB_SIZE = args.emb_size
SKIP_CONN = args.skip_conn
DIM = args.dim # Model dimensionality.
N_RNN = args.n_rnn # How many RNNs to stack in the frame-level model
RNN_TYPE = args.rnn_type
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1
LEARN_H0 = args.learn_h0
Q_LEVELS = args.q_levels # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
Q_TYPE = args.q_type # log- or linear-scale
WHICH_SET = args.which_set
BATCH_SIZE = args.batch_size
RESUME = args.resume

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
STOP_TIME = 60*60*24*3 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
N_SEQS = 20  # Number of samples to generate every time monitoring.
RESULTS_DIR = 'results_1t'
FOLDER_PREFIX = os.path.join(RESULTS_DIR, tag)
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

epoch_str = 'epoch'
iter_str = 'iter'
lowest_valid_str = 'lowest valid cost'
corresp_test_str = 'correponding test cost'
train_nll_str, valid_nll_str, test_nll_str = \
    'train NLL (bits)', 'valid NLL (bits)', 'test NLL (bits)'

if args.debug:
    import warnings
    warnings.warn('----------RUNNING IN DEBUG MODE----------')
    TRAIN_MODE = 'time'
    PRINT_TIME = 100
    STOP_TIME = 3000
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

def load_data(data_feeder):
    """
    Helper function to deal with interface of different datasets.
    `data_feeder` should be `train_feeder`, `valid_feeder`, or `test_feeder`.
    """
    return data_feeder(BATCH_SIZE,
                       SEQ_LEN,
                       OVERLAP,
                       Q_LEVELS,
                       Q_ZERO,
                       Q_TYPE)

### Creating computation graph ###
def sample_level_rnn(input_sequences, h0, reset):
    """
    input_sequences.shape: (batch size, seq len)
    h0.shape:              (batch size, N_RNN, DIM)
    reset.shape:           ()
    output.shape:          (batch size, seq len, DIM)
    """

    # Embedded inputs
    # Handling EMB_SIZE
    #################
    FRAME_SIZE = EMB_SIZE
    frames = lib.ops.Embedding(
        'SampleLevel.Embedding',
        Q_LEVELS,
        EMB_SIZE,
        input_sequences)

    # Real-valued inputs
    ####################
    # # 'frames' of size 1
    # FRAME_SIZE = 1
    # frames = input_sequences.reshape((
    #     input_sequences.shape[0],
    #     input_sequences.shape[1],
    #     1
    # ))
    # # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # # (a reasonable range to pass as inputs to the RNN)
    # frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    # frames *= lib.floatX(2)

    # Initial state of RNNs
    learned_h0 = lib.param(
        'SampleLevel.h0',
        numpy.zeros((N_RNN, H0_MULT*DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_RNN, H0_MULT*DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    # Handling RNN_TYPE
    # Handling SKIP_CONN
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('SampleLevel.GRU',
                                                   N_RNN,
                                                   FRAME_SIZE,
                                                   DIM,
                                                   frames,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('SampleLevel.LSTM',
                                                    N_RNN,
                                                    FRAME_SIZE,
                                                    DIM,
                                                    frames,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    out = lib.ops.Linear(
        'SampleLevel.L1',
        DIM,
        DIM,
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    out = T.nnet.relu(out)

    out = lib.ops.Linear(
        'SampleLevel.L2',
        DIM,
        DIM,
        out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    out = T.nnet.relu(out)

    out = lib.ops.Linear(
        'SampleLevel.L3',
        DIM,
        DIM,
        out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    out = T.nnet.relu(out)

    # We apply the softmax later
    out = lib.ops.Linear(
        'SampleLevel.Output',
        DIM,
        Q_LEVELS,
        out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    return (out, last_hidden)

sequences = T.imatrix('sequences')
h0        = T.tensor3('h0')
reset     = T.iscalar('reset')
mask      = T.matrix('mask')

if args.debug:
    # Solely for debugging purposes.
    # Maybe I should set the compute_test_value=warn from here.
    sequences.tag.test_value = numpy.zeros((BATCH_SIZE, SEQ_LEN+OVERLAP), dtype='int32')
    h0.tag.test_value = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM), dtype='float32')
    reset.tag.test_value = numpy.array(1, dtype='int32')
    mask.tag.test_value = numpy.ones((BATCH_SIZE, SEQ_LEN+OVERLAP), dtype='float32')

input_sequences = sequences[:, :-1]
target_sequences = sequences[:, 1:]

target_mask = mask[:, 1:]

sample_level_outputs, new_h0 = sample_level_rnn(input_sequences, h0, reset)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(sample_level_outputs.reshape((-1, Q_LEVELS))),
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
    [sequences, h0, reset, mask],
    [cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

# Validation and Test function, hence no updates
test_fn = theano.function(
    [sequences, h0, reset, mask],
    [cost, new_h0],
    on_unused_input='warn'
)

# Sampling at audio sample level
generate_outputs, generate_new_h0 = sample_level_rnn(sequences, h0, reset)
generate_fn = theano.function(
    [sequences, h0, reset],
    [lib.ops.softmax_and_sample(generate_outputs), generate_new_h0],
    on_unused_input='warn'
)

# Uniform [-0.5, 0.5) for half of initial state for generated samples
# to study the behaviour of the model and also to introduce some diversity
# to samples in a simple way. [it's disabled]
fixed_rand_h0 = numpy.random.rand(N_SEQS//2, N_RNN, H0_MULT*DIM)
fixed_rand_h0 -= 0.5
fixed_rand_h0 = fixed_rand_h0.astype('float32')

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

    total_time = time()
    # Generate N_SEQS' sample files, each 5 seconds long
    N_SECS = 5
    LENGTH = N_SECS*BITRATE if not args.debug else 100

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, 0] = Q_ZERO

    # First half zero, others fixed random at each checkpoint
    h0 = numpy.zeros(
            (N_SEQS-fixed_rand_h0.shape[0], N_RNN, H0_MULT*DIM),
            dtype='float32'
    )
    h0 = numpy.concatenate((h0, fixed_rand_h0), axis=0)

    for t in xrange(1, LENGTH):
        samples[:, t:t+1], h0 = generate_fn(
            samples[:, t-1:t],
            h0,
            numpy.int32(t == 1)
        )

    total_time = time() - total_time
    log = "{} samples of {} seconds length generated in {} seconds."
    log = log.format(N_SEQS, N_SECS, total_time)
    print log,

    for i in xrange(N_SEQS):
        samp = samples[i]
        if Q_TYPE == 'mu-law':
            from datasets.dataset import mu2linear
            samp = mu2linear(samp)
        elif Q_TYPE == 'a-law':
            raise NotImplementedError('a-law is not implemented')
        write_audio_file("sample_{}_{}".format(tag, i), samp)

def monitor(data_feeder):
    """
    Cost and time of test_fn on a given dataset section.
    Pass only one of `valid_feeder` or `test_feeder`.
    Don't pass `train_feed`.

    :returns:
        Mean cost over the input dataset (data_feeder)
        Total time spent
    """
    _total_time = time()
    _h0 = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM), dtype='float32')
    _costs = []
    _data_feeder = load_data(data_feeder)
    for _seqs, _reset, _mask in _data_feeder:
        _cost, _h0 = test_fn(_seqs, _h0, _reset, _mask)
        _costs.append(_cost)

    return numpy.mean(_costs), time() - _total_time

print "Wall clock time spent before training started: {:.2f}h"\
        .format((time()-exp_start)/3600.)
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
epoch = 0

h0 = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM), dtype='float32')

# Initial load train dataset
tr_feeder = load_data(train_feeder)

### Handling the resume option:
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
    consume_time = time()
    for i in xrange(iters_to_consume):
        tr_feeder.next()
    consume_time = time() - consume_time
    print "Train data ready in {:.2f}secs after consuming {} minibatches.".\
            format(consume_time, iters_to_consume)

    lib.load_params(res_path)
    print "Parameters from last available checkpoint loaded."

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
        tr_feeder = load_data(train_feeder)

        # and start taking new mini-batches again.
        mini_batch = tr_feeder.next()
        epoch += 1
        end_of_batch = True
        print "[Another epoch]",

    seqs, reset, mask = mini_batch

    start_time = time()
    cost, h0 = train_fn(seqs, h0, reset, mask)
    total_time += time() - start_time
    #print "This cost:", cost, "This h0.mean()", h0.mean()

    costs.append(cost)

    # Monitoring step
    if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
        (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='time-iters' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='iters-time' and total_iters-last_print_iters >= PRINT_ITERS) or \
        end_of_batch:
        # 0. Validation
        print "\nValidation!",
        valid_cost, valid_time = monitor(valid_feeder)
        print "Done!"

        # 1. Test
        test_time = 0.
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

        # 2. Stdout the training progress
        print_info = "epoch:{}\ttotal iters:{}\twall clock time:{:.2f}h\n"
        print_info += ">>> Lowest valid cost:{}\t Corresponding test cost:{}\n"
        print_info += "\ttrain cost:{:.4f}\ttotal time:{:.2f}h\tper iter:{:.3f}s\n"
        print_info += "\tvalid cost:{:.4f}\ttotal time:{:.2f}h\n"
        print_info += "\ttest  cost:{:.4f}\ttotal time:{:.2f}h"
        print_info = print_info.format(epoch,
                                       total_iters,
                                       (time()-exp_start)/3600,
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

        tag = "e{}_i{}_t{:.2f}_tr{:.4f}_v{:.4f}"
        tag = tag.format(epoch,
                         total_iters,
                         total_time/3600,
                         numpy.mean(cost),
                         valid_cost)
        tag += ("_best" if new_lowest_cost else "")

        # 3. Save params of model (IO bound, time consuming)
        # If saving params is not successful, there shouldn't be any trace of
        # successful monitoring step in train_log as well.
        print "Saving params!",
        lib.save_params(
                os.path.join(PARAMS_PATH, 'params_{}.pkl'.format(tag))
        )
        print "Done!"

        # 4. Save and graph training progress (fast)
        training_info = {epoch_str : epoch,
                         iter_str : total_iters,
                         train_nll_str : numpy.mean(costs),
                         valid_nll_str : valid_cost,
                         test_nll_str : test_cost,
                         lowest_valid_str : lowest_valid_cost,
                         corresp_test_str : corresponding_test_cost,
                         'train time' : total_time,
                         'valid time' : valid_time,
                         'test time' : test_time,
                         'wall clock time' : time()-exp_start}
        lib.save_training_info(training_info, FOLDER_PREFIX)
        print "Train info saved!",

        y_axis_strs = [train_nll_str, valid_nll_str, test_nll_str]
        lib.plot_traing_info(iter_str, y_axis_strs, FOLDER_PREFIX)
        print "And plotted!"

        # 5. Generate and save samples (time consuming)
        # If not successful, we still have the params to sample afterward
        print "Sampling!",
        # Generate samples
        generate_and_save_samples(tag)
        print "Done!"

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
                    .format((time()-exp_start)/3600)

        sys.exit()
