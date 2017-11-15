"""
Functions for dealing with data input and output.

"""

import os
import gzip
import logging
import numpy as np
import struct
from random import shuffle

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                            GENERAL I/O FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def smart_open(filename, mode=None):
    """Opens a file normally or using gzip based on the extension."""
    if os.path.splitext(filename)[-1] == ".gz":
        if mode is None:
            mode = "rb"
        return gzip.open(filename, mode)
    else:
        if mode is None:
         mode = "r"
        return open(filename, mode)

def read_kaldi_ark_from_scp(uid, offset, batch_size, buffer_size, scp_fn, ark_base_dir=""):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys. Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py

    Parameters
    ----------
    ark_base_dir : str
        The base directory for the archives to which the SCP points.
    """

    ark_dict = {}
    totframes = 0
    lines = 0
    with open(scp_fn) as f:
        for line in f:
            lines = lines + 1
            if lines<=uid:
                continue
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split()
            ark_path, pos = path_pos.split(":")
            ark_path = os.path.join(ark_base_dir, ark_path)
            ark_read_buffer = smart_open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            #assert header[0] == "B", "Input .ark file is not binary"
            rows = 0
            cols = 0
            m,rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n,cols = struct.unpack("<bi", ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            if len(tmp_mat) != rows * cols:
                return {}, lines
            utt_mat = np.reshape(tmp_mat, (rows, cols))
            #utt_mat_list=utt_mat.tolist()
            ark_read_buffer.close()
            ark_dict[utt_id] = utt_mat
            totframes += rows
            if totframes>=(batch_size*buffer_size-offset):
                break

    return ark_dict,lines

def kaldi_write_mats(ark_path, utt_id, utt_mat):
    ark_write_buf = smart_open(ark_path, "ab")
    utt_mat = np.asarray(utt_mat, dtype=np.float32)
    batch, rows, cols = utt_mat.shape
    ark_write_buf.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
    ark_write_buf.write(struct.pack('<cxcccc', b' ',b'B',b'F',b'M',b' '))
    ark_write_buf.write(struct.pack('<bi', 4, rows))
    ark_write_buf.write(struct.pack('<bi', 4, cols))
    ark_write_buf.write(utt_mat)

def load_utterance_locations(data_dir, frame_file):

    locations = {}

    with open(os.path.join(data_dir, frame_file)) as f:
        for line in f:
            utterance_id, path = line.replace("\n", "").split()
            path, location = path.split(":")
            ark_path = os.path.join(data_dir, path)
            locations[utterance_id] = int(location)

    return locations, ark_path

def read_mat(buff, byte):
    buff.seek(byte, 0)
    header = struct.unpack("<xcccc", buff.read(5))
    m, rows = struct.unpack("<bi", buff.read(5))
    n, cols = struct.unpack("<bi", buff.read(5))
    tmp_mat = np.frombuffer(buff.read(rows * cols * 4), dtype=np.float32)
    return np.reshape(tmp_mat, (rows, cols))

def load_senones(data_dir, senone_file):

    senones = {}

    with open(os.path.join(data_dir, senone_file)) as f:
        for line in f:
            line = line.split()
            labels = [int(line[i]) for i in range(2, len(line), 4)]
            onehot = np.zeros((len(labels), 1999), dtype=np.int)
            for i, label in enumerate(labels):
                onehot[i, label] = 1

            senones[line[0]] = onehot

    return senones

def count_frames(data_dir, frame_file, input_featdim):

    frame_count = 0
    current_byte = 0

    for line in open(os.path.join(data_dir, frame_file)):
        byte = int(line[line.index(':') + 1 :])
        frame_count += (byte - current_byte - 25) // 4 // input_featdim
        current_byte = byte

    return frame_count

class DataLoader:
    """ Class for loading features and senone labels from file into a buffer, and batching. """

    def __init__(self,
            base_dir,
            frame_file,
            batch_size,
            buffer_size,
            context,
            out_frames,
            shuffle,
            input_featdim = 771,
            clean_file = None,
            senone_file = None,
        ):

        """ Initialize the data loader including filling the buffer """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.context = context
        self.out_frames = out_frames
        self.shuffle = shuffle

        self.uid = 0
        self.offset = 0

        in_locations, self.in_ark_path = load_utterance_locations(base_dir, frame_file)

        self.clean_file = clean_file
        if clean_file:
            clean_locations, self.clean_ark_path = load_utterance_locations(base_dir, clean_file)

        self.senone_file = senone_file
        if senone_file:
            senone_labels = load_senones(base_dir, senone_file)

        self.locations = []
        for key in in_locations:
            location = {'id':key, 'in_byte': in_locations[key]}

            if clean_file:
                location['clean_byte'] = clean_locations[key]

            if senone_file:
                location['senones'] = senone_labels[key]

            self.locations.append(location)

        self.frame_count = count_frames(base_dir, frame_file, input_featdim)

        self.empty = True


    def read_mats(self):
        """ Read features from file into a buffer """
        #Read a buffer containing buffer_size*batch_size+offset
        #Returns a line number of the scp file

        result = {'in_dict':{}}
        in_ark_buffer = smart_open(self.in_ark_path, "rb")

        if self.clean_file is not None:
            result['clean_dict'] = {}
            clean_ark_buffer = smart_open(self.clean_ark_path, "rb")

        if self.senone_file is not None:
            result['senone_dict'] = {}

        totframes = 0
        while totframes < self.batch_size * self.buffer_size - self.offset and self.uid < len(self.locations):
            in_mat = read_mat(in_ark_buffer, self.locations[self.uid]['in_byte'])
            result['in_dict'][self.locations[self.uid]['id']] = in_mat

            if self.clean_file is not None:
                clean_mat = read_mat(clean_ark_buffer, self.locations[self.uid]['clean_byte'])
                result['clean_dict'][self.locations[self.uid]['id']] = clean_mat

            if self.senone_file is not None:
                result['senone_dict'][self.locations[self.uid]['id']] = self.locations[self.uid]['senones']

            totframes += len(in_mat)
            self.uid += 1

        in_ark_buffer.close()

        if self.clean_file:
            clean_ark_buffer.close()

        return result

    def _fill_buffer(self):
        """ Read data from files into buffers """

        # Read data
        mats = self.read_mats()
        frame_dict = mats['in_dict']

        if self.senone_file is not None:
            senone_dict = mats['senone_dict']

        if self.clean_file is not None:
            clean_dict = mats['clean_dict']

        if len(frame_dict) == 0:
            self.empty = True
            return

        ids = sorted(frame_dict.keys())

        if not hasattr(self, 'offset_frames'):
            self.offset_frames = np.empty((0, frame_dict[ids[0]].shape[1]), np.float32)

        if not hasattr(self, 'offset_senones') and self.senone_file is not None:
            self.offset_senones = np.empty((0, senone_dict[ids[0]].shape[1]), np.float32)

        if not hasattr(self, 'offset_clean') and self.clean_file is not None:
            self.offset_clean = np.empty((0, clean_dict[ids[0]].shape[1]), np.float32)

        # Create frame buffers
        frames = [frame_dict[i] for i in ids]
        frames = np.vstack(frames)
        frames = np.concatenate((self.offset_frames, frames), axis=0)

        if self.clean_file is not None:
            clean = [clean_dict[i] for i in ids]
            clean = np.vstack(clean)
            clean = np.concatenate((self.offset_clean, clean), axis=0)

        if self.senone_file is not None:
            senone = [senone_dict[i] for i in ids]
            senone = np.vstack(senone)
            senone = np.concatenate((self.offset_senones, senone), axis=0)

        # Put one batch into the offset frames
        cutoff = self.batch_size * self.buffer_size
        if frames.shape[0] >= cutoff:
            self.offset_frames = frames[cutoff:]
            frames = frames[:cutoff]

            if self.senone_file is not None:
                self.offset_senones = senone[cutoff:]
                senone = senone[:cutoff]

            if self.clean_file is not None:
                self.offset_clean = clean[cutoff:]
                clean = clean[:cutoff]

            self.offset = self.offset_frames.shape[0]

        # Generate a random permutation of indexes
        if self.shuffle:
            self.indexes = np.random.permutation(frames.shape[0])
        else:
            self.indexes = np.arange(frames.shape[0])

        frames = np.pad(
            array     = frames,
            pad_width = ((self.context + self.out_frames // 2,),(0,)),
            mode      = 'edge')
        self.frame_buffer = frames

        if self.clean_file is not None:
            clean = np.pad(
                array     = clean,
                pad_width = ((self.context,),(0,)),
                mode      = 'edge')
            self.clean_buffer = clean

        if self.senone_file is not None:
            self.senone_buffer = senone


    def batchify(self, shuffle_batches=False, include_deltas=True):
        """ Make a batch of frames and senones """

        batch_index = 0
        self.reset(shuffle_batches)
        batch = {}

        while not self.empty:
            start = batch_index * self.batch_size
            end = min((batch_index+1) * self.batch_size, len(self.indexes))

            # Collect the data
            batch['frame'] = np.stack((self.frame_buffer[i:i+self.out_frames+2*self.context,]
                for i in self.indexes[start:end]), axis = 0)

            if not include_deltas:
                batch['frame'] = batch['frame'][:,:,:257]

            if self.clean_file is not None:
                batch['clean'] = np.stack((self.clean_buffer[i:i+self.out_frames,]
                    for i in self.indexes[start:end]), axis = 0)

            if self.senone_file is not None:
                batch['label'] = self.senone_buffer[self.indexes[start:end]]
            elif self.clean_file is not None:
                batch['label'] = batch['clean']

            # Increment batch, and if necessary re-fill buffer
            batch_index += 1
            if batch_index * self.batch_size >= len(self.indexes):
                batch_index = 0
                self._fill_buffer()

            yield batch


    def reset(self, shuffle_batches):
        self.uid = 0
        self.offset = 0
        self.empty = False
        if shuffle_batches:
            shuffle(self.locations)

        self._fill_buffer()
