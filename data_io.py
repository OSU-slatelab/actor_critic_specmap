"""
Functions for dealing with data input and output.

"""

import os
import gzip
import logging
import numpy as np
import struct

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

def np_from_text(text_fn, phonedict, txt_base_dir=""):
    ark_dict = {}
    with open(text_fn) as f:
        for line in f:
            if line == "":
                continue
        utt_id = line.replace("\n", "").split(" ")[0]
        text = line.replace("\n", "").split(" ")[1:]
        rows = len(text)
        #cols = 51
        utt_mat = np.zeros((rows))
    for i in range(len(text)):
        utt_mat[i] = phonedict[text[i]]
        ark_dict[utt_id] = utt_mat
    return ark_dict

def read_senones_from_text(uid, offset, batch_size, buffer_size, senone_fn, senone_base_dir=""): 
    senonedict = {}
    totframes = 0
    lines = 0
    with open(senone_fn) as f:
        for line in f:
            lines += 1
            if lines<=uid:
                continue
            if line == "":
                continue
            A = []
            utt_id = line.split()[0]
            prev_word = ""
            for word in line.split():
                if prev_word=='[':
                    A.append(word)
                prev_word=word
            totframes += len(A)
            senone_mat = np.zeros((len(A),1999))
            for i in range(len(A)):
                senone_mat[i][int(A[i])] = 1
            senonedict[utt_id] = senone_mat
            if totframes>=(batch_size*buffer_size-offset):
                break

    return senonedict, lines
            
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


class DataLoader:
    """ Class for loading features and senone labels from file into a buffer, and batching. """

    def __init__(self,
            base_dir,
            frame_file,
            senone_file,
            batch_size,
            buffer_size,
            context,
            out_frames,
            shuffle):
        """ Initialize the data loader including filling the buffer """
        self.data_dir = base_dir
        self.frame_file = frame_file
        self.senone_file = senone_file
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.context = context
        self.out_frames = out_frames
        self.shuffle = shuffle
        
        self.uid = 0
        self.offset = 0

        self._count_frames_from_senone_file()

        self.empty = True

    def _count_frames_from_senone_file(self):

        self.frame_count = 0

        for line in open(os.path.join(self.data_dir, self.senone_file)):
            self.frame_count += (len(line.split()) - 1) // 4


    def read_mats(self):
        """ Read features from file into a buffer """
        #Read a buffer containing buffer_size*batch_size+offset 
        #Returns a line number of the scp file
        scp_fn = os.path.join(self.data_dir, self.frame_file)
        ark_dict, uid = read_kaldi_ark_from_scp(
                self.uid,
                self.offset,
                self.batch_size,
                self.buffer_size,
                scp_fn,
                self.data_dir)

        return ark_dict, uid

    def read_senones(self):
        """ Read senones from file """
        scp_fn = os.path.join(self.data_dir, self.senone_file)
        senone_dict, uid = read_senones_from_text(
                self.uid,
                self.offset,
                self.batch_size,
                self.buffer_size,
                scp_fn,
                self.data_dir)

        return senone_dict, uid

    def _fill_buffer(self):
        """ Read data from files into buffers """

        # Read data
        ark_dict, uid_new    = self.read_mats()
        senone_dict, uid_new = self.read_senones()

        if len(ark_dict) == 0:
            self.empty = True
            return

        self.uid = uid_new

        ids = sorted(ark_dict.keys())

        if not hasattr(self, 'offset_frames'):
            self.offset_frames = np.empty((0, ark_dict[ids[0]].shape[1]), np.float32)

        if not hasattr(self, 'offset_senones'):
            self.offset_senones = np.empty((0, senone_dict[ids[0]].shape[1]), np.float32)
        # Create frame buffer
        mats = [ark_dict[i] for i in ids]
        mats2 = np.vstack(mats)
        mats2 = np.concatenate((self.offset_frames, mats2),axis=0)
        # Create senone buffer
        mats_senone = [senone_dict[i] for i in ids]
        mats2_senone = np.vstack(mats_senone)
        mats2_senone = np.concatenate((self.offset_senones, mats2_senone), axis=0)
        # Put one batch into the offset frames
        cutoff = self.batch_size * self.buffer_size
        if mats2.shape[0] >= cutoff:
            self.offset_frames = mats2[cutoff:]
            mats2 = mats2[:cutoff]
            self.offset = self.offset_frames.shape[0]
            self.offset_senones = mats2_senone[cutoff:]
            mats2_senone = mats2_senone[:cutoff]

        mats2 = np.pad(
            array     = mats2,
            pad_width = ((self.context + self.out_frames // 2,),(0,)),
            mode      = 'edge')

        # Generate a random permutation of indexes
        if self.shuffle:
            self.indexes = np.random.permutation(mats2_senone.shape[0])
        else:
            self.indexes = np.arange(mats2_senone.shape[0])

        self.frame_buffer = mats2
        self.senone_buffer = mats2_senone

    def batchify(self, pretrain):
        """ Make a batch of frames and senones """

        batch_index = 0
        if self.empty:
            self._fill_buffer()
            self.empty = False
 
        while not self.empty:
            start = batch_index * self.batch_size
            end = min((batch_index+1) * self.batch_size, self.senone_buffer.shape[0])

            # Collect the data 
            frame_batch = np.stack((self.frame_buffer[i:i+self.out_frames+2*self.context,]
                for i in self.indexes[start:end]), axis = 0)

            if pretrain:
                label_batch = frame_batch[:,self.context:-self.context,0:40]
            else:
                label_batch = self.senone_buffer[self.indexes[start:end]]

            # Increment batch, and if necessary re-fill buffer
            batch_index += 1
            if batch_index * self.batch_size >= self.senone_buffer.shape[0]:
                batch_index = 0
                self._fill_buffer()

            yield frame_batch, label_batch

    def reset(self):
        self.uid = 0
        self.offset = 0
        self.empty = False

        self._fill_buffer()
