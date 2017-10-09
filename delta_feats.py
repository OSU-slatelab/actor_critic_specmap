"""
Creates delta feature ops 

Author: Deblin Bagchi
Date: Fall 2017
"""

import tensorflow as tf
import numpy as np

class DeltaFeatures:
    def __init__(order=2, window=2):
        assert order >=0 && order < 1000, "invalid order specified"
        assert window >=0 && window < 1000, "invalid window specified"
        scales = np.zeros(order+1)
        scales[0] = 1.0
        for (i = 1; i <= order; i++) 
            prev_scales = scales[i-1]
            cur_scales = scales[i]
            _window = window; 
            assert _window!=0, "invalid window size" 
            prev_offset = (prev_scales.Dim()-1)/2
            cur_offset = prev_offset + _window;
            cur_scales.Resize(prev_scales.Dim() + 2*window);  

        normalizer = 0.0;
        for (j = -window; j <= window; j++)
            normalizer += j*j;
        for (k = -prev_offset; k <= prev_offset; k++)
            cur_scales(j+k+cur_offset) +=
            static_cast<BaseFloat>(j) * prev_scales(k+prev_offset);
        cur_scales.Scale(1.0 / normalizer);
    
    def process(input_feats,frame):
   KALDI_ASSERT(frame < input_feats.NumRows());
  int32 num_frames = input_feats.NumRows(),
      feat_dim = input_feats.NumCols();
  KALDI_ASSERT(static_cast<int32>(output_frame->Dim()) == feat_dim * (opts_.order+1));
  output_frame->SetZero();
  for (int32 i = 0; i <= opts_.order; i++) {
    const Vector<BaseFloat> &scales = scales_[i];
    int32 max_offset = (scales.Dim() - 1) / 2;
    SubVector<BaseFloat> output(*output_frame, i*feat_dim, feat_dim);
    for (int32 j = -max_offset; j <= max_offset; j++) {
      // if asked to read
      int32 offset_frame = frame + j;

 void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features) {
  output_features->Resize(input_features.NumRows(),
                          input_features.NumCols()
                          *(delta_opts.order + 1));
  DeltaFeatures delta(delta_opts);
  for (int32 r = 0; r < static_cast<int32>(input_features.NumRows()); r++) {
    SubVector<BaseFloat> row(*output_features, r);
    delta.Process(input_features, r, &row);
  }
}




