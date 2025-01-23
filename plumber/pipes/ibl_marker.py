from ..node import Pipe

import os
import torch
import yaml

from one.api import ONE

from daart.data import DataGenerator
from daart.models import Segmenter
from daart.transforms import ZScore

from .ibl_utils import extract_marker_data

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'

# Connect to the IBL database
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

class IBLMarkerExtractor(Pipe):

    def __init__(self, label: str, path, smooth, mtype = "DLC") -> None:
        super().__init__(label)
        self.path = path
        self.l_thresh = 0.0
        self.view = 'left'
        self.paw = 'paw_r'
        self.smooth = smooth
        self.mtype = mtype

    def pipe(self, inputs):
        eid = inputs["eid_stream"]["eid"]
        sess = inputs["eid_stream"]["sess"]

        file = extract_marker_data(
            one=one,
            eid=eid, 
            mtype=self.mtype,
            l_thresh=self.l_thresh, 
            view=self.view, 
            paw=self.paw, 
            path=self.path)

        return self.create_data_generator(
            sess_id=sess, 
            input_file=file, 
            path=self.path)

    # Function to create data generator
    def create_data_generator(self, sess_id, input_file, path):

        model_dir = path + '/daart_models'
        
        hparams_file = os.path.join(model_dir, 'hparams.yaml')
        hparams = yaml.safe_load(open(hparams_file, 'rb'))
        
        signals = ['markers']
        transforms = [ZScore()]
        paths = [input_file]

        return DataGenerator(
            [sess_id], [signals], [transforms], [paths],
            batch_size=hparams['batch_size'], sequence_length=hparams['sequence_length'], sequence_pad=hparams['sequence_pad'],
            trial_splits=hparams['trial_splits'], input_type=hparams['input_type'],
            device=device
        )