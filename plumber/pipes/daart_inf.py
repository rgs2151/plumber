from ..node import Pipe

import os
import torch
import yaml
import numpy as np

from daart.models import Segmenter

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'

class DaartInf(Pipe):

    def __init__(self, label: str, path, model_name) -> None:
        super().__init__(label)

        self.path = path

        # Load the Model
        model_dir = path + '/daart_models'
        model_file = os.path.join(model_dir, f'{model_name}.pt')

        hparams_file = os.path.join(model_dir, 'hparams.yaml')
        hparams = yaml.safe_load(open(hparams_file, 'rb'))

        self.model = Segmenter(hparams)
        self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.model.to(device)
        self.model.eval()


    def pipe(self, inputs):
        # Function to run inference
        data_gen = list(inputs.values())[0]
        tmp = self.model.predict_labels(data_gen, return_scores=True)
        probs = np.vstack(tmp['labels'][0])
        return np.argmax(probs, axis=1)
