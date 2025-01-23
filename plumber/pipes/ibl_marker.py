from ..node import Pipe

import os
import torch
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from one.api import ONE
from brainbox.io.one import SessionLoader

from daart.data import DataGenerator
from daart.models import Segmenter
from daart.transforms import ZScore

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

        file = self.extract_marker_data(
            eid=eid, 
            mtype=self.mtype,
            l_thresh=self.l_thresh, 
            view=self.view, 
            paw=self.paw, 
            smooth=self.smooth, 
            path=self.path)
        
        return self.create_data_generator(
            sess_id=sess, 
            input_file=file, 
            path=self.path)

    # Function to extract and/or smooth ibl data
    def extract_marker_data(self, eid, mtype, l_thresh, view, paw, smooth, path):
        #sess_id = dropbox_marker_paths[eid]
        # type = 'DLC' or 'LP'
        
        sl = SessionLoader(one=one, eid=eid)

        if mtype == 'DLC':
            # Load the pose data
            sl.load_pose(likelihood_thr=l_thresh, views=[view])
            times = sl.pose[f'{view}Camera'].times.to_numpy()
            markers = sl.pose[f'{view}Camera'].loc[:, (f'{paw}_x', f'{paw}_y')].to_numpy()  
        
        elif mtype == 'LP':
            d = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'], query_type='remote')
            times = d['times']
            markers = d['lightningPose'].loc[:, (f'{paw}_x', f'{paw}_y')].to_numpy()

        else:
            raise ValueError('mtype must be either "DLC" or "LP"')


        # Load wheel data
        sl.load_wheel()
        wh_times = sl.wheel.times.to_numpy()
        wh_vel_oversampled = sl.wheel.velocity.to_numpy()
        
        # Resample wheel data at marker times
        interpolator = interp1d(wh_times, wh_vel_oversampled, fill_value='extrapolate')
        wh_vel = interpolator(times)

        if smooth == True:
            # Smooth the marker data
            markers[:, 0] = self.smooth_interpolate_signal_sg(markers[:, 0], window=7)
            markers[:, 1] = self.smooth_interpolate_signal_sg(markers[:, 1], window=7)
            
        # Process the data
        markers_comb = np.hstack([markers, wh_vel[:, None]])
        velocity = np.vstack([np.array([0, 0, 0]), np.diff(markers_comb, axis=0)])
        markers_comb = np.hstack([markers_comb, velocity])
        markers_z = (markers_comb - np.mean(markers_comb, axis=0)) / np.std(markers_comb, axis=0)
        feature_names = ['paw_x_pos', 'paw_y_pos', 'wheel_vel', 'paw_x_vel', 'paw_y_vel', 'wheel_acc']
        df = pd.DataFrame(markers_z, columns=feature_names)

        if smooth == False:
            df.to_csv(path + f'/marker_features/{eid}_features.csv')
            markers_file = path + f'/marker_features/{eid}_features.csv'
        else:
            df.to_csv(path + f'/marker_features/{eid}_features_smooth.csv')
            markers_file = path + f'/marker_features/{eid}_features_smooth.csv'
        return markers_file
    
        # Functions to Perform Smoothing
    def non_uniform_savgol(self, x, y, window, polynom):
        if len(x) != len(y):
            raise ValueError('"x" and "y" must be of the same size')
        if len(x) < window:
            raise ValueError('The data size must be larger than the window size')
        if type(window) is not int:
            raise TypeError('"window" must be an integer')
        if window % 2 == 0:
            raise ValueError('The "window" must be an odd integer')
        if type(polynom) is not int:
            raise TypeError('"polynom" must be an integer')
        if polynom >= window:
            raise ValueError('"polynom" must be less than "window"')

        half_window = window // 2
        polynom += 1

        A = np.empty((window, polynom))
        tA = np.empty((polynom, window))
        t = np.empty(window)
        y_smoothed = np.full(len(y), np.nan)

        for i in range(half_window, len(x) - half_window, 1):
            x0 = x[i]
            A[:, 0] = 1.0
            for j in range(1, polynom):
                A[:, j] = (x[i - half_window:i + half_window + 1] - x0)**j
            tA = A.T
            t = np.linalg.inv(tA @ A) @ tA @ y[i - half_window:i + half_window + 1]
            y_smoothed[i] = t[0]

        y_smoothed[:half_window + 1] = y[:half_window + 1]
        y_smoothed[-half_window - 1:] = y[-half_window - 1:]

        return y_smoothed

    # Function to smooth and interpolate the marker signals
    def smooth_interpolate_signal_sg(self, signal, window=31, order=3, interp_kind='linear'):

        signal_noisy_w_nans = np.copy(signal)
        timestamps = np.arange(signal_noisy_w_nans.shape[0])
        good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
        if len(good_idxs) < window:
            print('Not enough non-nan indices to filter; returning original signal')
            return signal_noisy_w_nans
        signal_smooth_nonans = self.non_uniform_savgol(
            timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
        signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
        signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
        interpolater = interp1d(
            timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')
        signal = interpolater(timestamps)
        return signal
    
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