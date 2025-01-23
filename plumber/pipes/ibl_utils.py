import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from brainbox.io.one import SessionLoader


def extract_marker_data(one, eid, mtype, l_thresh, view, paw, path):
    """
    Extract marker data from a session and save processed features to a CSV file.

    Parameters:
    one : ONE
        ONE instance to load data.
    eid : str
        Experiment ID.
    mtype : str
        Marker type, either 'DLC' or 'LP'.
    l_thresh : float
        Likelihood threshold for pose estimation.
    view : str
        Camera view, e.g., 'left', 'right'.
    paw : str
        Paw to extract data for, e.g., 'left_paw', 'right_paw'.
    path : str
        Path to save the output CSV file.

    Returns:
    str
        Path to the saved CSV file with processed marker features.

    """
    sl = SessionLoader(one=one, eid=eid)

    # Read marker data
    times, markers = read_markers(one, eid, sl, mtype, l_thresh, view, paw)

    # Load wheel data
    sl.load_wheel()
    wh_times = sl.wheel.times.to_numpy()
    wh_vel_oversampled = sl.wheel.velocity.to_numpy()
    
    # Resample wheel data at marker times
    interpolator = interp1d(wh_times, wh_vel_oversampled, fill_value='extrapolate')
    wh_vel = interpolator(times)

    # Process the data
    markers_comb = np.hstack([markers, wh_vel[:, None]])
    velocity = np.vstack([np.array([0, 0, 0]), np.diff(markers_comb, axis=0)])
    markers_comb = np.hstack([markers_comb, velocity])
    markers_z = (markers_comb - np.mean(markers_comb, axis=0)) / np.std(markers_comb, axis=0)
    feature_names = ['paw_x_pos', 'paw_y_pos', 'wheel_vel', 'paw_x_vel', 'paw_y_vel', 'wheel_acc']
    df = pd.DataFrame(markers_z, columns=feature_names)

    df.to_csv(path + f'/marker_features/{eid}_features.csv')
    markers_file = path + f'/marker_features/{eid}_features.csv'

    return markers_file

def read_markers(one, eid, sl, mtype, l_thresh, view, paw):

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
    
    return times, markers








#EOD