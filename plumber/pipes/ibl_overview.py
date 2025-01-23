from ..node import Pipe
from .ibl_utils import read_markers

# Core imports
import numpy as np
import os
import pandas as pd

# Science imports
from scipy.interpolate import interp1d
from scipy import stats

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

# Video imports
import seaborn as sns
import cv2

# IBL imports
import ibllib.io.video as vidio
from brainbox.io.one import SessionLoader
from one.api import ONE

# Connect to the IBL database
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

model_to_use = 'e_mode'
silence_window = 5
frame_counter_duration = 20

l_thresh = 0.0
view = 'left'
paw = 'paw_r'
colors = ['red', 'blue', 'green', 'purple']
state_labels = ['Still', 'Move', 'Wheel Turn', 'Groom']
state_keys = {1: 'Still', 2: 'Move', 3: 'Wheel Turn', 4: 'Groom'}
state_map = np.vectorize(lambda x: state_keys[x])


class OverviewPlotter(Pipe):
    def __init__(self, label: str, artifact_path, mtype, eid) -> None:
        super().__init__(label)
        self.mtype = mtype
        self.eid = eid
        self.artifact_path = artifact_path
        self.export_path = self.artifact_path + '/session_overviews/'
        # Make this directory if it doesn't exist
        os.makedirs(self.export_path, exist_ok=True)
    
    def pipe(self, inputs):
        # print(inputs)
        # self.eid = inputs["eid_stream"]["eid"]
        self.preds = inputs["ens"]

        vid_url = vidio.url_from_eid(self.eid, one=one)[view]
        fps, frame = get_video_frames(vid_url)

        # Get the data
        marker_data, cam_times = extract_marker_data(self.eid, self.mtype)
        data_df, tframes = gen_data_df(marker_data, self.preds, cam_times)
        sd_df, durations = duration_data(data_df, tframes)
        interval_df = interval_data(data_df, self.eid)
        er, vr = raster_data(interval_df, data_df)
        still_to_wheel_turn_frames, wheel_turn_to_still_frames, still_to_move_frames, move_to_still_frames = wheel_data(data_df)
        
        fig = plt.figure(figsize=(25, 12), dpi=300)  # Adjust the figure size as needed
        gs = gridspec.GridSpec(
            5, 
            9, 
            figure=fig, 
            width_ratios=[1, 1, 1, 1, 0.5, 1, 1, 1, 1], 
            wspace=0.35, 
            hspace=0.4
        )  # Create an 8x4 grid
    
        # Plots
        plot_paws_and_speed(gs, fig, data_df, frame, self.eid, cbar=False)
        plot_state_durations(gs, fig, durations)
        plot_total_duration(gs, fig, durations)
        plot_raster(er, vr, gs, fig, xlim=120, fps=fps)
        plot_vert_info(gs, fig, interval_df["choice_data"], interval_df["trial_duration"])
        plot_wheel_speed(gs, fig, still_to_wheel_turn_frames, wheel_turn_to_still_frames, data_df, fps)
        plot_paw_speed(gs, fig, still_to_wheel_turn_frames, wheel_turn_to_still_frames, still_to_move_frames, move_to_still_frames, data_df, fps)
        plt_ens_var_hist(gs, fig, data_df)
        fig.suptitle(f"Overview of eid: {self.eid}", fontsize=14, y=0.95)

        # Save the figure to the export path
        output_path = self.export_path + f'{self.eid}_overview.jpg'
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

        return {"eid": self.eid, "output_path": output_path}

####################################################
# Data extraction functions
####################################################

# Function to extract and/or smooth ibl data
def get_video_frames(vid_url):
    # read the video frames
    cap = cv2.VideoCapture(vid_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()

    # Release the VideoCapture object
    cap.release()

    return fps, np.array(frame)

def extract_transition_data_with_window(frames, data, data_col, pre_window=30, post_window=50):
    transition_data = []
    for frame in frames:
        if frame - pre_window >= 0 and frame + post_window < len(data):  # Ensure bounds
            transition_data.append(data.iloc[frame - pre_window:frame + post_window][data_col].values)
    return np.array(transition_data)

def extract_marker_data(eid, mtype):
    #sess_id = dropbox_marker_paths[eid]

    # Load the pose data
    sl = SessionLoader(one=one, eid=eid)

    # Load the DLC data
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
    feature_names = ['paw_x_pos', 'paw_y_pos', 'wheel_vel', 'paw_x_vel', 'paw_y_vel', 'wheel_acc']
    df = pd.DataFrame(markers_comb, columns=feature_names)
    return df, times

def gen_data_df(marker_data, preds, cam_times):
    clip_length = preds.shape[1]

    data_df = marker_data.iloc[:clip_length, :].copy()

    # Calculate the ensemble mode for each frame across the models
    ensemble_mode = stats.mode(preds, axis=0).mode

    # Calculate variance for each frame across the models
    ensemble_variance = stats.variation(preds, axis=0)

    # calculate the paw speed
    data_df['paw_speed'] = np.sqrt(data_df['paw_x_vel']**2 + data_df['paw_y_vel']**2)

    # calculate the wheel speed
    data_df['wheel_speed'] = data_df['wheel_vel'].abs()

    # Add the inferred states to the data
    preds_labelled = state_map(preds)
    for i in range(0, 5):
        data_df[f'mod_{i+1}'] = preds_labelled[i]

    # add them all do the data frame
    data_df['e_mode'] = state_map(ensemble_mode)
    data_df['e_var'] = ensemble_variance
    data_df['frame_id'] = np.arange(0, len(data_df))

    # Put camera times in there
    data_df['cam_times'] = cam_times[:clip_length]

    # organize the columns:
    cols = ['paw_x_pos', 'paw_y_pos', 'paw_x_vel', 'paw_y_vel', 'paw_speed',
            'wheel_vel', 'wheel_speed', 'wheel_acc', 'cam_times',
            'mod_1', 'mod_2', 'mod_3', 'mod_4', 'mod_5', 'e_mode', 'e_var', 'frame_id']

    ####################
    # Silencing the +-5 frames around the transitions
    model_to_use = 'e_mode'
    any_transition_frames = data_df[
        (data_df[model_to_use].shift() != data_df[model_to_use])
    ].index

    # Make it so that +- 5 frames around the transition frames are also set to 0
    for noise_frame in any_transition_frames:
        data_df.loc[noise_frame - silence_window:noise_frame + silence_window, 'e_var'] = 0
    ####################

    # reorder the columns
    data_df = data_df[cols]

    return data_df, any_transition_frames

def duration_data( data_df, any_transition_frames):
    sd_df = data_df[["e_mode", "cam_times", "frame_id"]].copy()

    # Step 0: Lets filter out the noise frames that we dont want to consider
    # Make it so that +- 5 frames around the transition frames are also set to 
    # whatever the emode was at -6 frames
    for noise_frame in any_transition_frames:
        last_persisted_state_idx = noise_frame - silence_window - 1
        if last_persisted_state_idx < 0: last_persisted_state_idx = 0
        sd_df.loc[noise_frame - silence_window:noise_frame + silence_window, 'e_mode'] = sd_df.loc[last_persisted_state_idx, 'e_mode']

    # Step 1: Identify continuous state segments
    sd_df['group'] = (sd_df['e_mode'] != sd_df['e_mode'].shift()).cumsum()

    # Step 2: Calculate the duration of each frame
    sd_df['frame_duration'] = sd_df['cam_times'].shift(-1) - sd_df['cam_times']

    # Handle the NaN value for the last frame (no next frame to subtract)
    sd_df['frame_duration'] = sd_df['frame_duration'].fillna(1/60)

    # Step 3: Sum frame durations within each group to get state durations
    durations = sd_df.groupby('group').agg(
        e_mode=('e_mode', 'first'),
        duration=('frame_duration', 'sum')
    ).reset_index(drop=True)

    return sd_df, durations

def interval_data(data_df, eid):
    ct = data_df["cam_times"]
    tr = one.load_object(eid, 'trials')

    interval_df = pd.DataFrame(tr['intervals'], columns=['start', 'end'])
    interval_df["frame_start"] = np.searchsorted(ct, interval_df["start"]).astype(int)
    interval_df["frame_end"] = np.searchsorted(ct, interval_df["end"]).astype(int)
    interval_df["firstMovement_times"] = tr["firstMovement_times"]
    interval_df["frame_fmt"] = np.searchsorted(ct, interval_df["firstMovement_times"])
    interval_df["stimOn_times"] = tr["stimOn_times"]
    interval_df["stimOff_times"] = tr["stimOff_times"]
    interval_df["feedback_times"] = tr["feedback_times"]
    interval_df["trial_idx"] = np.arange(0, len(interval_df))
    interval_df["choice_data"] = (tr["feedbackType"]+ 1)/ 2

    interval_df["trial_duration"] = interval_df["end"] - interval_df["start"]

    # from data_df, e_var column, calculate the average variance within the frame_start and frame_end
    def calculate_avg_var(row):
        # Ensure frame_start and frame_end are valid integers
        frame_start = max(0, int(row["frame_start"]))
        frame_end = min(len(data_df) - 1, int(row["frame_end"]))  # Ensure within bounds
        # Return the mean of e_var for the valid range
        return data_df["e_var"].iloc[frame_start:frame_end].mean()
    
    # Apply the function to calculate avg_var
    interval_df["avg_var"] = interval_df.apply(calculate_avg_var, axis=1)

    return interval_df

def raster_data(interval_df, data_df):
    # temporary holder for the data
    ens_raster_holder = np.zeros((len(interval_df), (interval_df["frame_end"] - interval_df["frame_start"]).max()))
    ens_raster_holder[:] = np.nan
    var_raster_holder = np.zeros_like(ens_raster_holder)
    var_raster_holder[:] = np.nan

    for idx, row in interval_df.iterrows():
        # start = int(row["frame_start"])
        previous_space = 20
        start = int(row["frame_fmt"])
        end = int(row["frame_end"])
        class_to_int = {label: i for i, label in enumerate(state_labels)}
        ens_raster_holder[idx, :end - start + previous_space] = data_df["e_mode"][start - previous_space:end].map(class_to_int).values
        var_raster_holder[idx, :end - start + previous_space] = data_df["e_var"][start - previous_space:end].values

    return ens_raster_holder, var_raster_holder

def wheel_data(data_df):
    # Add a column to track when the state changes
    data_df['state_change'] = (data_df[model_to_use] != data_df[model_to_use].shift()).cumsum()

    # Compute the duration of each state block
    data_df['state_duration'] = data_df.groupby('state_change').cumcount() + 1

    # Reverse duration for the next state (counting backward)
    data_df['state_duration_next'] = data_df[::-1].groupby('state_change').cumcount() + 1
    # Ensure the previous state lasted at least 10 frames and the next state will last at least 10 frames
    still_to_wheel_turn_frames = data_df[
        (data_df[model_to_use].shift() == 'Still') & 
        (data_df[model_to_use] == 'Wheel Turn') & 
        (data_df['state_duration'].shift() >= frame_counter_duration) & 
        (data_df['state_duration_next'] >= frame_counter_duration)
    ].index

    wheel_turn_to_still_frames = data_df[
        (data_df[model_to_use].shift() == 'Wheel Turn') & 
        (data_df[model_to_use] == 'Still') & 
        (data_df['state_duration'].shift() >= frame_counter_duration) & 
        (data_df['state_duration_next'] >= frame_counter_duration)
    ].index

    still_to_move_frames = data_df[
        (data_df[model_to_use].shift() == 'Still') & 
        (data_df[model_to_use] == 'Move') & 
        (data_df['state_duration'].shift() >= frame_counter_duration) & 
        (data_df['state_duration_next'] >= frame_counter_duration)
    ].index

    move_to_still_frames = data_df[
        (data_df[model_to_use].shift() == 'Move') & 
        (data_df[model_to_use] == 'Still') & 
        (data_df['state_duration'].shift() >= frame_counter_duration) & 
        (data_df['state_duration_next'] >= frame_counter_duration)
    ].index

    return still_to_wheel_turn_frames, wheel_turn_to_still_frames, still_to_move_frames, move_to_still_frames

####################################################
# Plotting functions
####################################################

def plot_paws_and_speed(gs, fig, data_df, frame, eid_inferred, cbar=False):

    ############# LOCATION #############
    ax_paws = [(0, 0), (0, 1), (0, 2), (0, 3)]    # Axis for the Paws
    ############# LOCATION #############

    frame_marker_df = data_df.copy()

    sl = SessionLoader(one=one, eid=eid_inferred)
    sl.load_pose(likelihood_thr=l_thresh, views=[view])
    marker_px = sl.pose[f'{view}Camera'].loc[:, (f'{paw}_x', f'{paw}_y')].to_numpy()[:frame_marker_df.shape[0], :].T

    frame_marker_df['paw_x_pos'] = marker_px[0]
    frame_marker_df['paw_y_pos'] = marker_px[1]
    frame_marker_df['paw_speed_normalized'] = np.clip(frame_marker_df['paw_speed']*9, 0, 30)

    states = state_labels # ['Still', 'Move', 'Wheel Turn', 'Groom']


    # for ax, state in zip(axs.flatten(), states):
    for i, (row, col) in enumerate(ax_paws):
        
        ax = fig.add_subplot(gs[row, col])
        state = states[i]

        # Filter data for the current state
        state_data = frame_marker_df[frame_marker_df['e_mode'] == state]
        
        # Plot the base frame
        ax.imshow(frame)
        
        # Overlay points with alpha based on speed
        sc = ax.scatter(
            state_data['paw_x_pos'],
            state_data['paw_y_pos'],
            c=state_data['paw_speed_normalized'],
            cmap="plasma",
            alpha=1,
            marker='+',
            s=0.1,  # Marker size
            label=f'{state} paw positions'
        )
        
        ax.set_title(f"Paw Positions: {state}")
        ax.axis('off')  # Hide axes for better visualization

    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar_ax.axis('off')
    if cbar:
        # cbar = mpl.colorbar.ColorbarBase(cbar_ax, orientation='vertical', cmap='plasma',location='left')
        cbar = fig.colorbar(sc, ax=cbar_ax, orientation='vertical', location='right', fraction=0.99)
        cbar.set_label("Paw Speed")
        cbar.set_ticks([0, 6, 12, 18, 24, 30])
        cbar.set_ticklabels([0, 6, 12, 18, 24, "*30+"])
        cbar.ax.yaxis.set_label_position('left')

def plot_state_durations(gs, fig, durations):
    ############# LOCATION #############
    ax_durations = [(1, 0), (1, 1), (1, 2), (1, 3)]
    ############# LOCATION #############

    # Step 4: Plot histograms for each 'e_mode' state
    for idx, (row, col) in enumerate(ax_durations):

        ax = fig.add_subplot(gs[row, col])

        state_data = durations[durations['e_mode'] == state_labels[idx]]['duration']

        # ax.hist(state_data, bins=50, edgecolor='white', linewidth=0.5, color=colors[idx])
        sns.histplot(state_data, bins=50, color=colors[idx], ax=ax, log_scale=(True, False), fill=True)
        ax.set_title(f'Durations of {state_labels[idx]}')
        ax.set_xlabel('Duration (s)')
        if idx == 0: ax.set_ylabel('Frequency')
        else: ax.set_ylabel('')
        ax.set_yscale('log')

        # Set consistent formatting for log scale
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Suppress minor tick labels for clarity
        
        # Set consistent formatting for log scale
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Suppress minor tick labels for clarity

def plot_total_duration(gs, fig, durations):
    # Step 1: Calculate the total duration per state
    total_durations = durations.groupby('e_mode')['duration'].sum().reset_index()

    # Optional: Sort the states by total duration (descending order)
    total_durations.sort_values('duration', ascending=False, inplace=True)

    color_map = {state: color for state, color in zip(state_labels, colors)}
    total_durations['color'] = total_durations['e_mode'].map(color_map)

    # Create the subplot
    ax = fig.add_subplot(gs[4, 0:2])

    cols = total_durations['color'].tolist()
    bp = sns.barplot(data=total_durations, x='e_mode', y='duration', hue='e_mode', palette=cols, ax=ax, legend=False)

    for i in bp.containers:
        bp.bar_label(i, padding=3, fmt="%.1f", label_type='center')  # Add padding and format numbers

    # Add bar labels and ensure they stay within the chart
    # ax.set_title('Total Duration per State')
    ax.set_xlabel('States')
    ax.set_ylabel('Total Duration (s)')
    ax.set_yscale('log')

def plot_raster(er, vr, gs, fig, xlim = 120, fps = 60):
    # plt.style.use('default')
    colors = ['red', 'blue', 'green', 'purple']
    cmap = ListedColormap(colors)

    # fig, axs = plt.subplots(1,3, figsize=(20, 10), dpi=300)

    ens_ax = fig.add_subplot(gs[0:5, 5:7])
    var_ax = fig.add_subplot(gs[0:5, 7:9])

    # Convert frames to seconds
    x_ticks = np.linspace(-20 / fps, (xlim - 20) / fps, 10)  # 10 ticks, starting from -20 frames


    sns.heatmap(er, cmap=cmap, cbar=False, ax=ens_ax)
    ens_ax.set_title("Raster of Ensemble Mode States")
    ens_ax.set_xlabel("Time from first movement onset (s)")

    # yt = np.arange(len(er)) + 1
    # ens_ax.set_yticks(yt)
    # ens_ax.set_yticklabels(yt, rotation=0, fontsize=5)
    # Calculate the number of trials
    num_trials = len(er)

    # Set up to 20 evenly spaced ticks along the y-axis, always in integers
    max_ticks = 50
    if num_trials <= max_ticks:
        yt = np.arange(1, num_trials + 1)  # Show all ticks if trials are fewer than max_ticks
    else:
        yt = np.linspace(1, num_trials, max_ticks, dtype=int)  # 20 evenly spaced ticks

    ens_ax.set_yticks(yt)
    ens_ax.set_yticklabels(yt, rotation=0, fontsize=10)
    ens_ax.set_ylabel("")
    
    ens_ax.set_xlim(0, xlim)
    ens_ax.set_xticks(np.linspace(0, xlim, 10))  # 10 evenly spaced ticks
    ens_ax.set_xticklabels([f"{tick:.2f}s" for tick in x_ticks], rotation = 0)  # Format as seconds
    # vertical line at 20 for first movement onset
    ens_ax.axvline(x=20, color='black', linestyle='--')


    sns.heatmap(vr, cmap="gray", cbar=False, ax=var_ax, cbar_kws={"aspect": 100, "shrink": 1, "pad": 0.01})
    var_ax.axvline(x=20, color='white', linestyle='--')
    # sns.heatmap(vr, cmap="rocket", cbar=True, ax=var_ax,)
    var_ax.set_title("Raster of Ensemble Mode Variance")
    var_ax.set_xlabel("Time from first movement onset (s)")
    # var_ax.set_ylabel("Trial #")
    var_ax.set_yticks([])
    var_ax.set_xlim(0, xlim)

    # Set ticks and labels
    var_ax.set_xticks(np.linspace(0, xlim, 10))  # 10 evenly spaced ticks
    var_ax.set_xticklabels([f"{tick:.2f}s" for tick in x_ticks], rotation=0)  # Format as seconds

def plot_vert_info(gs, fig, correct_info, trial_duration, yticks = None):
    # Create a nested GridSpec to split the single column (gs[:, 4]) into smaller sub-columns
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 4], wspace=1.1)

    choice_ax = fig.add_subplot(inner_gs[0, 0])  # First sub-column

    # Binary information axis
    binary_cmap = ListedColormap(['black', 'white'])  # Red for wrong, green for correct
    binary_data = np.expand_dims(correct_info, axis=1)  # Reshape for heatmap
    sns.heatmap(binary_data, cmap=binary_cmap, cbar=False, ax=choice_ax)
    # choice_ax.set_title("Mouse Correctness")
    choice_ax.set_xticks([])  # Remove x-axis
    choice_ax.set_yticks([] if yticks is None else np.arange(len(yticks)))
    choice_ax.set_yticklabels([] if yticks is None else yticks, fontsize=5, rotation=0)
    choice_ax.set_ylabel("Incorrect Response", fontsize=12)

    duration_ax = fig.add_subplot(inner_gs[0, 1])
    duration_data = np.expand_dims(trial_duration, axis=1)
    sns.heatmap(duration_data, cmap="gray", cbar=False, ax=duration_ax)
    duration_ax.set_xticks([])
    duration_ax.set_yticks([] if yticks is None else np.arange(len(yticks)))
    duration_ax.set_yticklabels([] if yticks is None else yticks, fontsize=5, rotation=0)
    duration_ax.set_ylabel("Trial Duration", fontsize=12)

def plot_wheel_speed(gs, fig, still_to_wheel_turn_frames, wheel_turn_to_still_frames, data_df, fps):

    still_to_wheel_turn_data = extract_transition_data_with_window(still_to_wheel_turn_frames, data_df, data_col="wheel_speed")
    wheel_turn_to_still_data = extract_transition_data_with_window(wheel_turn_to_still_frames, data_df, data_col="wheel_speed")

    still_to_wheel_turn_avg = still_to_wheel_turn_data.mean(axis=0)
    wheel_turn_to_still_avg = wheel_turn_to_still_data.mean(axis=0)

    ax1 = fig.add_subplot(gs[2, 0:2])
    ax2 = fig.add_subplot(gs[2, 2:4])

    x = np.arange(-30,50,1)  # 100 frames
    # convert it to seconds using the frame rate
    x = x / fps
    alpha = 0.2

    for line in still_to_wheel_turn_data:
        ax1.plot(x, line, color="gray", alpha=alpha)
    ax1.plot(x, still_to_wheel_turn_avg, "k--", label="Still → Wheel Turn", linewidth=2)

    for line in wheel_turn_to_still_data:
        ax2.plot(x, line, color="gray", alpha=alpha)
    ax2.plot(x, wheel_turn_to_still_avg, "k--", label="Wheel Turn → Still", linewidth=2)

    for a in [ax1, ax2]:
        # a.set_title("Wheel Speed Transitions")
        a.set_xlabel("Time from Transition (seconds)")
        a.set_ylabel("Wheel Speed")
        a.legend(loc='upper right')
        a.grid()

def plot_paw_speed(gs, fig, still_to_wheel_turn_frames, wheel_turn_to_still_frames, still_to_move_frames, move_to_still_frames, data_df, fps):

    still_to_wheel_turn_data = extract_transition_data_with_window(still_to_wheel_turn_frames, data_df, data_col="paw_speed")
    wheel_turn_to_still_data = extract_transition_data_with_window(wheel_turn_to_still_frames, data_df, data_col="paw_speed")
    still_to_move_data = extract_transition_data_with_window(still_to_move_frames, data_df, data_col="paw_speed")
    move_to_still_data = extract_transition_data_with_window(move_to_still_frames, data_df, data_col="paw_speed")

    still_to_wheel_turn_avg = still_to_wheel_turn_data.mean(axis=0)
    wheel_turn_to_still_avg = wheel_turn_to_still_data.mean(axis=0)
    still_to_move_avg = still_to_move_data.mean(axis=0)
    move_to_still_avg = move_to_still_data.mean(axis=0)

    ax = fig.add_subplot(gs[3, 0:4])

    x = np.arange(-30,50,1)
    x = x / fps

    lines = [still_to_wheel_turn_avg, wheel_turn_to_still_avg, still_to_move_avg, move_to_still_avg]

    for i, line_name in enumerate(["Still → Wheel Turn", "Wheel Turn → Still", "Still → Move", "Move → Still"]):
        ax.plot(x, lines[i], label=line_name)

    ax.axvline(0, color='k', linestyle='--', label="State Transition")  # Highlight transition frame
    ax.set_xlabel("Time from Transition (s)")
    ax.set_ylabel("Paw Speed")
    ax.legend()
    ax.grid()

def plt_ens_var_hist(gs, fig, data_df):

    ax = fig.add_subplot(gs[4, 2:4])

    ax = sns.histplot(data_df["e_var"], bins=50, color='k', log_scale=False, fill=True)
    ax.set_xlabel("Ensemble Mode Variance")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')













































