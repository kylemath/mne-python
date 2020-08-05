import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

import mne
from mne.datasets.testing import data_path

###############################################################################
# ANC dataset
boxy_data_folder = mne.datasets.boxy_example.data_path()
boxy_raw_dir = os.path.join(boxy_data_folder, 'Participant-1')

# Load AC and Phase data
raw_ANC_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
raw_ANC_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
raw_ANC_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

# Plot the raw data
scalings = dict(fnirs_cw_amplitude=2e2, fnirs_fd_phase=2e2)

raw_ANC_dc.plot(n_channels=10, duration=20, scalings=scalings,
                show_scrollbars=True)
raw_ANC_ac.plot(n_channels=10, duration=20, scalings=scalings,
                show_scrollbars=True)
raw_ANC_ph.plot(n_channels=10, duration=20, scalings=scalings,
                show_scrollbars=True)
###############################################################################

###############################################################################
# Shortened ANC dataset (3 seconds)
boxy_raw_dir = os.path.join(data_path(download=False),
                            'BOXY', 'boxy_short_recording')

mne_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
mne_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
mne_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

# Plot the raw data
scalings = dict(fnirs_cw_amplitude=2e2, fnirs_fd_phase=2e2)

mne_dc.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)
mne_ac.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)
mne_ph.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)

# Load p_pod data if we want to compare.
p_pod_dir = os.path.join(data_path(download=False),
                         'BOXY', 'boxy_short_recording',
                         'boxy_p_pod_files', '1anc071a_001.mat')
ppod_data = spio.loadmat(p_pod_dir)

ppod_ac = np.transpose(ppod_data['ac'])
ppod_dc = np.transpose(ppod_data['dc'])
ppod_ph = np.transpose(ppod_data['ph'])

# Plot and compare MNE and P_Pod
# AC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(mne_ac.times, ppod_ac[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('P_POD Raw Data AC')

axes[1].plot(mne_ac.times, mne_ac._data[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('MNE Raw Data AC')

# DC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(mne_dc.times, ppod_dc[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('P_POD Raw Data DC')

axes[1].plot(mne_dc.times, mne_dc._data[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('MNE Raw Data DC')

# Ph
i_chan = 0
ylim = [0, 400]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(mne_ph.times, ppod_ph[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('P_POD Raw Data Ph')

axes[1].plot(mne_ph.times, mne_ph._data[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('MNE Raw Data Ph')

# Compare MNE loaded data to p_pod loaded data.
thresh = 1e-10
assert (abs(ppod_ac - mne_ac._data) <= thresh).all()
assert (abs(ppod_dc - mne_dc._data) <= thresh).all()
assert (abs(ppod_ph - mne_ph._data) <= thresh).all()
###############################################################################

###############################################################################
# Load parsed and unparsed files.

# Unparsed digaux.
boxy_raw_dir = os.path.join(data_path(download=False),
                            'BOXY', 'boxy_digaux_recording', 'unparsed')

unp_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
unp_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
unp_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

# Plot the raw data
scalings = dict(fnirs_cw_amplitude=2e2, fnirs_fd_phase=2e2)

unp_dc.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)
unp_ac.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)
unp_ph.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)

# Parsed digaux.
boxy_raw_dir = os.path.join(data_path(download=False),
                            'BOXY', 'boxy_digaux_recording', 'parsed')

par_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
par_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
par_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

# Plot the raw data
scalings = dict(fnirs_cw_amplitude=2e2, fnirs_fd_phase=2e2)

par_dc.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)
par_ac.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)
par_ph.plot(n_channels=10, duration=20, scalings=scalings,
            show_scrollbars=True)

# Plot and compare MNE parsed and unparsed
# AC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(par_ac.times, par_ac._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('Parsed Raw Data AC')

axes[1].plot(unp_ac.times, unp_ac._data[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('Unparsed Raw Data AC')

# DC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(par_dc.times, par_dc._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('Parsed Raw Data DC')

axes[1].plot(unp_dc.times, unp_dc._data[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('Unparsed Raw Data DC')

# Ph
i_chan = 0
ylim = [0, 400]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(par_ph.times, par_ph._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('Parsed Raw Data Ph')

axes[1].plot(unp_ph.times, unp_ph._data[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('Unparsed Raw Data Ph')

# Load p_pod data if we want to compare.
p_pod_dir = os.path.join(data_path(download=False),
                         'BOXY', 'boxy_digaux_recording', 'p_pod',
                         'p_pod_digaux_unparsed.mat')
ppod_data = spio.loadmat(p_pod_dir)

ppod_ac = np.transpose(ppod_data['ac'])
ppod_dc = np.transpose(ppod_data['dc'])
ppod_ph = np.transpose(ppod_data['ph'])

# Plot and compare MNE and P_Pod
# Parsed
# AC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(par_ac.times, par_ac._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('MNE Parsed Raw Data AC')

axes[1].plot(par_ac.times, ppod_ac[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('P_POD Raw Data AC')

# DC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(par_dc.times, par_dc._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('MNE Parsed Raw Data DC')

axes[1].plot(par_dc.times, ppod_dc[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('P_POD Raw Data DC')

# Ph
i_chan = 0
ylim = [0, 400]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(par_ph.times, par_ph._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('MNE Parsed Raw Data Ph')

axes[1].plot(par_ph.times, ppod_ph[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('P_POD Raw Data Ph')

thresh = 1e-10
assert (abs(ppod_ac - unp_ac._data) <= thresh).all()
assert (abs(ppod_dc - unp_dc._data) <= thresh).all()
assert (abs(ppod_ph - unp_ph._data) <= thresh).all()

# Plot and compare MNE and P_Pod
# Unparsed
# AC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(unp_ac.times, unp_ac._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('MNE Unparsed Raw Data AC')

axes[1].plot(unp_ac.times, ppod_ac[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('P_POD Raw Data AC')

# DC
i_chan = 0
ylim = [-100, 100]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(unp_dc.times, unp_dc._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('MNE Unparsed Raw Data DC')

axes[1].plot(unp_dc.times, ppod_dc[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('P_POD Raw Data DC')

# Ph
i_chan = 0
ylim = [0, 400]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

axes[0].plot(unp_ph.times, unp_ph._data[i_chan, :], 'r')
axes[0].set_ylim(ylim)
axes[0].set_ylabel('\u03BCV')
axes[0].set_title('MNE Unparsed Raw Data Ph')

axes[1].plot(unp_ph.times, ppod_ph[i_chan, :], 'b')
axes[1].set_ylim(ylim)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('\u03BCV')
axes[1].set_title('P_POD Raw Data Ph')

thresh = 1e-10
assert (abs(unp_dc._data - par_dc._data) == 0).all()
assert (abs(unp_ac._data - par_ac._data) == 0).all()
assert (abs(unp_ph._data - par_ph._data) == 0).all()
###############################################################################
