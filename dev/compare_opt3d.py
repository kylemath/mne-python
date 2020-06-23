import os
import numpy as np
import re as re
import scipy.io as spio
from copy import deepcopy

import mne

crnt_dir = os.getcwd()

# load AC, DC, and Phase data.
boxy_data_folder = mne.datasets.boxy_example.data_path()
boxy_raw_dir = os.path.join(boxy_data_folder, 'Participant-1')

raw_intensity_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC',
                                        verbose=True).load_data()
raw_intensity_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC',
                                        verbose=True).load_data()
raw_intensity_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph',
                                        verbose=True).load_data()

# Get channel indices for our two montages.
mtg_a = [raw_intensity_ac.ch_names[i_index] for i_index, i_label
         in enumerate(raw_intensity_ac.info['ch_names'])
         if re.search(r'_D[1-8] ', i_label)]
mtg_b = [raw_intensity_ac.ch_names[i_index] for i_index, i_label
         in enumerate(raw_intensity_ac.info['ch_names'])
         if re.search(r'_D(9|1[0-6]) ', i_label)]

##############################################################################
# Compare our raw data between p_pod and python.
thresh = 1e-10  # Determine to which decimal place we will compare.

crnt_dir = os.getcwd()
mtg_list = [mtg_a, mtg_b]
dtype_list = ['ac', 'dc', 'ph']

for m_num, i_mtg in enumerate(['a', 'b']):
    for b_num, i_blk in enumerate(['001', '002']):

        # Load our p_pod files.
        filename = os.path.join(crnt_dir, 'dev', 'matlab_files', 'p_pod_files',
                                '1anc071' + i_mtg + '.' + i_blk +
                                'raw_data.mat')
        ppod_raw = spio.loadmat(filename)

        ppod_dict = dict(ac=ppod_raw['ac'],
                         dc=ppod_raw['dc'],
                         ph=ppod_raw['ph'])

        # Grab the appropriate python data based on which block we're on.
        data_len = int((np.shape(raw_intensity_ac._data)[1])/2)
        py_dict = dict(ac=(raw_intensity_ac.copy().pick(mtg_list[m_num]).
                           _data[:, data_len * b_num:data_len * (b_num + 1)]),
                       dc=(raw_intensity_dc.copy().pick(mtg_list[m_num]).
                           _data[:, data_len * b_num:data_len * (b_num + 1)]),
                       ph=(raw_intensity_ph.copy().pick(mtg_list[m_num]).
                           _data[:, data_len * b_num:data_len * (b_num + 1)]))

        # Loop through our data types.
        for dtype in dtype_list:

            # Get our python data.
            py_data = deepcopy(np.transpose(py_dict[dtype]))

            # Swap channels back to original.
            for i_chan in range(0, np.size(py_data, axis=1), 2):
                py_data[:, [i_chan, i_chan + 1]] = (py_data[:,
                                                    [i_chan + 1, i_chan]])

            # Compare our raw data between p_pod and python.
            test = (abs(ppod_dict[dtype] - py_data) <= thresh)
            if test.all():
                print("SUCCESS! " + dtype + " for montage " + i_mtg +
                      " and block " + i_blk + " match up to " + str(thresh) +
                      " decimal points!")
            else:
                print("FAILURE! " + dtype + " for montage " + i_mtg +
                      " and block " + i_blk + " DOES NOT match up to " +
                      str(thresh) + " decimal points!")

##############################################################################
# Now let's get our epochs and compare to p_pod.

# Montage A Events.
mtg_a_events = (mne.find_events(raw_intensity_ac.copy().pick(
                mtg_a + ['Markers a']), stim_channel=['Markers a']))

mtg_a_event_dict = {'Montage_A/Event_1': 1,
                    'Montage_A/Event_2': 2}

# Montage B Events.
mtg_b_events = (mne.find_events(raw_intensity_ac.copy().pick(
                mtg_b + ['Markers b']), stim_channel=['Markers b']))

mtg_b_event_dict = {'Montage_B/Event_1': 1,
                    'Montage_B/Event_2': 2}

reject_criteria = None
tmin, tmax = -0.032, 0.208

# DC epochs.
epochs_dc_a = mne.Epochs(raw_intensity_dc.copy().pick(mtg_a), mtg_a_events,
                         event_id=mtg_a_event_dict, tmin=tmin,
                         tmax=tmax, reject=reject_criteria,
                         reject_by_annotation=False, proj=True,
                         baseline=None, preload=True, detrend=None,
                         verbose=True, event_repeated='drop')

epochs_dc_b = mne.Epochs(raw_intensity_dc.copy().pick(mtg_b), mtg_b_events,
                         event_id=mtg_b_event_dict, tmin=tmin,
                         tmax=tmax, reject=reject_criteria,
                         reject_by_annotation=False, proj=True,
                         baseline=None, preload=True, detrend=None,
                         verbose=True, event_repeated='drop')

# Swap channels back to original.
for i_epoch in range(np.size(epochs_dc_a, axis=0)):
    for i_chan in range(0, np.size(epochs_dc_a, axis=1), 2):
        epochs_dc_a._data[i_epoch, [i_chan, i_chan + 1], :] =\
         epochs_dc_a._data[i_epoch, [i_chan + 1, i_chan], :]

for i_epoch in range(np.size(epochs_dc_b, axis=0)):
    for i_chan in range(0, np.size(epochs_dc_b, axis=1), 2):
        epochs_dc_b._data[i_epoch, [i_chan, i_chan + 1], :] =\
         epochs_dc_b._data[i_epoch, [i_chan + 1, i_chan], :]

# AC epochs.
epochs_ac_a = mne.Epochs(raw_intensity_ac.copy().pick(mtg_a), mtg_a_events,
                         event_id=mtg_a_event_dict, tmin=tmin,
                         tmax=tmax, reject=reject_criteria,
                         reject_by_annotation=False, proj=True,
                         baseline=None, preload=True, detrend=None,
                         verbose=True, event_repeated='drop')

epochs_ac_b = mne.Epochs(raw_intensity_ac.copy().pick(mtg_b), mtg_b_events,
                         event_id=mtg_b_event_dict, tmin=tmin,
                         tmax=tmax, reject=reject_criteria,
                         reject_by_annotation=False, proj=True,
                         baseline=None, preload=True, detrend=None,
                         verbose=True, event_repeated='drop')

# Swap channels back to original.
for i_epoch in range(np.size(epochs_ac_a, axis=0)):
    for i_chan in range(0, np.size(epochs_ac_a, axis=1), 2):
        epochs_ac_a._data[i_epoch, [i_chan, i_chan + 1], :] =\
         epochs_ac_a._data[i_epoch, [i_chan + 1, i_chan], :]

for i_epoch in range(np.size(epochs_ac_b, axis=0)):
    for i_chan in range(0, np.size(epochs_ac_b, axis=1), 2):
        epochs_ac_b._data[i_epoch, [i_chan, i_chan + 1], :] =\
         epochs_ac_b._data[i_epoch, [i_chan + 1, i_chan], :]

# Phase Epochs.
epochs_ph_a = mne.Epochs(raw_intensity_ph.copy().pick(mtg_a), mtg_a_events,
                         event_id=mtg_a_event_dict, tmin=tmin,
                         tmax=tmax, reject=reject_criteria,
                         reject_by_annotation=False, proj=True,
                         baseline=None, preload=True, detrend=None,
                         verbose=True, event_repeated='drop')

epochs_ph_b = mne.Epochs(raw_intensity_ph.copy().pick(mtg_b), mtg_b_events,
                         event_id=mtg_b_event_dict, tmin=tmin,
                         tmax=tmax, reject=reject_criteria,
                         reject_by_annotation=False, proj=True,
                         baseline=None, preload=True, detrend=None,
                         verbose=True, event_repeated='drop')

# Swap channels back to original.
for i_epoch in range(np.size(epochs_ph_a, axis=0)):
    for i_chan in range(0, np.size(epochs_ph_a, axis=1), 2):
        epochs_ph_a._data[i_epoch, [i_chan, i_chan + 1], :] =\
         epochs_ph_a._data[i_epoch, [i_chan + 1, i_chan], :]

for i_epoch in range(np.size(epochs_ph_b, axis=0)):
    for i_chan in range(0, np.size(epochs_ph_b, axis=1), 2):
        epochs_ph_b._data[i_epoch, [i_chan, i_chan + 1], :] =\
         epochs_ph_b._data[i_epoch, [i_chan + 1, i_chan], :]

# Now compare epochs.
thresh = 1e-10  # Determine to which decimal place we will compare.

crnt_dir = os.getcwd()
mtg_list = [mtg_a, mtg_b]
dtype_list = ['ac', 'dc', 'ph']
epoch_dict = dict(ac_a=epochs_ac_a,
                  ac_b=epochs_ac_b,
                  dc_a=epochs_dc_a,
                  dc_b=epochs_dc_b,
                  ph_a=epochs_ph_a,
                  ph_b=epochs_ph_b)

for m_num, i_mtg in enumerate(['a', 'b']):

    # Load our p_pod files, both blocks.
    filename = os.path.join(crnt_dir, 'dev', 'matlab_files', 'p_pod_files',
                            '1anc071' + i_mtg + '.001' +
                            'epoch_data.mat')
    ppod_epoch1 = spio.loadmat(filename)

    filename = os.path.join(crnt_dir, 'dev', 'matlab_files', 'p_pod_files',
                            '1anc071' + i_mtg + '.002' +
                            'epoch_data.mat')
    ppod_epoch2 = spio.loadmat(filename)

    # Create an empy dictionary.
    ppod_dict = dict()

    # Loop through our data types.
    for dtype in dtype_list:

        # Create empty array to store our combined blocks.
        epoch_num = (len(ppod_epoch1[dtype + '_epochs']) +
                     len(ppod_epoch2[dtype + '_epochs']))
        ppod_epochs = np.zeros((epoch_num, 80, 16))

        # Loop through both blocks.
        for epoch_num, i_epoch in enumerate(ppod_epoch1[dtype + '_epochs']):
            ppod_epochs[epoch_num, :, :] = np.transpose(i_epoch[0])

        last_blk = epoch_num
        for epoch_num, i_epoch in enumerate(ppod_epoch2[dtype + '_epochs']):
            ppod_epochs[epoch_num + last_blk + 1, :, :] =\
                np.transpose(i_epoch[0])

        py_epochs = epoch_dict[dtype + '_' + i_mtg]

        # Compare our epochs between p_pod and python.
        test = (abs(ppod_epochs - py_epochs) <= thresh)
        if test.all():
            print("SUCCESS! Epochs for " + dtype + " for montage " + i_mtg +
                  " match up to " + str(thresh) +
                  " decimal points!")
        else:
            print("FAILURE! Epoch for " + dtype + " for montage " + i_mtg +
                  " DOES NOT match up to " +
                  str(thresh) + " decimal points!")
##############################################################################
# Finally we will compare our evoked activity to p_pod.

# Get our python data.
evoked_dc_a1 = epochs_dc_a['Montage_A/Event_1'].average()
evoked_dc_a2 = epochs_dc_a['Montage_A/Event_2'].average()

evoked_dc_b1 = epochs_dc_b['Montage_B/Event_1'].average()
evoked_dc_b2 = epochs_dc_b['Montage_B/Event_2'].average()

evoked_ac_a1 = epochs_ac_a['Montage_A/Event_1'].average()
evoked_ac_a2 = epochs_ac_a['Montage_A/Event_2'].average()

evoked_ac_b1 = epochs_ac_b['Montage_B/Event_1'].average()
evoked_ac_b2 = epochs_ac_b['Montage_B/Event_2'].average()

evoked_ph_a1 = epochs_ph_a['Montage_A/Event_1'].average()
evoked_ph_a2 = epochs_ph_a['Montage_A/Event_2'].average()

evoked_ph_b1 = epochs_ph_b['Montage_B/Event_1'].average()
evoked_ph_b2 = epochs_ph_b['Montage_B/Event_2'].average()

evoke_dict = dict(ac_a1=evoked_ac_a1,
                  ac_a2=evoked_ac_a2,
                  ac_b1=evoked_ac_b1,
                  ac_b2=evoked_ac_b2,
                  dc_a1=evoked_dc_a1,
                  dc_a2=evoked_dc_a2,
                  dc_b1=evoked_dc_b1,
                  dc_b2=evoked_dc_b2,
                  ph_a1=evoked_ph_a1,
                  ph_a2=evoked_ph_a2,
                  ph_b1=evoked_ph_b1,
                  ph_b2=evoked_ph_b2)

# Loop through our montages.
for m_num, i_mtg in enumerate(['a', 'b']):

    # Load in our p_pod files.
    filename = os.path.join(crnt_dir, 'dev', 'matlab_files', 'p_pod_files',
                            '1anc071' + i_mtg + '.002' +
                            'evoke_data.mat')
    ppod_evoke_all = spio.loadmat(filename)

    # Loop through our data types.
    for dtype in dtype_list:

        # Just get the relevant p_pod data.
        ppod_evoke = deepcopy(ppod_evoke_all['a' + dtype]).swapaxes(0, 2)

        # Grab the relevant python data, and combine blocks.
        py_evoke = np.zeros(np.shape(ppod_evoke))
        py_evoke[0, :, :] = evoke_dict[dtype + "_" + i_mtg + "1"]._data
        py_evoke[1, :, :] = evoke_dict[dtype + "_" + i_mtg + "2"]._data

        # Compare p_pod and python evoked.
        test = (abs(ppod_evoke - py_evoke) <= thresh)
        if test.all():
            print("SUCCESS! Evoked for " + dtype + " for montage " + i_mtg +
                  " match up to " + str(thresh) +
                  " decimal points!")
        else:
            print("FAILURE! Evoked for " + dtype + " for montage " + i_mtg +
                  " DOES NOT match up to " +
                  str(thresh) + " decimal points!")
