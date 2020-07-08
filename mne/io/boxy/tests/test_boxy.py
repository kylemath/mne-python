# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
#
# License: BSD (3-clause)

import os
import re as re

import numpy as np
import scipy.io as spio
import pytest
from numpy.testing import assert_array_equal

import mne
from mne.datasets.testing import data_path, requires_testing_data
from mne.transforms import apply_trans, get_ras_to_neuromag_trans

# Load AC, DC, and Phase data.
boxy_raw_dir = os.path.join(data_path(download=False),
                            'BOXY', 'boxy_short_recording')

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

mtg_list = [mtg_a, mtg_b]

# Determine to which decimal place we will compare.
thresh = 1e-10


@requires_testing_data
@pytest.mark.parametrize('datatype,data', [('ac', raw_intensity_ac),
                                           ('dc', raw_intensity_dc),
                                           ('ph', raw_intensity_ph)])
def test_boxy_load(datatype, data):
    assert data._data.shape == (162, 376)
    assert data.info['sfreq'] == 62.5

    # Check channel names for montages and markers
    assert data.info['ch_names'][:4] == ["S1_D1 690", "S1_D1 830",
                                         "S2_D1 690", "S2_D1 830"]
    assert data.info['ch_names'][80:84] == ["S6_D9 690", "S6_D9 830",
                                            "S7_D9 690", "S7_D9 830"]
    assert data.info['ch_names'][160:162] == ["Markers a", "Markers b"]

    # Check wavelengths for montages
    assert data.info['chs'][0]['loc'][9] == 690
    assert data.info['chs'][1]['loc'][9] == 830
    assert data.info['chs'][2]['loc'][9] == 690
    assert data.info['chs'][3]['loc'][9] == 830
    assert data.info['chs'][80]['loc'][9] == 690
    assert data.info['chs'][81]['loc'][9] == 830
    assert data.info['chs'][82]['loc'][9] == 690
    assert data.info['chs'][83]['loc'][9] == 830

    # Check our markers
    all_events = mne.find_events(data, stim_channel=['Markers a'])
    assert np.unique(all_events[:, 2]).tolist() == [1, 2, 1000, 2000]
    all_events = mne.find_events(data, stim_channel=['Markers b'])
    assert np.unique(all_events[:, 2]).tolist() == [1, 2, 1000, 2000]

    # Check location of sources and detectors
    # We'll check the first two unique sources and detectors for each montage
    # These values were taken from the .elp file
    fiducials = [[0.912775E-01, 0, 0],
                 [0.599716E-03, 0.784103E-01, -0.231296E-17],
                 [-0.606755E-02, -0.709034E-01, 0]]

    native_head_t = get_ras_to_neuromag_trans(fiducials[0],
                                              fiducials[1],
                                              fiducials[2])

    # Sources
    assert_array_equal(data.info['chs'][0]['loc'][3:6],
                       apply_trans(native_head_t, [-0.818852E-01,
                                                   -0.464419E-01,
                                                   0.880970E-01]))

    assert_array_equal(data.info['chs'][2]['loc'][3:6],
                       apply_trans(native_head_t, [-0.898024E-01,
                                                   -0.456557E-01,
                                                   0.600431E-01]))

    assert_array_equal(data.info['chs'][80]['loc'][3:6],
                       apply_trans(native_head_t, [-0.878098E-01,
                                                   -0.348737E-01,
                                                   0.907238E-01]))

    assert_array_equal(data.info['chs'][82]['loc'][3:6],
                       apply_trans(native_head_t, [-0.971674E-01,
                                                   -0.360763E-01,
                                                   0.599644E-01]))

    # Detectors
    assert_array_equal(data.info['chs'][0]['loc'][6:9],
                       apply_trans(native_head_t, [-0.966161E-01,
                                                   0.338437E-01,
                                                   0.558559E-01]))

    assert_array_equal(data.info['chs'][10]['loc'][6:9],
                       apply_trans(native_head_t, [-0.965480E-01,
                                                   0.295639E-01,
                                                   0.802247E-01]))

    assert_array_equal(data.info['chs'][80]['loc'][6:9],
                       apply_trans(native_head_t, [-0.958327E-01,
                                                   0.451337E-01,
                                                   0.541672E-01]))

    assert_array_equal(data.info['chs'][90]['loc'][6:9],
                       apply_trans(native_head_t, [-0.932942E-01,
                                                   0.408094E-01,
                                                   0.775681E-01]))

    # Check channel types
    chan_type = dict(ac='302 (FIFFV_COIL_FNIRS_CW_AMPLITUDE)',
                     dc='302 (FIFFV_COIL_FNIRS_CW_AMPLITUDE)',
                     ph='305 (FIFFV_COIL_FNIRS_FD_PHASE)')

    assert str(data.info['chs'][0]['kind']) == '1100 (FIFFV_FNIRS_CH)'
    assert str(data.info['chs'][0]['coil_type']) == chan_type[datatype]

    assert str(data.info['chs'][160]['kind']) == '3 (FIFFV_STIM_CH)'
    assert str(data.info['chs'][160]['coil_type']) == '0 (FIFFV_COIL_NONE)'

    assert str(data.info['chs'][161]['kind']) == '3 (FIFFV_STIM_CH)'
    assert str(data.info['chs'][161]['coil_type']) == '0 (FIFFV_COIL_NONE)'


@requires_testing_data
@pytest.mark.parametrize('datatype,data', [('ac', raw_intensity_ac),
                                           ('dc', raw_intensity_dc),
                                           ('ph', raw_intensity_ph)])
def test_boxy_raw(datatype, data):
    for m_num, i_mtg in enumerate(['a', 'b']):
        for b_num, i_blk in enumerate(['001', '002']):

            # Load our p_pod files.
            filename = os.path.join(data_path(download=False), 'BOXY',
                                    'boxy_short_recording', 'boxy_p_pod_files',
                                    '1anc071' + i_mtg + '.' + i_blk +
                                    'raw_data.mat')
            ppod_raw = spio.loadmat(filename)

            # Grab the appropriate python data based on which block we're on.
            data_len = int((np.shape(raw_intensity_ac._data)[1])/2)
            py_data = np.transpose(data.copy().pick(mtg_list[m_num]).
                                   _data[:, data_len * b_num:data_len *
                                         (b_num + 1)])

            # Swap channels back to original.
            for i_chan in range(0, np.size(py_data, axis=1), 2):
                py_data[:, [i_chan, i_chan + 1]] = (py_data[:,
                                                    [i_chan + 1, i_chan]])

            # Compare our raw data between p_pod and python.
            assert (abs(ppod_raw[datatype] - py_data) <= thresh).all()


@requires_testing_data
@pytest.mark.parametrize('datatype,data', [('ac', raw_intensity_ac),
                                           ('dc', raw_intensity_dc),
                                           ('ph', raw_intensity_ph)])
def test_boxy_epochs(datatype, data):
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

    # BOXY epochs.
    epochs_a = mne.Epochs(data.copy().pick(mtg_a), mtg_a_events,
                          event_id=mtg_a_event_dict, tmin=tmin,
                          tmax=tmax, reject=reject_criteria,
                          reject_by_annotation=False, proj=True,
                          baseline=None, preload=True, detrend=None,
                          verbose=True, event_repeated='drop')

    epochs_b = mne.Epochs(data.copy().pick(mtg_b), mtg_b_events,
                          event_id=mtg_b_event_dict, tmin=tmin,
                          tmax=tmax, reject=reject_criteria,
                          reject_by_annotation=False, proj=True,
                          baseline=None, preload=True, detrend=None,
                          verbose=True, event_repeated='drop')

    # Swap channels back to original.
    for i_epoch in range(np.size(epochs_a, axis=0)):
        for i_chan in range(0, np.size(epochs_a, axis=1), 2):
            epochs_a._data[i_epoch, [i_chan, i_chan + 1], :] =\
              epochs_a._data[i_epoch, [i_chan + 1, i_chan], :]

    for i_epoch in range(np.size(epochs_b, axis=0)):
        for i_chan in range(0, np.size(epochs_b, axis=1), 2):
            epochs_b._data[i_epoch, [i_chan, i_chan + 1], :] =\
              epochs_b._data[i_epoch, [i_chan + 1, i_chan], :]

    # Now compare epochs.
    epoch_dict = dict(a=epochs_a,
                      b=epochs_b)

    for m_num, i_mtg in enumerate(['a', 'b']):

        # Load our p_pod files, both blocks.
        filename = os.path.join(data_path(download=False), 'BOXY',
                                'boxy_short_recording', 'boxy_p_pod_files',
                                '1anc071' + i_mtg + '.001' +
                                'epoch_data.mat')
        ppod_epoch1 = spio.loadmat(filename)

        filename = os.path.join(data_path(download=False), 'BOXY',
                                'boxy_short_recording', 'boxy_p_pod_files',
                                '1anc071' + i_mtg + '.002' +
                                'epoch_data.mat')
        ppod_epoch2 = spio.loadmat(filename)

        # Create empty array to store our combined blocks.
        epoch_num = (len(ppod_epoch1[datatype + '_epochs']) +
                     len(ppod_epoch2[datatype + '_epochs']))
        ppod_epochs = np.zeros((epoch_num, 80, 16))

        # Loop through both blocks.
        for epoch_num, i_epoch in enumerate(ppod_epoch1[datatype + '_epochs']):
            ppod_epochs[epoch_num, :, :] = np.transpose(i_epoch[0])

        last_blk = epoch_num
        for epoch_num, i_epoch in enumerate(ppod_epoch2[datatype + '_epochs']):
            ppod_epochs[epoch_num + last_blk + 1, :, :] =\
                np.transpose(i_epoch[0])

        py_epochs = epoch_dict[i_mtg]

        # Compare our epochs between p_pod and python.
        assert (abs(ppod_epochs - py_epochs) <= thresh).all()


@requires_testing_data
@pytest.mark.parametrize('datatype,data', [('ac', raw_intensity_ac),
                                           ('dc', raw_intensity_dc),
                                           ('ph', raw_intensity_ph)])
def test_boxy_evoked(datatype, data):
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

    epochs_a = mne.Epochs(data.copy().pick(mtg_a), mtg_a_events,
                          event_id=mtg_a_event_dict, tmin=tmin,
                          tmax=tmax, reject=reject_criteria,
                          reject_by_annotation=False, proj=True,
                          baseline=None, preload=True, detrend=None,
                          verbose=True, event_repeated='drop')

    epochs_b = mne.Epochs(data.copy().pick(mtg_b), mtg_b_events,
                          event_id=mtg_b_event_dict, tmin=tmin,
                          tmax=tmax, reject=reject_criteria,
                          reject_by_annotation=False, proj=True,
                          baseline=None, preload=True, detrend=None,
                          verbose=True, event_repeated='drop')

    # Swap channels back to original.
    for i_epoch in range(np.size(epochs_a, axis=0)):
        for i_chan in range(0, np.size(epochs_a, axis=1), 2):
            epochs_a._data[i_epoch, [i_chan, i_chan + 1], :] =\
              epochs_a._data[i_epoch, [i_chan + 1, i_chan], :]

    for i_epoch in range(np.size(epochs_b, axis=0)):
        for i_chan in range(0, np.size(epochs_b, axis=1), 2):
            epochs_b._data[i_epoch, [i_chan, i_chan + 1], :] =\
              epochs_b._data[i_epoch, [i_chan + 1, i_chan], :]

    evoked_a1 = epochs_a['Montage_A/Event_1'].average()
    evoked_a2 = epochs_a['Montage_A/Event_2'].average()

    evoked_b1 = epochs_b['Montage_B/Event_1'].average()
    evoked_b2 = epochs_b['Montage_B/Event_2'].average()

    evoke_dict = dict(a1=evoked_a1,
                      a2=evoked_a2,
                      b1=evoked_b1,
                      b2=evoked_b2)

    # Loop through our montages.
    for m_num, i_mtg in enumerate(['a', 'b']):

        # Load in our p_pod files.
        filename = os.path.join(data_path(download=False), 'BOXY',
                                'boxy_short_recording', 'boxy_p_pod_files',
                                '1anc071' + i_mtg + '.002' +
                                'evoke_data.mat')
        ppod_evoke_all = spio.loadmat(filename)

        # Just get the relevant p_pod data.
        ppod_evoke = ppod_evoke_all['a' + datatype].swapaxes(0, 2)

        # Grab the relevant python data, and combine blocks.
        py_evoke = np.zeros(np.shape(ppod_evoke))
        py_evoke[0, :, :] = evoke_dict[i_mtg + "1"]._data
        py_evoke[1, :, :] = evoke_dict[i_mtg + "2"]._data

        # Compare p_pod and python evoked.
        assert (abs(ppod_evoke - py_evoke) <= thresh).all()
