import mne
import os
import sys
filename1 = sys.argv[1]
filename2 = sys.argv[2]
# outputfolder = r'G:\My Drive\formerlab\nnos\pilot for nnos grant\stim3'
# filename1 = r'G:\My Drive\formerlab\nnos\pilot for nnos grant\stim3\mse1new_stim3_Block-101_CH1.edf'
# filename2 = r'G:\My Drive\formerlab\nnos\pilot for nnos grant\stim3\mse2new_stim3_Block-101_CH3.edf'
edf1 = mne.io.read_raw_edf(filename1, preload=True)
edf2 = mne.io.read_raw_edf(filename2, preload=True)
sr = float(edf1.info["sfreq"])
print('Sampling rates:',sr,float(edf2.info["sfreq"]))

# Rename the channels in raw2
new_ch_names = [ch + '_added' for ch in edf2.ch_names]  # modify channel name to avoid repeats
edf2.rename_channels(dict(zip(edf2.ch_names, new_ch_names)))

# Append the channels of raw2 to raw1
for ch in edf2.ch_names:
    edf1.add_channels([edf2.copy().pick_channels([ch])])

# Merge the two raw objects
raw_merged = mne.concatenate_raws([edf1])

# Save the merged raw object to a new EDF file
#raw_merged.save(os.path.join(outputfolder,"merged.edf"), fmt='edf', overwrite=True)
mne.export.export_raw("merged.edf",raw_merged, fmt='edf', overwrite=True)
print('saved merged.edf')
