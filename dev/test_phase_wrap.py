import os
import pickle
import scipy
from scipy import io
import numpy as np

crnt_dir = os.getcwd()

filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 
                        'phase_wrap_data.pickle')

# filename_mat = os.path.join(crnt_dir, 'dev' , 'matlab_files', 
#                             'all_datamatlab.mat')

# with open(filename, 'wb') as f:
#     pickle.dump(all_data, f)
    
# scipy.io.savemat(filename_mat, mdict=dict(data=all_data))
	
with open(filename, 'rb') as f:
    all_data = pickle.load(f)
    
###phase unwrapping###
thresh = 1e-10 #determine to which decimal place we will compare

# print('Fixing first few bad points')

n_bad_points = 12
for i_point in range(n_bad_points):
    all_data[:,i_point] = all_data[:,n_bad_points] 
    
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_first_bad_python.mat')
first_bad_data = scipy.io.loadmat(filename)

test6 = abs(first_bad_data['data'] - all_data) <= thresh
if test6.all():
    print("SUCCESS! First bad matches Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! First bad DOES NOT match Matlab to " + str(thresh) + " decimal points!")

# print('Fixing phase wrap')

for i_chan in range(np.size(all_data, axis=0)):
    if np.mean(all_data[i_chan,:50]) < 180:
        wrapped_points = all_data[i_chan, :] > 270
        all_data[i_chan, wrapped_points] -= 360
    else:
        wrapped_points = all_data[i_chan,:] < 90
        all_data[i_chan, wrapped_points] += 360
      
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_unwrap_python.mat')
unwrapped_data = scipy.io.loadmat(filename)

test1 = abs(unwrapped_data['data'] - all_data) <= thresh
if test1.all():
    print("SUCCESS! Phase wrap matches Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! Phase wrap DOES NOT match Matlab to " + str(thresh) + " decimal points!")
        
# print('Detrending phase data')
# scipy.io.savemat(file_name = r"C:\Users\spork\Desktop\all_dataunwrap_matlab.mat",
#               mdict=dict(data=all_data))

y = np.linspace(1, np.size(all_data, axis=1),
                np.size(all_data, axis=1))
x = np.transpose(y)
for i_chan in range(np.size(all_data, axis=0)):
    poly_coeffs = np.polyfit(x,all_data[i_chan, :] ,3)
    tmp_ph = all_data[i_chan, :] - np.polyval(poly_coeffs,x)
    all_data[i_chan, :] = tmp_ph
    
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_detrend_python.mat')
detrend_data = scipy.io.loadmat(filename)
    
test2 = abs(detrend_data['data'] - all_data) <= thresh
if test2.all():
    print("SUCCESS! Detrend matches Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! Detrend DOES NOT match Matlab to " + str(thresh) + " decimal points!")

# print('Removing phase mean')
# scipy.io.savemat(file_name = r"C:\Users\spork\Desktop\all_datadetrend_matlab.mat",
#               mdict=dict(data=all_data))

mrph = np.mean(all_data,axis=1);
for i_chan in range(np.size(all_data, axis=0)):
    all_data[i_chan,:]=(all_data[i_chan,:]-mrph[i_chan])
    
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_mean_python.mat')
mean_data = scipy.io.loadmat(filename)

test3 = abs(mean_data['data'] - all_data) <= thresh
if test3.all():
    print("SUCCESS! Subtracting mean matches Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! Subtracting mean DOES NOT match Matlab to " + str(thresh) + " decimal points!")

# print('Removing phase outliers')
# scipy.io.savemat(file_name = r"C:\Users\spork\Desktop\all_datamean_matlab.mat",
#               mdict=dict(data=all_data))

ph_out_thr=3;
sdph=np.std(all_data,1, ddof = 1); #set ddof to 1 to mimic matlab
n_ph_out = np.zeros(np.size(all_data, axis=0), dtype= np.int8)

for i_chan in range(np.size(all_data, axis=0)):
    outliers = np.where(np.abs(all_data[i_chan,:]) > 
                (ph_out_thr*sdph[i_chan]))
    outliers = outliers[0]
    if len(outliers) > 0:
        if outliers[0] == 0:
            outliers = outliers[1:]
        if len(outliers) > 0:
            if outliers[-1] == np.size(all_data, axis=1) - 1:
                outliers = outliers[:-1]
            n_ph_out[i_chan] = int(len(outliers))
            for i_pt in range(n_ph_out[i_chan]):
                j_pt = outliers[i_pt]
                all_data[i_chan,j_pt] = (
                    (all_data[i_chan,j_pt-1] + 
                      all_data[i_chan,j_pt+1])/2)
                
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_outliers_python.mat')
outlier_data = scipy.io.loadmat(filename)

test4 = abs(outlier_data['data'] - all_data) <= thresh
if test4.all():
    print("SUCCESS! Removed outliers match Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! Removed outliers DO NOT match Matlab to " + str(thresh) + " decimal points!")
    
    
# now let's normalise our data #
for i_chan in range(len(all_data)):
    all_data[i_chan,:] = (all_data[i_chan,:] - np.mean(all_data[i_chan,:]))
    
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_norm_python.mat')
outlier_data = scipy.io.loadmat(filename)

test7 = abs(outlier_data['data'] - all_data) <= thresh
if test4.all():
    print("SUCCESS! Norm data matches Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! Norm data DOES NOT match Matlab to " + str(thresh) + " decimal points!")


#convert phase to pico seconds
# scipy.io.savemat(file_name = r"C:\Users\spork\Desktop\data_picosec_matlab.mat",
#               mdict=dict(data=all_data))
mtg_mdf = 1.1e8
for i_chan in range(np.size(all_data, axis=0)):
    all_data[i_chan,:] = ((1e12*all_data[i_chan,:])/(360*mtg_mdf))
    
filename = os.path.join(crnt_dir, 'dev' , 'matlab_files', 'data_picosec_python.mat')
picosec_data = scipy.io.loadmat(filename)

test5 = abs(picosec_data['data'] - all_data) <= thresh
if test5.all():
    print("SUCCESS! Pico seconds match Matlab to " + str(thresh) + " decimal points!")
else:
    print("FAILURE! Pico seconds DO NOT match Matlab to " + str(thresh) + " decimal points!")