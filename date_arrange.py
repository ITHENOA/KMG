import os
import numpy as np
from scipy.signal import find_peaks
import pywt
from skimage.transform import resize
from scipy.io import loadmat
import matplotlib.pyplot as plt

def find_label(file_name, gest):
    data = loadmat(file_name)
    key = data[gest]
    plt.plot(key)
    sensors_data = data['sensorsDataCalibratedFiltered']
    idx_max, _ = find_peaks(key)
    idx_min, _ = find_peaks(-key)
    idx = np.concatenate(([0], idx_max, idx_min, [len(key)]))
    idx = np.sort(idx)
    return sensors_data, idx


def scalogram(data, img_size):
    fb = pywt.CWT(data, np.arange(1, 128), 'mexh')
    cfs = np.abs(fb)
    img = (cfs - np.min(cfs)) / (np.max(cfs) - np.min(cfs))  # rescale between [0, 1]
    img = resize(img, img_size, anti_aliasing=True)
    return img

def get_new_name(current_folder, main_name, ext):
    img_names = os.listdir(current_folder)
    new_name = len(img_names)
    new_img_name = f"{main_name}_{new_name}.{ext}"
    return new_img_name


folders = ["2nd Patient's Dataset/12-Sep-2022-Test-Day-1-Evening-Layout3",
        "2nd Patient's Dataset/12-Sep-2022-Test-Day-1-Morning-Layout2",
        "2nd Patient's Dataset/13-Sep-2022-Test-Day-2-Morning-Layout1"]

key_gest = ["ring", "ring", "index", "index", "middle", "middle", "pinky", "pinky",
        "ring", "ring", "thumb", "thumb", "ring", "ring", "wrist", "wrist",
        "ring", "ring", "thumb", "thumb", "ring", "ring"]

gestures = ["fist", "fist", "index", "index", "middle", "middle", "pinky", "pinky",
        "ring", "ring", "thumb", "thumb", "tripod", "tripod", "wristUp", "wristUp",
        "wristUpFist", "wristUpFist", "wristUpThumb", "wristUpThumb",
        "wristUpTripod", "wristUpTripod"]

parent_folder = "data_mat_128_mag"
img_size = (128, 128)
n_channel = 16

for folder in folders:
    mat_files = [f for f in os.listdir(folder) if f.endswith('.mat')]
    for mat_file, gest in zip(mat_files, gestures):
        current_folder = os.path.join(parent_folder, gest)
        os.makedirs(current_folder, exist_ok=True)
        sig, idx = find_label(os.path.join(folder, mat_file), key_gest[gestures.index(gest)])
        for s in range(1, len(idx), 2):
            image = np.zeros((img_size[0], img_size[1], n_channel))
            for c in range(n_channel):
                sig_mag = np.sqrt(np.sum(sig[idx[s-1]:idx[s], c*3:(c+1)*3]**2, axis=1))
                image[:, :, c] = scalogram(sig_mag, img_size)
            img_name = get_new_name(current_folder, gest, "mat")
            np.save(os.path.join(current_folder, img_name), image)

