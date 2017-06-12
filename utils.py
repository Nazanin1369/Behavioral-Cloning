import os
import csv
import cv2

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

DRIVING_LOG = './data/driving_log.csv'
STEERING_CORRECTION = 0.28

orig_image_dir = './data/IMG/'
aug_image_dir = './data/augment/'
visualize_root = './data/visualize/'

class DrivingData:
    image_file = None
    steering_angle = 0.0
    directory = None

    def __init__(self, image, angle, directory):
        self.image_file = image
        self.steering_angle = angle
        self.directory = directory

def read_driving_log(filename):
    raw_samples = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            raw_samples.append(line)

    return raw_samples[1:]

def get_driving_data(data):
    driving_data = []

    for s in data:

        name = s[0].split('/')[-1]
        steering = float(s[3])

        driving_data.append(DrivingData(name, steering, orig_image_dir))

    return driving_data

def filter_low_steering_angles(data, threshold=0.01, keeppct=0.10):
    driving_data = []

    for dd in data:
        steering = dd.steering_angle

        if (abs(steering) > 0.9):
            print('anomaly: ', dd.image_file, ' steering: ', steering)
            continue

        if (abs(steering) < threshold):

            # only include a 10th of the near-zero steering angles
            if np.random.random_sample() < keeppct:
                driving_data.append(dd)
        else:
            driving_data.append(dd)

    return driving_data

# Use the left and right camera images 
# Use the center camera steering angle and add a correction factor
def augment_with_lf_camera(data, threshold=0.20):
    driving_data = []

    for dd in data:
        driving_data.append(dd)

        steering = dd.steering_angle

        # if greater than a threshold value, add values
        # for left and right camera data
        if (abs(steering) > threshold):
            correction = STEERING_CORRECTION
            steering_left = steering + correction
            steering_right = steering - correction

            fn = dd.image_file

            left_fn = fn.replace('center', 'left')
            driving_data.append(DrivingData(left_fn, steering_left, orig_image_dir))

            right_fn = fn.replace('center', 'right')
            driving_data.append(DrivingData(right_fn, steering_right, orig_image_dir))

    return driving_data



# taken from Vivek Yadav
def change_brightness(image, angle):
    # Randomly select a percent change
    change_pct = np.random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image, angle

# taken from Vivek Yadav
def add_shadow(image, angle):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2) == 1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image, angle

# taken from Vivek Yadav
def shift_image(image, angle):
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    angle = angle + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1,0, transX], [0,1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))

    return image, angle

def flip_image(image, angle):
    image = np.fliplr(image)
    angle = angle * -1.0

    return image, angle

def augment(func, data, prefix, threshold=0.20, nowrite=False):
    driving_data = []

    fnprfx = prefix

    print('augmenting ... : ', prefix)

    for dd in data:
        steering = dd.steering_angle

        # if greater than a threshold value, add values
        # for left and right camera data
        if (abs(steering) > threshold):

            src_fn = orig_image_dir + dd.image_file
            aug_fn = aug_image_dir + fnprfx + dd.image_file

            #print(src_fn, aug_fn)
            image = cv2.imread(src_fn)

            image, angle = func(image, steering)

            if (not nowrite):
                #print('writing image ... : ', aug_fn)
                cv2.imwrite(aug_fn, image)

            driving_data.append(DrivingData(fnprfx + dd.image_file, angle, aug_image_dir))


    return driving_data

def plot_histogram(driving_data, bins, fn, title):
    data = list(map(lambda d: d.steering_angle, driving_data))

    n_data = len(data)
    print ("Number of samples: ", n_data)

    num_bins = bins
    samples_per_bin = n_data/num_bins
    hist, bins = np.histogram(data, num_bins)

    fig = plt.figure()

    width = 0.9 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(data), np.max(data)), (samples_per_bin, samples_per_bin), 'k-')

    plt.title(title)
    
    fig.savefig(fn)

# Use to visualize raw data
def augment_and_visualize_data(data):
    nbin = 30

    current_data = []

    # let's see distribution for raw data
    plot_histogram(data, nbin, 
        visualize_root + 'raw_samples-histogram.png', 'Raw driving data')
    
    # reduce occurence of low-angle data
    filtered_low = filter_low_steering_angles(data)

    # see what data distribution look now
    plot_histogram(filtered_low, nbin, 
        visualize_root + 'filtered_samples-histogram.png', 'Low-steering angles removed')

    data_with_lf = augment_with_lf_camera(filtered_low)

    plot_histogram(data_with_lf, nbin, 
        visualize_root + 'with_lf_samples-histogram.png', 'With L/F angles')

    # brightness
    augmented_brightness = augment(change_brightness, data_with_lf, 'augment_brightness_')

    current_data.extend(data_with_lf)
    current_data.extend(augmented_brightness)

    plot_histogram(data_with_lf, nbin, 
        visualize_root + 'with_brightness_samples-histogram.png', 'After adding brightness')

    # shadow
    data_with_shadow = augment(add_shadow, data_with_lf, 'augment_shadow_')
    current_data.extend(data_with_shadow)

    plot_histogram(current_data, nbin, 
        visualize_root + 'with_shadow_samples-histogram.png', 'After adding shadow')

    #shift
    data_with_shift = augment(shift_image, data_with_lf, 'augment_shift_')
    current_data.extend(data_with_shift)

    plot_histogram(current_data, nbin, 
        visualize_root + 'with_shift_samples-histogram.png', 'After adding shift')

    # flipping
    data_flipped = augment(flip_image, data_with_lf, 'augment_flipped_')
    current_data.extend(data_flipped)

    plot_histogram(current_data, nbin, 
        visualize_root + 'with_flipped_samples-histogram.png', 'After flipping')

    return current_data

def main():
    samples = get_driving_data(read_driving_log(DRIVING_LOG))
    augment_and_visualize_data(samples)

if __name__ == "__main__":
    main()