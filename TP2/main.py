import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import os
import scipy.stats as stats

def normalize_by_collumn(array):

    return (array - array.min(0)) / array.ptp(0)

def extractFeatures():

    array_feat = np.array([])

    counter = 0

    path = "./MER/Songs/"

    for file in os.listdir(path):

        song_path = path + file

        song, sr = lib.load(song_path, sr= 22050, mono= True)

        window_length = 0.09288

        n_fft = int(sr * window_length)

        hop_length = int(sr * 0.02322)

        #MFCC
        mfccs = lib.feature.mfcc(y=song,n_mfcc=13)

        mfccs_stats = np.zeros((13,7))

        for i in range(0,len(mfccs)):

            mfccs_stats[i][0] = np.mean(mfccs[i])
            mfccs_stats[i][1] = np.std(mfccs[i])
            mfccs_stats[i][2] = stats.skew(mfccs[i])
            mfccs_stats[i][3] = stats.kurtosis(mfccs[i])
            mfccs_stats[i][4] = np.median(mfccs[i])
            mfccs_stats[i][5] = np.max(mfccs[i])
            mfccs_stats[i][6] = np.min(mfccs[i])

        mfccs_stats = mfccs_stats.flatten()

        #Spectral centroid
        spectral_centroid = lib.feature.spectral_centroid(y=song)

        sc_stats = np.zeros((1,7))

        for i in range(0,len(spectral_centroid)):

            sc_stats[i][0] = np.mean(spectral_centroid[i])
            sc_stats[i][1] = np.std(spectral_centroid[i])
            sc_stats[i][2] = stats.skew(spectral_centroid[i])
            sc_stats[i][3] = stats.kurtosis(spectral_centroid[i])
            sc_stats[i][4] = np.median(spectral_centroid[i])
            sc_stats[i][5] = np.max(spectral_centroid[i])
            sc_stats[i][6] = np.min(spectral_centroid[i])        

        sc_stats = sc_stats.flatten()



        #Spectral bandwidth
        spectral_bandwidth = lib.feature.spectral_bandwidth(y=song)

        sb_stats = np.zeros((1,7))

        for i in range(0,len(spectral_bandwidth)):

            sb_stats[i][0] = np.mean(spectral_bandwidth[i])
            sb_stats[i][1] = np.std(spectral_bandwidth[i])
            sb_stats[i][2] = stats.skew(spectral_bandwidth[i])
            sb_stats[i][3] = stats.kurtosis(spectral_bandwidth[i])
            sb_stats[i][4] = np.median(spectral_bandwidth[i])
            sb_stats[i][5] = np.max(spectral_bandwidth[i])
            sb_stats[i][6] = np.min(spectral_bandwidth[i])        

        sb_stats = sb_stats.flatten()

        #Spectral contrast
        spectral_contrast = lib.feature.spectral_contrast(y=song)

        scon_stats = np.zeros((7,7))

        for i in range(0,len(spectral_contrast)):

            scon_stats[i][0] = np.mean(spectral_contrast[i])
            scon_stats[i][1] = np.std(spectral_contrast[i])
            scon_stats[i][2] = stats.skew(spectral_contrast[i])
            scon_stats[i][3] = stats.kurtosis(spectral_contrast[i])
            scon_stats[i][4] = np.median(spectral_contrast[i])
            scon_stats[i][5] = np.max(spectral_contrast[i])
            scon_stats[i][6] = np.min(spectral_contrast[i])        

        scon_stats = scon_stats.flatten()

        print(scon_stats.shape)
        print(scon_stats)



        return

    return array_feat


def main():

    features_file = open("./MER/top100_features.csv",'r')

    feat_array = np.genfromtxt(features_file, delimiter=',', dtype=None)

    feat_array = feat_array[1::,1:len(feat_array[0])-1:].astype(np.float64)

    feat_norm = (feat_array - feat_array.min(0)) / feat_array.ptp(0)

    np.savetxt("result.csv", feat_norm, delimiter=',')


if __name__ == "__main__":

    extractFeatures()

