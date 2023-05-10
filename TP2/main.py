import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import os
import scipy.stats as stats


def normalize_by_collumn(array):

    return (array - array.min(0)) / array.ptp(0)


def extractFeatures():

    final_array = []

    counter = 0

    path = "./MER/Songs/"

    for file in os.listdir(path):

        song_path = path + file

        song, sr = lib.load(song_path, sr=22050, mono=True)

        window_length = 0.09288

        # MFCC
        mfccs = lib.feature.mfcc(y=song, n_mfcc=13)

        mfccs_stats = np.zeros((13, 7))

        for i in range(0, len(mfccs)):

            mfccs_stats[i][0] = np.mean(mfccs[i])
            mfccs_stats[i][1] = np.std(mfccs[i])
            mfccs_stats[i][2] = stats.skew(mfccs[i])
            mfccs_stats[i][3] = stats.kurtosis(mfccs[i])
            mfccs_stats[i][4] = np.median(mfccs[i])
            mfccs_stats[i][5] = np.max(mfccs[i])
            mfccs_stats[i][6] = np.min(mfccs[i])

        mfccs_stats = mfccs_stats.flatten()

        # Spectral centroid
        spectral_centroid = lib.feature.spectral_centroid(y=song)

        sc_stats = np.zeros((1, 7))

        for i in range(0, len(spectral_centroid)):

            sc_stats[i][0] = np.mean(spectral_centroid[i])
            sc_stats[i][1] = np.std(spectral_centroid[i])
            sc_stats[i][2] = stats.skew(spectral_centroid[i])
            sc_stats[i][3] = stats.kurtosis(spectral_centroid[i])
            sc_stats[i][4] = np.median(spectral_centroid[i])
            sc_stats[i][5] = np.max(spectral_centroid[i])
            sc_stats[i][6] = np.min(spectral_centroid[i])

        sc_stats = sc_stats.flatten()

        # Spectral bandwidth
        spectral_bandwidth = lib.feature.spectral_bandwidth(y=song)

        sb_stats = np.zeros((1, 7))

        for i in range(0, len(spectral_bandwidth)):

            sb_stats[i][0] = np.mean(spectral_bandwidth[i])
            sb_stats[i][1] = np.std(spectral_bandwidth[i])
            sb_stats[i][2] = stats.skew(spectral_bandwidth[i])
            sb_stats[i][3] = stats.kurtosis(spectral_bandwidth[i])
            sb_stats[i][4] = np.median(spectral_bandwidth[i])
            sb_stats[i][5] = np.max(spectral_bandwidth[i])
            sb_stats[i][6] = np.min(spectral_bandwidth[i])

        sb_stats = sb_stats.flatten()

        # Spectral contrast
        spectral_contrast = lib.feature.spectral_contrast(y=song)

        scon_stats = np.zeros((7, 7))

        for i in range(0, len(spectral_contrast)):

            scon_stats[i][0] = np.mean(spectral_contrast[i])
            scon_stats[i][1] = np.std(spectral_contrast[i])
            scon_stats[i][2] = stats.skew(spectral_contrast[i])
            scon_stats[i][3] = stats.kurtosis(spectral_contrast[i])
            scon_stats[i][4] = np.median(spectral_contrast[i])
            scon_stats[i][5] = np.max(spectral_contrast[i])
            scon_stats[i][6] = np.min(spectral_contrast[i])

        scon_stats = scon_stats.flatten()

        #Spectral flatness
        spectral_flatness = lib.feature.spectral_flatness(y=song)

        sf_stats = np.zeros((1,7))

        for i in range(0, len(spectral_flatness)):

            sf_stats[i][0] = np.mean(spectral_flatness[i])
            sf_stats[i][1] = np.std(spectral_flatness[i])
            sf_stats[i][2] = stats.skew(spectral_flatness[i])
            sf_stats[i][3] = stats.kurtosis(spectral_flatness[i])
            sf_stats[i][4] = np.median(spectral_flatness[i])
            sf_stats[i][5] = np.max(spectral_flatness[i])
            sf_stats[i][6] = np.min(spectral_flatness[i])

        sf_stats = sf_stats.flatten()

        #Spectral rolloff
        spectral_rolloff = lib.feature.spectral_rolloff(y=song)

        sr_stats = np.zeros((1,7))

        for i in range(0, len(spectral_rolloff)):

            sr_stats[i][0] = np.mean(spectral_rolloff[i])
            sr_stats[i][1] = np.std(spectral_rolloff[i])
            sr_stats[i][2] = stats.skew(spectral_rolloff[i])
            sr_stats[i][3] = stats.kurtosis(spectral_rolloff[i])
            sr_stats[i][4] = np.median(spectral_rolloff[i])
            sr_stats[i][5] = np.max(spectral_rolloff[i])
            sr_stats[i][6] = np.min(spectral_rolloff[i])

        sr_stats = sr_stats.flatten()

        #F0
        fmin = float(lib.note_to_hz('C2'))
        fmax = float(lib.note_to_hz('C7'))
        f0 = lib.yin(y=song, fmin=fmin, fmax=fmax)

        f0_stats = np.zeros((7,))

        f0_stats[0] = np.mean(f0)
        f0_stats[1] = np.std(f0)
        f0_stats[2] = stats.skew(f0)
        f0_stats[3] = stats.kurtosis(f0)
        f0_stats[4] = np.median(f0)
        f0_stats[5] = np.max(f0)
        f0_stats[6] = np.min(f0)

        #RMS
        rms = lib.feature.rms(y=song)

        rms_stats = np.zeros((1,7))

        for i in range(0, len(rms)):

            rms_stats[i][0] = np.mean(rms[i])
            rms_stats[i][1] = np.std(rms[i])
            rms_stats[i][2] = stats.skew(rms[i])
            rms_stats[i][3] = stats.kurtosis(rms[i])
            rms_stats[i][4] = np.median(rms[i])
            rms_stats[i][5] = np.max(rms[i])
            rms_stats[i][6] = np.min(rms[i])

        rms_stats = rms_stats.flatten()

        #Zero crossing rate
        zero_crossing_rate = lib.feature.zero_crossing_rate(y=song)

        zero_crossing_rate_stats = np.zeros((1,7))

        for i in range(0, len(zero_crossing_rate)):

            zero_crossing_rate_stats[i][0] = np.mean(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][1] = np.std(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][2] = stats.skew(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][3] = stats.kurtosis(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][4] = np.median(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][5] = np.max(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][6] = np.min(zero_crossing_rate[i])

        zero_crossing_rate_stats = zero_crossing_rate_stats.flatten()

        #Tempo
        tempo = lib.feature.tempo(y=song)

        tempo_stats = np.zeros((7,))

        tempo_stats[0] = np.mean(tempo)
        tempo_stats[1] = np.std(tempo)
        tempo_stats[2] = stats.skew(tempo)
        tempo_stats[3] = stats.kurtosis(tempo)
        tempo_stats[4] = np.median(tempo)
        tempo_stats[5] = np.max(tempo)
        tempo_stats[6] = np.min(tempo)

        feat_array = np.concatenate((mfccs_stats,sc_stats,sb_stats,scon_stats,sf_stats,sr_stats,f0_stats,rms_stats,zero_crossing_rate_stats,tempo_stats))
        
        final_array.append(feat_array)

    return np.array(final_array)


def main():

    features_file = open("./MER/top100_features.csv", 'r')

    feat_array = np.genfromtxt(features_file, delimiter=',', dtype=None)

    feat_array = feat_array[1::, 1:len(feat_array[0])-1:].astype(np.float64)

    feat_norm = (feat_array - feat_array.min(0)) / feat_array.ptp(0)

    np.savetxt("result.csv", feat_norm, delimiter=',')


if __name__ == "__main__":

    print(extractFeatures().shape)
