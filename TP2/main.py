import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import os
import scipy.stats as stats
import scipy.spatial.distance as ssd
import pandas


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

        # Spectral flatness
        spectral_flatness = lib.feature.spectral_flatness(y=song)

        sf_stats = np.zeros((1, 7))

        for i in range(0, len(spectral_flatness)):

            sf_stats[i][0] = np.mean(spectral_flatness[i])
            sf_stats[i][1] = np.std(spectral_flatness[i])
            sf_stats[i][2] = stats.skew(spectral_flatness[i])
            sf_stats[i][3] = stats.kurtosis(spectral_flatness[i])
            sf_stats[i][4] = np.median(spectral_flatness[i])
            sf_stats[i][5] = np.max(spectral_flatness[i])
            sf_stats[i][6] = np.min(spectral_flatness[i])

        sf_stats = sf_stats.flatten()

        # Spectral rolloff
        spectral_rolloff = lib.feature.spectral_rolloff(y=song)

        sr_stats = np.zeros((1, 7))

        for i in range(0, len(spectral_rolloff)):

            sr_stats[i][0] = np.mean(spectral_rolloff[i])
            sr_stats[i][1] = np.std(spectral_rolloff[i])
            sr_stats[i][2] = stats.skew(spectral_rolloff[i])
            sr_stats[i][3] = stats.kurtosis(spectral_rolloff[i])
            sr_stats[i][4] = np.median(spectral_rolloff[i])
            sr_stats[i][5] = np.max(spectral_rolloff[i])
            sr_stats[i][6] = np.min(spectral_rolloff[i])

        sr_stats = sr_stats.flatten()

        # F0
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

        # RMS
        rms = lib.feature.rms(y=song)

        rms_stats = np.zeros((1, 7))

        for i in range(0, len(rms)):

            rms_stats[i][0] = np.mean(rms[i])
            rms_stats[i][1] = np.std(rms[i])
            rms_stats[i][2] = stats.skew(rms[i])
            rms_stats[i][3] = stats.kurtosis(rms[i])
            rms_stats[i][4] = np.median(rms[i])
            rms_stats[i][5] = np.max(rms[i])
            rms_stats[i][6] = np.min(rms[i])

        rms_stats = rms_stats.flatten()

        # Zero crossing rate
        zero_crossing_rate = lib.feature.zero_crossing_rate(y=song)

        zero_crossing_rate_stats = np.zeros((1, 7))

        for i in range(0, len(zero_crossing_rate)):

            zero_crossing_rate_stats[i][0] = np.mean(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][1] = np.std(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][2] = stats.skew(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][3] = stats.kurtosis(
                zero_crossing_rate[i])
            zero_crossing_rate_stats[i][4] = np.median(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][5] = np.max(zero_crossing_rate[i])
            zero_crossing_rate_stats[i][6] = np.min(zero_crossing_rate[i])

        zero_crossing_rate_stats = zero_crossing_rate_stats.flatten()

        # Tempo
        tempo = lib.feature.tempo(y=song)

        tempo_stats = np.zeros((7,))

        tempo_stats[0] = np.mean(tempo)
        tempo_stats[1] = np.std(tempo)
        tempo_stats[2] = stats.skew(tempo)
        tempo_stats[3] = stats.kurtosis(tempo)
        tempo_stats[4] = np.median(tempo)
        tempo_stats[5] = np.max(tempo)
        tempo_stats[6] = np.min(tempo)

        feat_array = np.concatenate((mfccs_stats, sc_stats, sb_stats, scon_stats, sf_stats,
                                    sr_stats, f0_stats, rms_stats, zero_crossing_rate_stats, tempo_stats))

        final_array.append(feat_array)

    return np.array(final_array)


def featureArr(file):
    print(" Reading Top 100 features file")
    top100 = np.genfromtxt(file, delimiter=',')
    nl, nc = top100.shape
    print("dim ficheiro top100_features.csv original = ", nl, "x", nc)
    print()
    print(top100)
    top100 = top100[1:, 1:(nc-1)]  # eliminar a 1ª linha e 1ª coluna
    nl, nc = top100.shape
    print()
    print("dim top100 data = ", nl, "x", nc)
    print()
    print(top100)
    return top100


def calculateDistances(top100: np.ndarray, features: np.ndarray):
    features[np.isnan(features)] = 0
    top100Euclidean = np.empty((900, 900))
    top100Manhattan = np.empty((900, 900))
    top100Cosine = np.empty((900, 900))
    featuresEuclidean = np.empty((900, 900))
    featuresManhattan = np.empty((900, 900))
    featuresCosine = np.empty((900, 900))

    for n in range(top100.shape[0]):
        for m in range(top100.shape[0]):
            # print(f"({n}, {m})")
            top100Euclidean[n, m] = ssd.euclidean(top100[n, :], top100[m, :])
            featuresEuclidean[n, m] = ssd.euclidean(
                features[n, :], features[m, :])
            top100Manhattan[n, m] = ssd.cityblock(top100[n, :], top100[m, :])
            featuresManhattan[n, m] = ssd.cityblock(
                features[n, :], features[m, :])
            top100Cosine[n, m] = ssd.cosine(top100[n, :], top100[m, :])
            featuresCosine[n, m] = ssd.cosine(features[n, :], features[m, :])

    distances = {
        'top100': {
            'euclidean': top100Euclidean,
            'manhattan': top100Manhattan,
            'cosine': top100Cosine
        },
        'features': {
            'euclidean': featuresEuclidean,
            'manhattan': featuresManhattan,
            'cosine': featuresCosine
        }
    }

    np.savetxt("dist/top100/euclidean.csv",
               distances['top100']['euclidean'], delimiter=",", fmt="%f")
    np.savetxt("dist/top100/manhattan.csv",
               distances['top100']['manhattan'], delimiter=",", fmt="%f")
    np.savetxt("dist/top100/cosine.csv",
               distances['top100']['cosine'], delimiter=",", fmt="%f")
    np.savetxt("dist/features/euclidean.csv",
               distances['features']['euclidean'], delimiter=",", fmt="%f")
    np.savetxt("dist/features/manhattan.csv",
               distances['features']['manhattan'], delimiter=",", fmt="%f")
    np.savetxt("dist/features/cosine.csv",
               distances['features']['cosine'], delimiter=",", fmt="%f")
    return [[top100Euclidean, top100Manhattan, top100Cosine], [featuresEuclidean, featuresManhattan, featuresCosine]]


def ranking(index, matrix, dset, path):
    dist = ['Euclidean', 'Manhattan', 'Cosine']
    ranking = [[], [], []]
    for i in range(3):
        row = matrix[i][index, :]
        indices = np.argsort(row)[1:21]
        for j in indices:
            ranking[i].append(dset[j].split("/")[-1])

        if not os.path.isdir(path):
            os.makedirs(path)
        file = f"{path}/{dist[i]}.txt"
        if os.path.isfile(file):
            continue
        print("here")
        print(file)
        with open(file, "w") as file:
            for song in ranking[i]:
                print(song, file=file)

    return ranking

def mdScores(index, metadata, size):
    scores = np.zeros((1, size))

    for i in range(len(metadata)):
        if i != index:
            score = 0

            if metadata[i][1] == metadata[index][1]: 
                score = score + 1
            if metadata[i][3] == metadata[index][3]: 
                score = score + 1

            for j in metadata[i][11].split("; "):
                for k in metadata[index][11].split("; "):
                    if j == k:
                        score = score + 1
                        break

            for j in metadata[i][9].split("; "):
                for k in metadata[index][9].split("; "):
                    if j == k:
                        score = score + 1
                        break

            scores[0, i] = score

    return scores

def mdRanking(index, metadata, dset, path):
    scores = mdScores(index, metadata, len(metadata))[0]

    indices = np.argsort(scores)[::-1]
    indices = indices[:20]

    print(f"Scores metadata = {np.sort(scores)[::-1][:20]}")

    ranking = list()
    for i in indices:
        ranking.append(dset[i].split("/")[-1])

    if path is not None:
        if not os.path.isdir(path): os.makedirs(path)
        filename = f"{path}/{dset[index].split('/')[-1]}.txt"
        if not os.path.isfile(filename):
            with open(filename, 'w') as file:
                for song in ranking:
                    print(song, file=file)

    return ranking

def precision(feat, t100, md):
    mds = set(md)
    pr = [[i(0, feat, md, mds), i(1, feat, md, mds), i(2, feat, md, mds)],[i(0, t100, md, mds), i(1, t100, md, mds), i(2, t100, md, mds)]]
    return pr

def i(int, arr, arr2, sets):
    return len(set(arr[0]).intersection(sets)) / len(arr2) * 100


def main():

    features_file = open("./MER/top100_features.csv", 'r')

    feat_array = np.genfromtxt(features_file, delimiter=',', dtype=None)

    feat_array = feat_array[1::, 1:len(feat_array[0])-1:].astype(np.float64)

    feat_norm = (feat_array - feat_array.min(0)) / feat_array.ptp(0)

    np.savetxt("result.csv", feat_norm, delimiter=',')


if __name__ == "__main__":

    # print(extractFeatures().shape)

    top100: np.ndarray
    
    if os.path.isfile("./result.csv"):
        top100 = np.genfromtxt("./result.csv", delimiter=",")
    else:
        top100 = featureArr("./MER/top100_features.csv")
        np.savetxt("./MER/top100_features.csv",top100, fmt="%f", delimiter=",")
    
    features: np.ndarray
    
    if os.path.isfile("./FMrosa.csv"):
        features = np.genfromtxt("./FMrosa.csv", delimiter=",")
    else:
        features = extractFeatures()
        np.savetxt("./FMrosa.csv", features, fmt="%f", delimiter=",")
    
    distances = calculateDistances(top100, features)
    d_top100 = distances[0]
    d_features = distances[1]

    p = "./MER/Songs"
    dset = [f"{p}/{x}" for x in sorted(os.listdir(p))]
    fileName = "./MER/panda_dataset_taffc_metadata.csv"
    queries = os.listdir("./Queries")

    for query in queries:
        print("Query: ", query)
        index = dset.index(f"{p}/{query}")
        featuresRanking = ranking(index, d_features, dset, path=f"featuresRankings")
        top100Ranking = ranking(index, d_top100, dset, path=f"top100Rankings")
        metadata = np.genfromtxt(fileName, delimiter=",", dtype=str)[1::]
        metadataRanking = mdRanking(index, metadata, dset, path=f"metadataRankings")
        precise = precision(featuresRanking, top100Ranking, metadataRanking)
        print("\nPrecisions: ")
        print(precise)
