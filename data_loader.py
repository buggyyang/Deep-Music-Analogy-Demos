import numpy as np
from sklearn.utils import shuffle


class MusicArrayLoader():
    def __init__(self, data_path, length, step_size):
        self.dataset = np.load(data_path, allow_pickle=True)
        self.__length = length  # 32
        self.__chunk_melodies = []
        self.__chunk_chords = []
        self.__current_index = 0
        self.__step_size = step_size  # 16
        self.__epoch = 0

    def __clipping(self, melody, chord):
        """
        input is (N, 130) and (N, 12)
        outputs a list of (32, 130) matrices, and a list of (12, 130) matrices.
        """
        len_m = melody.shape[0]
        len_c = chord.shape[0]
        clipped_melodies, clipped_chords = [], []
        if len_m > len_c:
            chord = np.pad(chord, ((0, len_m - len_c), (0, 0)), 'constant')
        elif len_c > len_m:
            melody = np.pad(melody, ((0, len_c - len_m), (0, 0)), 'constant')
            melody[-(len_c - len_m):, 129] = 1
        len_m = melody.shape[0]
        len_c = chord.shape[0]
        assert (len_m == len_c)
        for i in range(0, len_m, self.__step_size):
            if ((i + self.__length) < len_m) and (melody[i, 128] != 1):
                clipped_melodies.append(melody[i:i + self.__length])
                clipped_chords.append(chord[i:i + self.__length])
        return clipped_melodies, clipped_chords

    def chunking(self):
        for melody, chord in zip(self.dataset[0], self.dataset[1]):
            # melody.shape = (N, 130), chord.shape = (N, 12)
            m, c = self.__clipping(melody, chord)
            self.__chunk_melodies += m
            self.__chunk_chords += c
        self.__chunk_melodies = np.asarray(self.__chunk_melodies)
        self.__chunk_chords = np.asarray(self.__chunk_chords)
        assert (len(self.__chunk_melodies) == len(self.__chunk_chords))

    def get_n_music(self):
        return len(self.dataset[0])

    def check(self):
        if len(self.__chunk_melodies) == 0:
            raise ValueError('please chunk the music_array clips first')

    def get_n_sample(self):
        self.check()
        return len(self.__chunk_melodies)

    def get_n_epoch(self):
        return self.__epoch

    def reset(self):
        self.__current_index = 0
        self.__epoch = 0

    def shuffle_samples(self):
        self.check()
        self.__chunk_melodies, self.__chunk_chords = shuffle(
            self.__chunk_melodies, self.__chunk_chords)

    def get_batch(self, batch_size):
        self.check()
        if (self.__current_index + batch_size) > self.get_n_sample():
            t = self.__current_index
            self.__current_index = 0
            self.__epoch += 1
            return self.__chunk_melodies[t:], self.__chunk_chords[t:]
        else:
            t = self.__current_index
            self.__current_index += batch_size
            return self.__chunk_melodies[
                t:self.__current_index], self.__chunk_chords[
                    t:self.__current_index]
