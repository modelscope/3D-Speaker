//
// Created by shuli on 2024/3/6.
//

#include "feature/feature_functions.h"

int speakerlab::round_up_to_nearest_power_of_two(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void speakerlab::init_bit_reverse_index(std::vector<int> &bit_rev_index, int n) {
    int bits = static_cast<int>(std::log2(n));
    bit_rev_index.resize(n);
    for (int i = 0; i < n; ++i) {
        bit_rev_index[i] = 0;
        int x = i;
        for (int j = 0; j < bits; ++j) {
            bit_rev_index[i] = (bit_rev_index[i] << 1) | (x & 1);
            x >>= 1;
        }
    }
}

void speakerlab::init_sin_tbl(std::vector<float> &sin_tbl, int n) {
    sin_tbl.resize(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        sin_tbl[i] = sin(-2 * M_PI * i / n);
    }
}

void speakerlab::custom_fft(const std::vector<int> &bit_rev_index,
                            const std::vector<float> &sin_tbl,
                            std::vector<std::complex<float>> &data) {
    int n = data.size();

    for (int i = 0; i < n; ++i) {
        if (i < bit_rev_index[i]) {
            std::swap(data[i], data[bit_rev_index[i]]);
        }
    }

    for (int hs = 1; hs < n; hs *= 2) {
        int step = n / (hs * 2);
        for (int i = 0; i < n; i += hs * 2) {
            for (int j = i, k = 0; j < i + hs; ++j, k += step) {
                float cos_value = sin_tbl[(n / 4 - k + n / 2) % (n / 2)];
                if (k >= n / 4) cos_value = -cos_value;
                float sin_value = sin_tbl[k % (n / 2)];
                std::complex<float> t = std::complex<float>(-cos_value, sin_value) * data[j + hs];
                data[j + hs] = data[j] - t;
                data[j] += t;
            }
        }
    }
}