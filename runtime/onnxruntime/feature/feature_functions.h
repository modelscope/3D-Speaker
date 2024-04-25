//
// Created by shuli on 2024/3/4.
//

#ifndef SPEAKERLABENGINES_FEATURE_FUNCTIONS_H
#define SPEAKERLABENGINES_FEATURE_FUNCTIONS_H

#include <vector>
#include <cmath>
#include <complex>

namespace speakerlab {
    int round_up_to_nearest_power_of_two(int n);

    void init_bit_reverse_index(std::vector<int> &bit_rev_index, int n);

    void init_sin_tbl(std::vector<float> &sin_tbl, int n);

    void custom_fft(const std::vector<int> &bit_rev_index,
                    const std::vector<float> &sin_tbl,
                    std::vector<std::complex<float>> &data);
}

#endif //SPEAKERLABENGINES_FEATURE_FUNCTIONS_H
