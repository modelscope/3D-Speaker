//
// Created by shuli on 2024/3/4.
//

#ifndef SPEAKERLABENGINES_FEATURE_COMMON_H
#define SPEAKERLABENGINES_FEATURE_COMMON_H

#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "feature_basic.h"

namespace speakerlab {

    class FramePreprocessor {
    public:
        FramePreprocessor() {}

        explicit FramePreprocessor(const FrameExtractionOptions &frame_opts);

        void dither(std::vector<float> &wav_data);

        void remove_dc_offset(std::vector<float> &wav_data);

        void pre_emphasis(std::vector<float> &wav_data);

        void windows_function(std::vector<float> &wav_data);

        void frame_pre_process(std::vector<float> &wav_data);

    private:
        FrameExtractionOptions opts_;
        std::default_random_engine generator_;
        std::normal_distribution<float> distribution_;
    };

    class MelBankProcessor {
    public:
        MelBankProcessor() {}

        explicit MelBankProcessor(const MelBanksOptions &mel_opts);

        void init_mel_bins(float sample_frequency, int window_length_padded);

        std::vector<std::pair<int, std::vector<float>>> get_mel_bins() {
            return mel_bins_;
        }

        inline float inverse_mel_scale(float mel_freq) {
            return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
        }

        inline float mel_scale(float freq) {
            return 1127.0f * logf (1.0f + freq / 700.0f);
        }
    private:
        MelBanksOptions opts_;
        std::vector<float> center_frequency_;
        std::vector<std::pair<int, std::vector<float>>> mel_bins_;
    };

    void subtract_feature_mean(std::vector<std::vector<float>>& features);
}

#endif //SPEAKERLABENGINES_FEATURE_COMMON_H
