//
// Created by shuli on 2024/3/4.
//

#include "feature/feature_common.h"


speakerlab::FramePreprocessor::FramePreprocessor(
        const speakerlab::FrameExtractionOptions &frame_opts) : opts_(frame_opts),
                                                                generator_(0),
                                                                distribution_(0, 1.0) {
}

void speakerlab::FramePreprocessor::dither(std::vector<float> &wav_data) {
    if (opts_.dither == 0.0) return;
    for (size_t i = 0; i < wav_data.size(); i++) {
        wav_data[i] += opts_.dither * distribution_(generator_);
    }
}

void speakerlab::FramePreprocessor::remove_dc_offset(std::vector<float> &wav_data) {
    if (!opts_.remove_dc_offset) return;
    float mean = 0.0;
    for (size_t j = 0; j < wav_data.size(); ++j) mean += wav_data[j];
    mean /= wav_data.size();
    for (size_t j = 0; j < wav_data.size(); ++j) wav_data[j] -= mean;
}


void speakerlab::FramePreprocessor::pre_emphasis(std::vector<float> &wav_data) {
    float pre_emphasis_coefficient = opts_.pre_emphasis_coefficient;
    if (pre_emphasis_coefficient == 0.0) return;
    for (size_t i = wav_data.size() - 1; i > 0; i--) {
        wav_data[i] -= pre_emphasis_coefficient * wav_data[i - 1];
    }
    wav_data[0] -= pre_emphasis_coefficient * wav_data[0];
}

void speakerlab::FramePreprocessor::windows_function(std::vector<float> &wav_data) {
    std::vector<float> window;
    int frame_length = opts_.compute_window_size();
    assert(wav_data.size() == frame_length);
    window.resize(frame_length);
    double a = 2 * M_PI / (frame_length - 1);
    for (size_t i = 0; i < frame_length; i++) {
        double i_fl = static_cast<double>(i);
        if (opts_.window_type == "hanning") {
            window[i] = 0.5 - 0.5 * cos(a * i_fl);
        } else if (opts_.window_type == "sine") {
            // when you are checking ws wikipedia, please
            // note that 0.5 * a = M_PI/(frame_length-1)
            window[i] = sin(0.5 * a * i_fl);
        } else if (opts_.window_type == "hamming") {
            window[i] = 0.54 - 0.46 * cos(a * i_fl);
        } else if (opts_.window_type == "povey") {  // like hamming but goes to zero at edges.
            window[i] = pow(0.5 - 0.5 * cos(a * i_fl), 0.85);
        } else if (opts_.window_type == "rectangular") {
            window[i] = 1.0;
        } else if (opts_.window_type == "blackman") {
            window[i] = opts_.blackman_coefficient - 0.5 * cos(a * i_fl) +
                        (0.5 - opts_.blackman_coefficient) * cos(2 * a * i_fl);
        } else {
            std::cerr << "Unknown window type " << opts_.window_type << std::endl;
        }
    }
    for (size_t i = 0; i < wav_data.size(); i++) {
        wav_data[i] *= window[i];
    }
}

void speakerlab::FramePreprocessor::frame_pre_process(std::vector<float> &wav_data) {
    dither(wav_data);
    remove_dc_offset(wav_data);
    pre_emphasis(wav_data);
    windows_function(wav_data);
}

speakerlab::MelBankProcessor::MelBankProcessor(const speakerlab::MelBanksOptions &mel_opts): opts_(mel_opts) {
    if (opts_.num_bins < 3) {
        std::cerr << "Mel Banks do not have enough " << opts_.num_bins << " mel bins" << std::endl;
    }
}

void speakerlab::MelBankProcessor::init_mel_bins(float sample_frequency, int window_padded_length) {
    int num_fft_bins = window_padded_length / 2;
    int num_bins = opts_.num_bins;
    float nyquist = 0.5 * sample_frequency;
    float low_frequency = opts_.low_freq, high_frequency;
    if (opts_.high_freq > 0.0) high_frequency = opts_.high_freq;
    else high_frequency = nyquist + opts_.high_freq;
    
    std::cout << "In init_mel_bins: num_fft_bins = " << num_fft_bins 
              << " num_bins = " << num_bins
              << " nyquist = " << nyquist
              << " low_frequency = " << low_frequency
              << " high_frequency = " << high_frequency
              << std::endl;

    if (low_frequency < 0.0 || low_frequency >= nyquist ||
        high_frequency <= 0.0 || high_frequency > nyquist || high_frequency <= low_frequency) {
        std::cerr << "Bad values in options: low-frequency " << low_frequency
                  << " and high-frequency " << high_frequency << " vs nyquist " << nyquist;
    }

    float fft_bin_width = sample_frequency / window_padded_length;
    float mel_low_frequency = mel_scale(low_frequency);
    float mel_high_frequency = mel_scale(high_frequency);
    float mel_frequency_delta = (mel_high_frequency - mel_low_frequency) / (num_bins + 1);

    mel_bins_.resize(num_bins);
    center_frequency_.resize(num_bins);
    for (size_t index = 0; index < num_bins; index++) {
        float left_mel = mel_low_frequency + index * mel_frequency_delta;
        float middle_mel = mel_low_frequency + (index + 1) * mel_frequency_delta;
        float right_mel = mel_low_frequency + (index + 2) * mel_frequency_delta;
        center_frequency_[index] = inverse_mel_scale(middle_mel);

        std::vector<float> cur_mel_bin(num_fft_bins);
        int first_index = -1, last_index = -1;
        for (int i = 0; i < num_fft_bins; i++) {
            float frequency = (fft_bin_width * i);
            float mel = mel_scale(frequency);
            if (mel > left_mel && mel < right_mel) {
                float weight = 0.0;
                if (mel <= middle_mel) {
                    weight = (mel - left_mel) / (middle_mel - left_mel);
                } else {
                    weight = (right_mel - mel) / (right_mel - middle_mel);
                }
                cur_mel_bin[i] = weight;
                if (first_index == -1) first_index = i;
                last_index = i;
            }
        }
        mel_bins_[index].first = first_index;
        mel_bins_[index].second.resize(last_index + 1 - first_index);
        mel_bins_[index].second.assign(cur_mel_bin.begin() + first_index,
                                       cur_mel_bin.begin() + last_index + 1);
    }
}

void speakerlab::subtract_feature_mean(std::vector<std::vector<float>>& feature) {
    if (feature.empty() || feature[0].empty()) return;
    size_t feat_dim = feature[0].size();
    std::vector<float> means(feat_dim, 0.0f);

    for (const auto& feature_vector : feature) {
        for (size_t i = 0; i < feat_dim; ++i) {
            means[i] += feature_vector[i];
        }
    }
    for (float& mean : means) {
        mean /= (float)feature.size();
    }

    // subtract feature mean
    for (auto& feature_vector : feature) {
        for (size_t i = 0; i < feat_dim; ++i) {
            feature_vector[i] -= means[i];
        }
    }
}
