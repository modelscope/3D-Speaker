//
// Created by shuli on 2024/3/4.
//

#include "feature/feature_fbank.h"
#include "feature/feature_functions.h"


speakerlab::FbankComputer::FbankComputer(const speakerlab::FbankOptions &opts) : opts_(opts),
                                                                                 frame_preprocessor_(opts.frame_opts),
                                                                                 log_energy_floor_(0.0),
                                                                                 mel_bank_processor_(opts.mel_opts) {
    int frame_length = opts_.frame_opts.compute_window_size();
    const int fft_n = round_up_to_nearest_power_of_two(frame_length);
    init_sin_tbl(sin_tbl_, fft_n);
    init_bit_reverse_index(bit_rev_index_, fft_n);

    int padded_window_length = opts_.frame_opts.padded_window_size();
    mel_bank_processor_.init_mel_bins(opts.frame_opts.sample_freq, padded_window_length);
}

speakerlab::Feature speakerlab::FbankComputer::compute_feature(WavReader wav_reader) {
    bool check_wav = check_wav_and_config(wav_reader);
    if (!check_wav) {
        throw std::invalid_argument("WavReader is negative");
    }

    int frame_length = opts_.compute_window_size();
    int frame_shift = opts_.compute_window_shift();
    int fft_n = round_up_to_nearest_power_of_two(frame_length);
    speakerlab::Wave wav_data = wav_reader.get_float_wav_data();
    int num_samples = wav_data.size();
    int num_frames = 1 + ((num_samples - frame_length) / frame_shift);
    speakerlab::Feature feature;
    feature.resize(num_frames);

    float epsilon = std::numeric_limits<float>::epsilon();
    int fbank_num_bins = opts_.get_fbank_num_bins();
    std::vector<std::pair<int, std::vector<float>>> mel_bins = mel_bank_processor_.get_mel_bins();
    std::cout << "frame_length: " << frame_length << " "
              << "frame_shift: " << frame_shift << " "
              << "fft_n: " << fft_n << " "
              << "num_frames: " << num_frames << " "
              << "mel_bins: " << mel_bins.size() << " "
              << "fbank_num_bins " << fbank_num_bins << " "
              << "epsilon: " << epsilon
              << std::endl;

    for (int i = 0; i < num_frames; i++) {
        std::vector<float> cur_wav_data(wav_data.data() + i * frame_shift,
                                        wav_data.data() + i * frame_shift + frame_length);
        // Contain dither,
        frame_preprocessor_.frame_pre_process(cur_wav_data);

        // build FFT
        std::vector<std::complex<float>> cur_window_data(fft_n);
        for (int j = 0; j < fft_n; j++) {
            if (j < frame_length) {
                cur_window_data[j] = std::complex<float>(cur_wav_data[j], 0.0);
            } else {
                cur_window_data[j] = std::complex<float>(0.0, 0.0);
            }
        }
        custom_fft(bit_rev_index_, sin_tbl_, cur_window_data);
        std::vector<float> power(fft_n / 2);
        for (int j = 0; j < fft_n / 2; j++) {
            power[j] = cur_window_data[j].real() * cur_window_data[j].real() +
                       cur_window_data[j].imag() * cur_window_data[j].imag();
        }
        if (!opts_.use_power) {
            for (int j = 0; j < fft_n / 2; j++) {
                power[j] = powf(power[j], 0.5);
            }
        }
        // mel filter
        feature[i].resize(opts_.get_fbank_num_bins());
        for (int j = 0; j < fbank_num_bins; j++) {
            float mel_energy = 0.0;
            int start_index = mel_bins[j].first;
            for (int k = 0; k < mel_bins[j].second.size(); k++) {
                mel_energy += mel_bins[j].second[k] * power[k + start_index];
            }
            if (opts_.use_log_fbank) {
                if (mel_energy < epsilon) mel_energy = epsilon;
                mel_energy = logf(mel_energy);
            }
            feature[i][j] = mel_energy;
        }
    }
    return feature;
}

bool speakerlab::FbankComputer::check_wav_and_config(const WavReader &wav_reader) {
    if (wav_reader.num_channel() != 1) {
        std::cerr << "Num channel " << wav_reader.num_channel() << " not equal to 1" << std::endl;
        return false;
    }
    int window_size = opts_.compute_window_size();
    int window_shift = opts_.compute_window_shift();
    int padded_window_size = opts_.paddle_window_size();

    if (window_size < 2 || window_size > wav_reader.num_sample()) {
        std::cerr << "Choose a window size " << window_size << " that is [2,  " << wav_reader.num_sample() << "]"
                  << std::endl;
        return false;
    }
    if (window_shift <= 0) {
        std::cerr << "Window shift " << window_shift << " must be greater than 0" << std::endl;
        return false;
    }
    if (padded_window_size % 2 == 1) {
        std::cerr << "The padded `window_size` must be divisible by two.";
        return false;
    }
    if (opts_.frame_opts.pre_emphasis_coefficient < 0.0 || opts_.frame_opts.pre_emphasis_coefficient > 1.0) {
        std::cerr << "Pre-emphasis coefficient " << opts_.frame_opts.pre_emphasis_coefficient
                  << " must be between [0, 1]" << std::endl;
        return false;
    }
    return true;
}

speakerlab::FbankOptions speakerlab::FbankOptions::load_from_json(const nlohmann::json &json_dict) {
    speakerlab::FbankOptions fbank_opts;
    fbank_opts.frame_opts = speakerlab::FrameExtractionOptions::load_from_json(json_dict["FrameExtractionOptions"]);
    fbank_opts.mel_opts = speakerlab::MelBanksOptions::load_from_json(json_dict["MelBanksOptions"]);
    fbank_opts.use_energy = json_dict.value<bool>("use_energy", false);
    fbank_opts.energy_floor = json_dict.value<float>("energy_floor", 0.0);
    fbank_opts.raw_energy = json_dict.value<bool>("raw_energy", true);
    fbank_opts.use_log_fbank = json_dict.value<bool>("use_log_fbank", true);
    fbank_opts.use_power = json_dict.value<bool>("use_power", true);
    return fbank_opts;
}
