//
// Created by shuli on 2024/3/4.
//

#ifndef SPEAKERLABENGINES_FEATURE_FBANK_H
#define SPEAKERLABENGINES_FEATURE_FBANK_H

#include "utils/wav_reader.h"
#include "feature/feature_common.h"
#include "feature/feature_basic.h"
#include "feature/feature_common.h"

#include <map>
#include <vector>
#include <complex>

namespace speakerlab {
    typedef std::vector<std::vector<float>> Feature;
    typedef std::vector<float> Wave;

    struct FbankOptions {
        FrameExtractionOptions frame_opts;
        MelBanksOptions mel_opts;
        bool use_energy;
        float energy_floor;
        bool raw_energy;
        bool use_log_fbank;
        bool use_power;

        explicit FbankOptions() :
                mel_opts(80),
                use_energy(false),
                energy_floor(0.0),
                raw_energy(true),
                use_log_fbank(true),
                use_power(true) {}

        inline int compute_window_shift() { return frame_opts.compute_window_shift(); }

        inline int compute_window_size() { return frame_opts.compute_window_size(); }

        inline int paddle_window_size() { return frame_opts.padded_window_size(); }

        inline int get_fbank_num_bins() { return mel_opts.num_bins; }

        static FbankOptions load_from_json(const nlohmann::json &json_dict);

        std::string show() const {
            std::string frame_str = frame_opts.show();
            std::string mel_str = mel_opts.show();
            std::ostringstream oss;
            oss << "FbankOptions [ " << frame_str << "\n" << mel_str << "\n"
                << "use_energy: " << (use_energy ? "true" : "false") << "\t"
                << "energy_floor: " << energy_floor << "\t"
                << "raw_energy: " << (raw_energy ? "true" : "false") << "\t"
                << "use_log_fbank: " << (use_log_fbank ? "true" : "false") << "\t"
                << "use_power: " << (use_power ? "true" : "false") << "]";

            return oss.str();
        }
    };

    class FbankComputer {
    public:
        FbankComputer() {};

        explicit FbankComputer(const FbankOptions &opts);

        Feature compute_feature(WavReader wav_reader);

        bool check_wav_and_config(const WavReader &wav_reader);

    private:
        FbankOptions opts_;
        FramePreprocessor frame_preprocessor_;
        MelBankProcessor mel_bank_processor_;
        float log_energy_floor_;
        std::vector<int> bit_rev_index_;
        std::vector<float> sin_tbl_;
    };
}

#endif //SPEAKERLABENGINES_FEATURE_FBANK_H
