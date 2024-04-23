//
// Created by shuli on 2024/3/5.
//

#ifndef SPEAKERLABENGINES_FEATURE_BASIC_H
#define SPEAKERLABENGINES_FEATURE_BASIC_H

#include <string>
#include <sstream>
#include "nlohmann/json.hpp"

namespace speakerlab{

    struct FrameExtractionOptions {
        float sample_freq;
        float frame_shift_ms; // frame shift in million seconds
        float frame_length_ms; // frame length in million seconds
        float dither;
        float pre_emphasis_coefficient;
        bool remove_dc_offset;
        std::string window_type;
        bool round_to_power_of_two;
        float blackman_coefficient;
        bool snip_edges;
        bool allow_down_sample;
        bool allow_up_sample;
        int max_feature_vectors;

        explicit FrameExtractionOptions() :
                sample_freq(16000),
                frame_shift_ms(10.0),
                frame_length_ms(25.0),
                dither(1.0),
                pre_emphasis_coefficient(0.97),
                remove_dc_offset(true),
                window_type("povey"),
                round_to_power_of_two(true),
                blackman_coefficient(0.42),
                snip_edges(true),
                allow_down_sample(false),
                allow_up_sample(false),
                max_feature_vectors(-1) {}

        int compute_window_shift();
        int compute_window_size();
        int padded_window_size();

        static FrameExtractionOptions load_from_json(const nlohmann::json &json_dict);

        // show all the parameters
        std::string show() const {
            std::ostringstream oss;
            oss << "FrameExtractionOptions [ " << "sample_freq: " << sample_freq << "\t"
                << "frame_shift_ms: " << frame_shift_ms << "\t"
                << "frame_length_ms: " << frame_length_ms << "\t"
                << "dither: " << dither << "\t"
                << "pre_emphasis_coefficient: " << pre_emphasis_coefficient << "\t"
                << "remove_dc_offset: " << (remove_dc_offset ? "true" : "false") << "\t"
                << "window_type: " << window_type << "\t"
                << "round_to_power_of_two: " << (round_to_power_of_two ? "true" : "false") << "\t"
                << "blackman_coefficient: " << blackman_coefficient << "\t"
                << "snip_edges: " << (snip_edges ? "true" : "false") << "\t"
                << "allow_down_sample: " << (allow_down_sample ? "true" : "false") << "\t"
                << "allow_up_sample: " << (allow_up_sample ? "true" : "false") << "\t"
                << "max_feature_vectors: " << max_feature_vectors << " ]";
            return oss.str();
        }
    };

    struct MelBanksOptions {
        int num_bins;
        float low_freq;
        float high_freq;
        float vtln_low;
        float vtln_high;

        explicit MelBanksOptions(int num_bins = 25) :
                num_bins(num_bins),
                low_freq(20),
                high_freq(0),
                vtln_low(100),
                vtln_high(-500) {}
        static MelBanksOptions load_from_json(const nlohmann::json &json_dict);
        
        std::string show() const {
            std::ostringstream oss;
            oss << "MelBanksOptions [ "<< "num_bins: " << num_bins << "\t"
                << "low_freq: " << low_freq << "\t"
                << "high_freq: " << high_freq << "\t"
                << "vtln_low: " << vtln_low << "\t"
                << "vtln_high: " << vtln_high << " ]";
            return oss.str();
        }
    };
}

#endif //SPEAKERLABENGINES_FEATURE_BASIC_H
