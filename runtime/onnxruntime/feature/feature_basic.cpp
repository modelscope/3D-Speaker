//
// Created by shuli on 2024/3/5.
//

#include "feature_basic.h"
#include "feature_functions.h"

int speakerlab::FrameExtractionOptions::compute_window_shift() {
    return static_cast<int>(sample_freq * 0.001 * frame_shift_ms);
}

int speakerlab::FrameExtractionOptions::compute_window_size() {
    return static_cast<int>(sample_freq * 0.001 * frame_length_ms);
}

int speakerlab::FrameExtractionOptions::padded_window_size() {
    int window_size = compute_window_size();
    if(round_to_power_of_two) {
        return round_up_to_nearest_power_of_two(window_size);
    }
    else {
        return window_size;
    }
}

speakerlab::FrameExtractionOptions speakerlab::FrameExtractionOptions::load_from_json(const nlohmann::json &json_dict) {
    FrameExtractionOptions frame_opts;

    frame_opts.sample_freq = json_dict.value<float>("sample_freq", 16000);
    frame_opts.frame_shift_ms = json_dict.value<float>("frame_shift_ms", 10.0);
    frame_opts.frame_length_ms = json_dict.value<float>("frame_length_ms", 25.0);
    frame_opts.dither = json_dict.value<float>("dither", 1.0);
    frame_opts.pre_emphasis_coefficient = json_dict.value<float>("pre_emphasis_coefficient", 0.97);
    frame_opts.remove_dc_offset = json_dict.value<bool>("remove_dc_offset", true);
    frame_opts.window_type = json_dict.value<std::string>("window_type", "povey");
    frame_opts.round_to_power_of_two = json_dict.value<bool>("round_to_power_of_two", true);
    frame_opts.blackman_coefficient = json_dict.value<float>("blackman_coefficient", 0.42);
    frame_opts.snip_edges = json_dict.value<bool>("snip_edges", true);
    frame_opts.allow_down_sample = json_dict.value<bool>("allow_down_sample", false);
    frame_opts.allow_up_sample = json_dict.value<bool>("allow_up_sample", false);
    frame_opts.max_feature_vectors = json_dict.value<int>("max_feature_vectors", -1);

    return frame_opts;
}


speakerlab::MelBanksOptions speakerlab::MelBanksOptions::load_from_json(const nlohmann::json &json_dict) {
    speakerlab::MelBanksOptions mel_opts;

    mel_opts.num_bins = json_dict.value<int>("num_bins", 80);
    mel_opts.low_freq = json_dict.value<float>("low_freq", 20);
    mel_opts.high_freq = json_dict.value<float>("high_freq", 0);
    mel_opts.vtln_low = json_dict.value<float>("vtln_low", 100);
    mel_opts.vtln_high = json_dict.value<float>("vtln_high", -500);

    return mel_opts;
}
