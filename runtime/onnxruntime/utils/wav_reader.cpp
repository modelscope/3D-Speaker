//
// Created by shuli on 2024/3/4.
//

#include "utils/wav_reader.h"

void speakerlab::WavReader::load_wav(const std::string &wav_filename) {
    std::ifstream in_file(wav_filename, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Could not open file: " << wav_filename << std::endl;
        return;
    }
    // read wav header
    in_file.read(reinterpret_cast<char *>(&wav_header_), sizeof(WavHeader));
    // check wav head
    if (strncmp(wav_header_.riff_header, "RIFF", 4) != 0 ||
        strncmp(wav_header_.wave_header, "WAVE", 4) != 0) {
        std::cerr << "Invalid file " << wav_filename << std::endl;
        return;
    }
    // skip the head
    char data_buffer[4];
    uint32_t data_size;
    do {
        in_file.read(data_buffer, sizeof(data_buffer));
        in_file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));
        if (strncmp(data_buffer, "data", 4) != 0) {
            in_file.seekg(data_size, std::ios::cur);
        }
    } while (strncmp(data_buffer, "data", 4) != 0);

    wav_data_.resize(data_size);
    in_file.read(wav_data_.data(), data_size);
    uint32_t bytes_per_sample = wav_header_.bits_per_sample / 8; // 每个样本的字节数
    uint32_t frame_size = bytes_per_sample * wav_header_.num_channels; // 每帧的字节数
    num_sample_ = data_size / frame_size; // 实际样本数
    is_valid_ = true;
}

std::vector<int16_t> speakerlab::WavReader::get_int16_wav_data() {
    size_t num_samples = wav_data_.size() / 2;
    std::vector<int16_t> int16_audio_data(num_samples);
    memcpy(int16_audio_data.data(), wav_data_.data(), wav_data_.size());
    return int16_audio_data;
}

std::vector<float> speakerlab::WavReader::get_float_wav_data() {
    size_t num_samples = wav_data_.size() / 2;
    std::vector<int16_t> int16_audio_data(num_samples);
    memcpy(int16_audio_data.data(), wav_data_.data(), wav_data_.size());
    std::vector<float> float_audio_data(num_samples);
    const auto max_val = static_cast<float>(std::numeric_limits<int16_t>::max());
    for (size_t i = 0; i < num_samples; ++i) {
        float_audio_data[i] = int16_audio_data[i] / max_val;
    }
    return float_audio_data;
}
