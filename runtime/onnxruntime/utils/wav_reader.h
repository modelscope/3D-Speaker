//
// Created by shuli on 2024/3/4.
//

#ifndef SPEAKERLABENGINES_WAV_READER_H
#define SPEAKERLABENGINES_WAV_READER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <limits>


namespace speakerlab {
    // abstract class for possible various audio type
    class AudioReader {
    public:
        AudioReader() = default;

        explicit AudioReader(const std::string &audio_filename) {};

        virtual ~AudioReader() = default;

        virtual int num_channel() const = 0;

        virtual uint32_t sample_rate() const = 0;

        virtual uint32_t bits_per_sample() const = 0;

        virtual uint32_t num_sample() const = 0;
    };

    // WavHeader: RIFF Format Header
    struct WavHeader {
        char riff_header[4];
        uint32_t file_size;
        char wave_header[4];
        char fmt_header[4];
        uint32_t fmt_chunk_size;
        uint16_t audio_format;
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
    };


    /*
    * Usage should be like:
    *  WavReader* wav_reader = new WavReader(path_to_wav);
    *  wav_reader
    */
    class WavReader : public AudioReader {
    public:
        WavReader() {};

        explicit WavReader(const std::string &wav_filename) : wav_filename_(wav_filename), is_valid_(false) {
            load_wav(wav_filename);
        }

        int num_channel() const override { return wav_header_.num_channels; }

        uint32_t sample_rate() const override { return wav_header_.sample_rate; }

        uint32_t bits_per_sample() const override { return wav_header_.bits_per_sample; }

        uint32_t num_sample() const override { return num_sample_; }

        bool is_valid() const { return is_valid_; }

        void load_wav(const std::string &wav_filename);

        std::vector<char> &get_raw_wav_data() { return wav_data_; }

        std::vector<int16_t> get_int16_wav_data();

        std::vector<float> get_float_wav_data();
    private:
        WavHeader wav_header_;
        std::vector<char> wav_data_;
//        std::vector<int16_t> wav_data_int16_;
        std::string wav_filename_;
        uint32_t num_sample_;
        bool is_valid_;
    };
}

#endif //SPEAKERLABENGINES_WAV_READER_H
