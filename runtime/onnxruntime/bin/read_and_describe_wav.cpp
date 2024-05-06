#include <iostream>
#include <string>
#include <vector>

#include "utils/wav_reader.h"


int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <wav_filename> " << std::endl;
        return 1;
    }

    std::string wav_filename(argv[1]);

    speakerlab::WavReader wav_reader(wav_filename);

    if (wav_reader.is_valid()) {
        // Show some information
        std::cout << "Sample rate: " << wav_reader.sample_rate() << " "
                  << "Num Channel: " << wav_reader.num_channel() << " "
                  << "Num Sample: " << wav_reader.num_sample() << " "
                  << "Duration: " << (double) wav_reader.num_sample() / (double) wav_reader.sample_rate() << " "
                  << "Bits Per Sample: " << wav_reader.bits_per_sample() << std::endl;
        std::vector<float> wav = wav_reader.get_float_wav_data();
    } else {
        std::cerr << "Wav " << wav_filename << " is not valid" << std::endl;
    }

    return 0;
}