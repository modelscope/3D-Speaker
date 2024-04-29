//
// Created by shuli on 2024/3/12.
//
#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <ctime>

#include "utils/wav_reader.h"
#include "feature/feature_fbank.h"
#include "feature/feature_common.h"
#include "model/speaker_embedding_model.h"


std::map<std::string, std::string>
read_wav_scp(const std::string &wav_scp_file) {
    std::ifstream fr(wav_scp_file);
    if (!fr.is_open()) {
        std::cerr << "Unable to open file: " + wav_scp_file << std::endl;
        return {};
    }
    std::map<std::string, std::string> wav_scp;
    std::string line, utt_id, wav_file;
    while (std::getline(fr, line)) {
        std::istringstream iss(line);
        if (!(iss >> utt_id >> wav_file)) {
            continue;
        }
        if (wav_scp.find(utt_id) == wav_scp.end()) {
            wav_scp[utt_id] = wav_file;
        } else {
            throw std::runtime_error("Invalid wav_scp_file : utt_id " + utt_id + " repeated\n");
        }
    }
    return wav_scp;
}


int write_wav_scp(const std::string &wav_scp_file, std::map<std::string, std::string> wav_scp) {
    std::ofstream ofs(wav_scp_file);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << wav_scp_file << std::endl;
        return 1;
    }
    for (const auto &kv: wav_scp) {
        ofs << kv.first << " " << kv.second << std::endl;
    }
    ofs.close();
    return 0;
}


int write_embedding(const std::string &embedding_file, speakerlab::Embedding &embedding) {
    std::ofstream fw(embedding_file);
    if (!fw.is_open()) {
        std::cerr << "Error: Could not open embedding file " << embedding_file << std::endl;
        return 1;
    }
    for (size_t i = 0; i < embedding.size(); ++i) {
        fw << embedding[i];
        if (i != embedding.size() - 1) {
            fw << " ";
        }
    }
    fw << std::endl;
    fw.close();
    return 0;
}


std::string normalize_for_path(const std::string utt_id) {
    std::string result = utt_id;
    std::replace(result.begin(), result.end(), '/', '-');
    return result;
}


int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <config_file> "
                  << "<onnx_file> <wav_scp_file> <embedding_scp_file> <embedding_save_path>" << std::endl;
        return 1;
    }

    std::string config_file(argv[1]);
    std::string onnx_file(argv[2]);
    std::string wav_scp_file(argv[3]);
    std::string embedding_scp_file(argv[4]);
    std::string embedding_save_path(argv[5]);

    // read wav.scp file
    std::map<std::string, std::string> wav_scp = read_wav_scp(wav_scp_file);

    // load feature extractor
    std::ifstream fr(config_file);
    nlohmann::json config;
    fr >> config;
    speakerlab::FbankOptions fbank_opts = speakerlab::FbankOptions::load_from_json(config);
    speakerlab::FbankComputer fbank_computer = speakerlab::FbankComputer(fbank_opts);

    // load onnx model
    speakerlab::OnnxSpeakerEmbeddingModel speaker_embedding_extractor(onnx_file);

    std::map<std::string, std::string> embedding_scp;
    std::clock_t start = std::clock();
    double total_wav_duration = 0.0;
    for (auto &it: wav_scp) {
        std::string utt_id = it.first;
        std::string wav_file = it.second;
        speakerlab::WavReader wav_reader(wav_file);
        total_wav_duration += static_cast<double>(wav_reader.num_sample()) / wav_reader.sample_rate();
        speakerlab::Feature feature = fbank_computer.compute_feature(wav_reader);
        speakerlab::subtract_feature_mean(feature);

        speakerlab::Embedding embedding;
        speaker_embedding_extractor.extract_embedding(feature, embedding);
        std::string cur_embedding_file = embedding_save_path + "/" + normalize_for_path(utt_id) + ".embedding";
        write_embedding(cur_embedding_file, embedding);
        embedding_scp[utt_id] = cur_embedding_file;
    }
    std::clock_t finish = std::clock();
    double elapsed = static_cast<double>(finish - start) / CLOCKS_PER_SEC;

    std::cout << "Elapsed time: " << elapsed << "s for wav duration " << total_wav_duration << std::endl;

    // write wav.scp file
    write_wav_scp(embedding_scp_file, embedding_scp);

    return 0;
}
