#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "utils/wav_reader.h"
//#include "feature/feature_basic.h"
#include "feature/feature_fbank.h"
#include "nlohmann/json.hpp"

std::pair<std::string, std::string>
extract_and_save_fbank_feature(const std::string &utt_id,
                               const std::string &wav_file,
                               const std::string &feature_save_path,
                               speakerlab::Feature &feature,
                               std::shared_ptr<speakerlab::FbankComputer> fbank_ptr) {
    speakerlab::WavReader wav_reader(wav_file);
    // std::vector<int16_t> wav_data = wav_reader.get_int16_wav_data();
    std::cout << "Read wav finished for utt_id " << utt_id << std::endl;

    feature = fbank_ptr -> compute_feature(wav_reader);
    std::string target_feature_file(feature_save_path + "/" + utt_id + ".feature");

    std::cout << "Extract feature finished for utt_id " << utt_id << std::endl;
    std::ofstream fw(target_feature_file);
    fw << std::fixed << std::setprecision(6); // precision 6

    for (auto &frame: feature){
        for (float &point : frame) {
            fw << point << " ";
        }
        fw << std::endl;
    }
    fw.close();
    return std::make_pair(utt_id, target_feature_file);
}

speakerlab::FbankOptions load_json_config(const std::string &fbank_config_file) {
    std::ifstream fr(fbank_config_file);
    nlohmann::json config;
    fr >> config;
    return speakerlab::FbankOptions::load_from_json(config);
}


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

int write_wav_scp(const std::string &wav_scp_file, std::map<std::string, std::string> wav_scp){
    std::ofstream ofs(wav_scp_file);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << wav_scp_file << std::endl;
        return 1;
    }
    for (const auto& kv : wav_scp) {
        ofs << kv.first << " " << kv.second << std::endl;
    }
    ofs.close();
    return 0;
}


int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "make_fbank_feature.cpp -> Usage: " << argv[0]
                  << " <feature_config_file> <wav_scp_file> <feature_scp_file> <feature_save_path>, "
                  << "just a demo and save fbank to text format."
                  << std::endl;
        return 1;
    }

    std::string feature_config_file(argv[1]);
    std::string wav_scp_file(argv[2]);
    std::string feature_scp_file(argv[3]);
    std::string feature_save_path(argv[4]);

    std::cout << "Feature config file: " << feature_config_file << "\n"
              << "Wav scp file: " << wav_scp_file << "\n"
              << "Feature scp file: " << feature_scp_file << "\n"
              << "Feature save path: " << feature_save_path << std::endl;

    // read wav_scp -> map<string, string> wav_scp
    std::map<std::string, std::string> wav_scp = read_wav_scp(wav_scp_file);
    std::cout << "Read wav.scp file finished: " << wav_scp_file << " collect " << wav_scp.size() << " items."<< std::endl; 

    // load config file
    speakerlab::FbankOptions opts = load_json_config(feature_config_file);
    auto fbank_computer = std::make_shared<speakerlab::FbankComputer>(opts);
    std::cout << "Build fbank computer finished" << std::endl;
    std::cout << opts.show() << std::endl;

    std::map<std::string, std::string> feature_scp;
    for (auto & it : wav_scp) {
        speakerlab::Feature cur_feature;
        std::string utt_id = it.first;
        std::cout << "start make fbank feature for " << utt_id << std::endl;
        std::pair<std::string, std::string> feature_entry = extract_and_save_fbank_feature(
                utt_id, it.second, feature_save_path, cur_feature, fbank_computer);
        std::cout << "Finish make fbank feature for " << utt_id << std::endl;
        feature_scp[feature_entry.first] = feature_entry.second;
    }

    int status_state = write_wav_scp(feature_scp_file, feature_scp);
    if (status_state == 1) {
        std::cerr << "Write wav.scp " << wav_scp_file << " failed" << std::endl;
        return 1;
    }
    else{
        std::cout << "Write wav.scp " << wav_scp_file << " succeed" << std::endl;
        return 0;
    }
}
