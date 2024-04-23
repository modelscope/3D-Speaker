//
// Created by shuli on 2024/3/12.
//

#ifndef SPEAKERLABENGINES_SPEAKER_EMBEDDING_MODEL_H
#define SPEAKERLABENGINES_SPEAKER_EMBEDDING_MODEL_H

#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <functional>
#include "onnxruntime_cxx_api.h"

namespace speakerlab {
    typedef std::vector<float> Embedding;
    typedef std::vector<std::vector<float>> Feature;

    class BasicSpeakerEmbeddingModel {
    public:

        virtual void extract_embedding(const Feature &feature, Embedding &embedding) {}

        virtual void describe_embedding_model() {}
    };

    class OnnxSpeakerEmbeddingModel : public BasicSpeakerEmbeddingModel {
    public:
        explicit OnnxSpeakerEmbeddingModel(const std::string &onnx_file);

        void describe_embedding_model() override;

        void extract_embedding(const speakerlab::Feature &feature, speakerlab::Embedding &embedding) override;

    private:
        // Ort::Session do not have default constructor, use point instead
        std::shared_ptr<Ort::Session> session_ptr_;
        Ort::Env env_;
        Ort::SessionOptions session_options_;
    };
}


#endif //SPEAKERLABENGINES_SPEAKER_EMBEDDING_MODEL_H
