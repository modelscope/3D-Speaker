//
// Created by shuli on 2024/3/12.
//

#include "speaker_embedding_model.h"

speakerlab::OnnxSpeakerEmbeddingModel::OnnxSpeakerEmbeddingModel(const std::string &onnx_file)
        : env_(ORT_LOGGING_LEVEL_WARNING, "speakerlab_onnxruntime") {
    session_options_.SetIntraOpNumThreads(1);
    session_ptr_ = std::make_shared<Ort::Session>(env_, onnx_file.c_str(), session_options_);
}

void speakerlab::OnnxSpeakerEmbeddingModel::describe_embedding_model() {
    size_t num_input_nodes = session_ptr_->GetInputCount();
    std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
    for (size_t i = 0; i < num_input_nodes; i++) {
        char *input_name = session_ptr_->GetInputName(i, Ort::AllocatorWithDefaultOptions());
        Ort::TypeInfo type_info = session_ptr_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << "Input " << i << " : name=" << input_name << std::endl;
        Ort::AllocatorWithDefaultOptions().Free(input_name);
    }
}

void speakerlab::OnnxSpeakerEmbeddingModel::extract_embedding(const speakerlab::Feature &feature,
                                                              speakerlab::Embedding &embedding) {
    // Feature -> Tensor
    if (feature.empty() || feature[0].empty()) {
        std::cerr << "Error: Feature is empty" << std::endl;
        return;
    }
    size_t frame_num = feature.size();
    size_t feature_dim = feature[0].size();

    std::vector<int64_t> input_tensor_shape = {1, static_cast<int64_t>(frame_num), static_cast<int64_t>(feature_dim)};

    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(frame_num * feature_dim);
    for (const auto &frame: feature) {
        input_tensor_values.insert(input_tensor_values.end(), frame.begin(), frame.end());
    }

    // Create the tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_tensor_values.data(),
                                                              input_tensor_values.size(),
                                                              input_tensor_shape.data(),
                                                              input_tensor_shape.size());

    // Define nodes, please match the export scripts
    const char *input_nodes[] = {"feature"};
    const char *output_nodes[] = {"embedding"};

    // Run
    auto output_tensors = session_ptr_->Run(Ort::RunOptions{nullptr},
                                            input_nodes,
                                            &input_tensor,
                                            1,
                                            output_nodes,
                                            1);

    // save output tensors to std::vector<float> embedding
    auto *float_arr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_tensor_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "Output embedding size = " << output_tensor_size << std::endl;

    embedding.assign(float_arr, float_arr + output_tensor_size);
}


