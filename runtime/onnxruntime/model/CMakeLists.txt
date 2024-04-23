add_library(model STATIC speaker_embedding_model.cpp)

target_include_directories(model PUBLIC ${ONNX_RUNTIME_INCLUDE_DIRS})
target_link_directories(model PUBLIC ${ONNX_RUNTIME_LIB_DIRS})
target_link_libraries(model PRIVATE onnxruntime)
