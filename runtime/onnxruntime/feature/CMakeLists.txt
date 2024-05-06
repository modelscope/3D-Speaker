add_library(feature STATIC
    feature_basic.cpp
        feature_fbank.h
        feature_functions.cpp
        feature_fbank.cpp
        feature_common.cpp
)
target_link_libraries(feature PUBLIC utils)