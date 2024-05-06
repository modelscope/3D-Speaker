add_library(utils STATIC
    wav_reader.cpp
)
target_link_libraries(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})