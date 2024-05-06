set(ONNX_RUNTIME_VERSION "1.12.0")

if (WIN32)
    set(ONNX_RUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-win-x64-${ONNX_RUNTIME_VERSION}.zip")
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ONNX_RUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNX_RUNTIME_VERSION}.tgz")
    else ()
        set(ONNX_RUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNX_RUNTIME_VERSION}.tgz")
    endif ()
elseif(UNIX AND NOT APPLE)
    set(ONNX_RUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}.tgz")
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()

FetchContent_Declare(onnxruntime
        URL ${ONNX_RUNTIME_URL}
        URL_HASH ${URL_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
    FetchContent_Populate(onnxruntime)
endif()

set(ONNX_RUNTIME_INCLUDE_DIRS ${onnxruntime_SOURCE_DIR}/include)
set(ONNX_RUNTIME_LIB_DIRS ${onnxruntime_SOURCE_DIR}/lib)

include_directories(${ONNX_RUNTIME_INCLUDE_DIRS})
link_directories(${ONNX_RUNTIME_LIB_DIRS})
