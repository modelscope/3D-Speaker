# SpeakerLab Engines
> We provide a simple speaker ONNX Runtime along with related scripts to facilitate the construction of an engine for extracting speaker embeddings. This is a sample project and, for ease of use, we have strived to keep the code as simple and clear as possible while minimizing the reliance on third-party libraries, allowing you to effortlessly port them into your project code.


## Installation and Usage

1. Step-1: Export Model to ONNX 

Please see the scripts `speakerlab/bin/export_speaker_embedding_onnx.py`. Here is a simple example:
```shell
# Please install onnx in your python environment before
python speakerlab/bin/export_speaker_embedding_onnx.py \
    --experiment_path your/experiment_path/ \
    --model_id iic/speech_eres2net_sv_en_voxceleb_16k \ # you can use other model_id
    --target_onnx_file path/to/save/onnx_model
```
Now you have exported the speaker embedding model to the onnx file.

The core part is the following export functions:
```python
def export_onnx_file(model, target_onnx_file):
    dummy_input = torch.randn(1, 345, 80) # You can change the shape
    torch.onnx.export(model,
                      dummy_input,
                      target_onnx_file,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      # Note: the input and output names must match in the ONNX Runtime
                      input_names=['feature'],
                      output_names=['embedding'],
                      # Set dynamic axes for batch size and frame num
                      dynamic_axes={'feature': {0: 'batch_size', 1: 'frame_num'},
                                    'embedding': {0: 'batch_size'}})
```
You can also export your own model to ONNX format and try it in our framework. If you have any questions, please contact us.

Our environment is on Linux, and the key packages' version are like(`pip show torch onnx onnxruntime`):
```text
Name: torch
Version: 1.13.1
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Requires: nvidia-cublas-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11, typing-extensions
Required-by: torchaudio
---
Name: onnx
Version: 1.14.1
Summary: Open Neural Network Exchange
Home-page: https://github.com/onnx/onnx
Author: ONNX
Author-email: onnx-technical-discuss@lists.lfaidata.foundation
License: Apache License v2.0
Requires: numpy, protobuf, typing-extensions
---
Name: onnxruntime
Version: 1.16.1
Summary: ONNX Runtime is a runtime accelerator for Machine Learning models
Home-page: https://onnxruntime.ai
Author: Microsoft Corporation
Author-email: onnxruntime@microsoft.com
License: MIT License
Requires: coloredlogs, flatbuffers, numpy, packaging, protobuf, sympy
```

There could be some problems when the environment is not matched, you can add the following code in `speakerlab/bin/export_speaker_embedding_onnx.py` to check whether the onnx model you export is correct.
```python
def main():
    args = get_args()
    logger.info(f"{args}")

    model_id = args.model_id
    experiment_path = args.experiment_path
    target_onnx_file = args.target_onnx_file
    if model_id is not None:
        speaker_embedding_model = build_model_from_modelscope_id(
            model_id, experiment_path
        )
    else:
        speaker_embedding_model = build_model_from_custom_work_path(
            experiment_path
        )

    logger.info(f"Load speaker embedding finished, export to onnx")
    # let function `export_onnx_file` return the random tensor
    inputs = export_onnx_file(speaker_embedding_model, target_onnx_file)

    with torch.no_grad():
        res0 = speaker_embedding_model(inputs)
    ort_sess = ort.InferenceSession(target_onnx_file)
    res1 = ort_sess.run(None, {'feature': inputs.numpy()})[0]
    res1 = torch.from_numpy(res1) # Here, convert it to torch.tensor

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    print(cos(res0, res1)) # The expected result should be tensor([1.0000])
```


2. Step-2: Compile ONNX Runtime Project
```shell
# Please install cmake and gcc in your system before.
cd runtime/onnxruntime/
mkdir build/ # you can change the folder name
cd build/
cmake ..
make
```
Please note that in `cmake/build_onnx.cmake` you can find:
```shell
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
```

We have **ONLY** tested in Linux and we will soon finish the testing process in other operaing systems. We have provided the URL for downloading the ONNX Runtime binary file according to your operating system.

If everything is ok, you can find the binary file in `build/bin/`: `extract_speaker_embedding`, `make_fbank_feature` and `read_and_describe_wav`.

3. Step-3: Extract Speaker Embeddings
Now you can extract embeddings! In our project, you should prepare a `wav.scp` file before which is format as:
```text
utt_id_1 /path/to/wav_1.wav
utt_id_2 /path/to/wav_2.wav
....
```
The usage for binary file `extract_speaker_embedding` will be like:
```shell
./extract_speaker_embedding path/to/fbank_config.json path/to/your/onnx_file /path/to/your/wav.scp /path/to/embedding_scp_file /path/to/save/embeddings/
```
The `fbank_config.json` is the config file for extracting FBank features and we provide a standard config file in `assets/fbank_config.json`.
The `/path/to/embedding_scp_file` and `/path/to/save/embeddings/` means the embedding index file and the path to save embeddings. Please make sure that the `/path/to/save/embeddings/` you have already created.


## Expriments
> Including both Fbank computing and speaker embedding extraction cost time.
> 
> Running On CPU: Intel(R) Xeon(R) Platinum 8163 CPU
> Test audio length is 3000ms.

|Model|Params|RTF|
|:-:|:-:|:-:|
|CAM++|7.18M|0.049|
|ERes2Net-Base|4.6M|0.076|
|ERes2Net-Large|22.6M|0.191|
|ERes2Net-Huge|55.16M|0.420|
|ERes2NetV2|17.8M|0.142|


## Structure
1. `asserts`: The sample resource including config files.
2. `bin`: The final target binary file.
3. `cmake`: The cmake folder for building third-party libraries.
4. `feature`: The feature-related process.
5. `model`: The speaker model folder.
6. `utils`: Some useful tools, like wav-reader.


## Third Party
Please see `cmake/` folder for more details. We use `FetchContent` for downloading these third-party libs which requires 
'cmake' > 3.14.
1. [nlohmann/json](https://json.nlohmann.me/) for loading json files.
2. [onnxruntime](https://github.com/microsoft/onnxruntime) for uploading model and inference.

## TODO
1. [ ] Add better logging system.
2. [ ] Add data format for embedding and feature loading and saving.
3. [ ] Support and check more feature extraction and speaker embedding models.
4. [ ] Test in various envrionments.

