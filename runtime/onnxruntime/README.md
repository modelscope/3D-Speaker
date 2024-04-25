# SpeakerLab Engines
> We provide a simple speaker ONNX Runtime along with related scripts to facilitate the construction of an engine for extracting speaker embeddings. This is a sample project and, for ease of use, we have strived to keep the code as simple and clear as possible while minimizing the reliance on third-party libraries, allowing you to effortlessly port them into your project code.


## Installation and Usage
1. Step-1: Export Model to ONNX 

Please see the scripts `speakerlab/bin/export_speaker_embedding_onnx.py`. Here is a simple example:
```shell
python speakerlab/bin/export_speaker_embedding_onnx.py \
    --experiment_path your/experiment_path/ \
    --model_id iic/speech_eres2net_sv_en_voxceleb_16k \
    --target_onnx_file path/to/save/eres2net.onnx
```
Now you have exported the speaker embedding model to the onnx file.


2. Step-2: Compile ONNX Runtime Project
```shell
cd runtime/onnxruntime/
mkdir build/ # you can change the folder name
cd build/
cmake ..
make
```
If everything is ok, you can find the binary file `build/bin/`: `extract_speaker_embedding`, `make_fbank_feature` and `read_and_describe_wav`

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
The `/path/to/embedding_scp_file` and `/path/to/save/embeddings/` means the embedding index file and the path to save embeddings.


## Structure
1. `asserts`: The sample resource, which will not be released to 3D-Speaker.
2. `bin`: The final target binary file.
3. `cmake`: The cmake folder for building third-party libraries.
4. `feature`: The feature-related process (support FBank so far).
5. `model`: The speaker model folder.
6. `utils`: Some useful tools, like wav-reader.


## Third Party
Please see `cmake/` folder for more details. We use `FetchContent` for downloading these third-party libs which requires 
'cmake' > 3.14.
1. [nlohmann/json](https://json.nlohmann.me/) for loading json files.
2. [jbeder/yaml-cpp](https://github.com/jbeder/yaml-cpp) for loading yaml files.

## TODO
1. [ ] Add better logging system.
2. [ ] Add data format for embedding and feature.
3. [ ] Support more feature extraction and speaker embedding models.
