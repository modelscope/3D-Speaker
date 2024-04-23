# SpeakerLab Engines

## Installation
Our project support default environments:

Put a table here.

[//]: # (TODO: Need to final test for installation, including Linux, MacOS and Windows) 

The command for building our project will be like:
```shell
mkdir build # you can change the folder name
cd build
cmake ..
cmake --build .
```

## Usage
1. Step-1: Export model to onnx file 

Please see the scripts `path/to/scripts` <-- TODO: Need to add scripts.

## Structure
1. `asserts`: The sample resource, which will not be released to 3D-Speaker.
2. `bin`: The final target binary file.
3. `cmake`: The cmake folder for building third-party libraries.
4. `feature`: The feature-related process (support FBank so far).
5. `model`: The speaker model folder.
6. `utils`: Some useful tools, like wav-reader.

## Experiments:
The results of our code:

Testing...


## Third Party
Please see `cmake/` folder for more details. We use `FetchContent` for downloading these third-party libs which requires 
'cmake' > 3.14.
1. [nlohmann/json](https://json.nlohmann.me/) for loading json files.
2. [gabime/spdlog](https://github.com/gabime/spdlog) for printing log.
3. [jbeder/yaml-cpp](https://github.com/jbeder/yaml-cpp) for loading yaml files.

## Acknowledge

