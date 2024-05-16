# Semantic Speaker Information Extraction


## Modelscope
You can find our pre-trained model on `Modelscope`:
* [Dialogue Detection](https://modelscope.cn/models/iic/speech_bert_dialogue-detetction_speaker-diarization_chinese/summary)
* [Speaker-Turn Detection](https://modelscope.cn/models/iic/speech_bert_semantic-spk-turn-detection-punc_speaker-diarization_chinese/summary)

You can use `modelscope` to download and use the model.


## Data Preparation

The datasets we use contains [Alimeeting](https://www.openslr.org/119/) and [Aishell-4](https://www.openslr.org/111/).
You can easily download from the [OpenSLR](https://www.openslr.org) website with the following commands:
```shell
# Alimeeting data download
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz # ([73.24G] (AliMeeting Train set, 8-channel microphone array speech) )
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_near.tar.gz # ([22.85G] (AliMeeting Train set, headset microphone speech) )
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz # ([3.42G] (AliMeeting Eval set, 8-channel microphone array speech, headset microphone speech) )
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz # ([8.90G] (AliMeeting Test set, 8-channel microphone array speech, headset microphone speech) )

# Aishell-4 data download
wget https://us.openslr.org/resources/111/train_L.tar.gz # [7.0G] You can change different links from the [OpenSLR](https://www.openslr.org) website.
wget https://us.openslr.org/resources/111/train_M.tar.gz # [25G] You can change different links from the [OpenSLR](https://www.openslr.org) website.
wget https://us.openslr.org/resources/111/train_S.tar.gz # [14G] You can change different links from the [OpenSLR](https://www.openslr.org) website.
wget https://us.openslr.org/resources/111/test.tar.gz # [5.2G] You can change different links from the [OpenSLR](https://www.openslr.org) website.
```
We have also provided the scripts in `egs/semantic_speaker/bert/local/` folder for you to download the data.

You can also organize your own dataset in the following format:
1. For Dialogue Detection Task
    ```json
      [
        {
          "utt_id": "utt_id",
          "conversation_id": "utt_id_1",
          "sentence": "",
          "label": false
        }
      ]
    ```
2. For Speaker-Turn Detection Task
    ```json
      [
        {
          "utt_id": "utt_id",
          "conversation_id": "utt_id_1",
          "sentence": "",
          "change_point_list": [20, 30, 40],
          "spk_num": 2,
          "spk_label_list": [0, 0, 0,...]
        }
      ]
    ```

You can find more details from `egs/semantic_speaker/bert/local/prepare_json_files_for_semantic_speaker.py`.

## Usage

We provide two speaker-related semantic tasks: `Dialogue Detection` and `Speaker Turn Detection`.
The run scripts are `egs/semantic_speaker/bert/run_dialogue_detection.sh` and `egs/semantic_speaker/bert/run_speaker_turn_detection.sh`.
They share the same data preprocessing and the core parts are `egs/semantic_speaker/bert/bin/run_dialogue_detection.py` 
and `egs/semantic_speaker/bert/bin/run_speaker_turn_detection.py`, respectively.

* Dialogue Detection
   To run `Speaker-Turn Detection` task, use the following command:
   ```shell
      bash run_speaker_turn_detection.sh exp/
   ```
   The only parameter is `exp/`, which is the path to the output directory. (We recommend you to use `exp/` as the output directory will be ignored by git)
   
   The shell script is like:
   ```shell
      python bin/run_dialogue_detection.py \
         --model_name_or_path bert-base-chinese \
         --max_seq_length 128 --pad_to_max_length \
         --train_file $json_path/train.dialogue_detection.json \
         --validation_file $json_path/valid.dialogue_detection.json \
         --test_file $json_path/test.dialogue_detection.json \
         --do_train --do_eval --do_predict \
         --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --num_train_epochs 5 \
         --output_dir $output_path --overwrite_output_dir
   ```
   The `--train_file`, `--validation_file`, and `--test_file` are the json files for training, validation, and testing, respectively.
   You can also change them to your own json files. 
   The `model_name_or_path` is the path to the pre-trained BERT model and you can change it to other pre-trained model.
* Speaker Turn Detection
   To run `Speaker-Turn Detection` task, use the following command:
   ```shell
      bash run_speaker_turn_detection.sh exp/
   ```
   The only parameter is `exp/`, which is the path to the output directory. 
   If you have run `Dialogue Detection` before, you can skip the stage 1(downloading datasets).
   
   The shell script is like:
   ```shell
      python bin/run_speaker_turn_detection.py \
          --model_name_or_path bert-base-chinese \
          --max_seq_length 128 --pad_to_max_length \
          --train_file $json_path/train.speaker_turn_detection.json \
          --validation_file $json_path/valid.speaker_turn_detection.json \
          --test_file $json_path/test.speaker_turn_detection.json \
          --do_train --do_eval --do_predict \
          --text_column_name sentence --label_column_name change_point_list --label_num 2 \
          --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --num_train_epochs 5 \
          --output_dir $output_path --overwrite_output_dir
   ```
   The `--train_file`, `--validation_file`, and `--test_file` are the json files for training, validation, and testing, respectively.
   The `text_column_name` and `label_column_name` are the column names of the text and label in the json files. To match the data preprocessing, 
   `sentence` and `change_point_list` should be fixed to read the specific columns.

## Acknowledgement
We have borrowed some codes from the `transformers`.


## Citation
We have published the following paper combining the semantic speaker-related information extraction task with speaker diarization tasks:
```latex
@inproceedings{Luyao2023ACL,
	author       = {Luyao Cheng and Siqi Zheng and Qinglin Zhang and Hui Wang and Yafeng Chen and Qian Chen},
	title        = {Exploring Speaker-Related Information in Spoken Language Understanding for Better Speaker Diarization},
	booktitle    = {Findings of the {ACL} 2023, Toronto, Canada, July 9-14, 2023},
	pages        = {14068--14077},
	year         = {2023},
}
@article{Cheng2023ImprovingSD,
  title={Improving Speaker Diarization using Semantic Information: Joint Pairwise Constraints Propagation},
  author={Luyao Cheng and Siqi Zheng and Qinglin Zhang and Haibo Wang and Yafeng Chen and Qian Chen and Shiliang Zhang},
  journal={ArXiv},
  year={2023},
  volume={abs/2309.10456},
}
```

The Alimeeting and Aishell-4 dataset we used are from the following papers:
```latex
@inproceedings{AISHELL-4_2021,
    title={AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario},
    author={Yihui Fu, Luyao Cheng, Shubo Lv, Yukai Jv, Yuxiang Kong, Zhuo Chen, Yanxin Hu, Lei Xie, Jian Wu, Hui Bu, Xin Xu, Jun Du, Jingdong Chen},
    booktitle={Interspeech},
    url={https://arxiv.org/abs/2104.03603},
    year={2021}
}

@inproceedings{Yu2022M2MeT,
    title={M2{M}e{T}: The {ICASSP} 2022 Multi-Channel Multi-Party Meeting Transcription Challenge},
    author={Yu, Fan and Zhang, Shiliang and Fu, Yihui and Xie, Lei and Zheng, Siqi and Du, Zhihao and Huang, Weilong and Guo, Pengcheng and Yan, Zhijie and Ma, Bin and Xu, Xin and Bu, Hui},
    booktitle={Proc. ICASSP},
    year={2022},
    organization={IEEE}
}
```
