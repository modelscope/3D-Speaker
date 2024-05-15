# Semantic Speaker Information Extraction


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

## Usage and Documents



## Citation
We have published the following paper combining the semantic speaker-related information extraction task with speaker diarization tasks:
```latex


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



