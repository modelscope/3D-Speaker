from typing import List
import os
import argparse
import tqdm

from speakerlab.utils.fileio import load_wav_scp, write_json_file, load_trans7time_list
from speakerlab.utils.utils import get_logger

logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser(description="Prepare files for semantic speaker tasks")
    parser.add_argument(
        "--flag", default=None, help="The flag to diff"
    )
    parser.add_argument(
        "--trans7time_scp_file", required=True, help="The input trans7time scp file"
    )
    parser.add_argument(
        "--save_path", required=True, help="The path to save online json"
    )
    parser.add_argument(
        "--sentence_length", type=int, default=96, help="The sentence length for sliding windows"
    )
    parser.add_argument(
        "--sentence_shift", type=int, default=32, help="The sentence shift for sliding windows"
    )

    return parser.parse_args()


def split_trans7time(trans7time_list):
    spk_sentence_list = []
    split_char_list = ["。", "？", "！"]
    for (spk_id, st, ed, content) in trans7time_list:
        temp_content = ""
        for ch in content:
            if ch in split_char_list:
                temp_content += ch
                spk_sentence_list.append((spk_id, temp_content, len(temp_content)))
                temp_content = ""
            else:
                temp_content += ch

        if len(temp_content) > 0:
            spk_sentence_list.append((spk_id, temp_content, len(temp_content)))
    return spk_sentence_list


def build_dialogue_detection_from_trans7time_shift_windows(utt_id: str,
                                                           trans7time_list: List,
                                                           sentence_length: int,
                                                           sentence_shift: int) -> List:
    spk_sentence_list = split_trans7time(trans7time_list)
    spk_sentence_num = len(spk_sentence_list)

    index = 0
    i = 0
    result_conversation_list = []
    while i < spk_sentence_num:
        cur_sentence = ""
        j = i
        count_sentence_length = 0
        spk_map = dict()
        spk_num = 0
        change_point_list = []
        role_tag_list = []
        while j < spk_sentence_num:
            cur_sentence_length = spk_sentence_list[j][2]
            cur_sentence_spk_id = spk_sentence_list[j][0]
            cur_sentence += spk_sentence_list[j][1]

            if cur_sentence_spk_id not in spk_map:
                spk_map[cur_sentence_spk_id] = spk_num
                spk_num += 1
            cur_sentence_spk_index = spk_map[cur_sentence_spk_id]
            if len(role_tag_list) != 0 and role_tag_list[-1] != cur_sentence_spk_index:
                change_point_list.append(count_sentence_length)
            role_tag_list.extend([cur_sentence_spk_index]
                                 * cur_sentence_length)

            count_sentence_length += cur_sentence_length
            if count_sentence_length >= sentence_length:
                break
            j += 1

        result_conversation_list.append({
            "utt_id": utt_id,
            "conversation_id": f"{utt_id}_{index + 1}",
            "sentence": cur_sentence,
            "label": bool(len(spk_map) > 1),
        })
        index += 1

        count_sentence_length = 0
        j = i + 1
        while j < spk_sentence_num:
            count_sentence_length += spk_sentence_list[j][2]
            if count_sentence_length >= sentence_shift:
                break
            j += 1
        i = j

    result_conversation_list = result_conversation_list[:-1]

    sentence = ""
    count_sentence_length = 0
    role_tag_list = []
    change_point_list = []
    spk_map = dict()
    spk_num = 0
    i = spk_sentence_num - 1
    while i >= 0:
        cur_spk_id = spk_sentence_list[i][0]
        cur_sentence = spk_sentence_list[i][1]
        cur_sentence_length = spk_sentence_list[i][2]

        if cur_spk_id not in spk_map:
            spk_map[cur_spk_id] = spk_num
            spk_num += 1
        cur_spk_index = spk_map[cur_spk_id]
        if len(role_tag_list) != 0 and role_tag_list[-1] != cur_spk_index:
            change_point_list.append(count_sentence_length)
        role_tag_list.extend([cur_spk_index] * cur_sentence_length)
        sentence = cur_sentence + sentence
        count_sentence_length += cur_sentence_length

        if count_sentence_length >= sentence_length:
            break
        i -= 1

    reversed(role_tag_list)

    result_conversation_list.append({
        "utt_id": utt_id,
        "conversation_id": f"{utt_id}_{index + 1}",
        "sentence": sentence,
        "label": bool(len(spk_map) > 1),
    })

    return result_conversation_list


def build_speaker_turn_detection_from_trans7time_shift_windows(utt_id: str,
                                                               trans7time_list: List,
                                                               sentence_length: int,
                                                               sentence_shift: int) -> List:
    spk_sentence_list = split_trans7time(trans7time_list)
    spk_sentence_num = len(spk_sentence_list)

    index = 0
    i = 0
    result_conversation_list = []
    while i < spk_sentence_num:
        cur_sentence = ""
        j = i
        count_sentence_length = 0
        spk_map = dict()
        spk_num = 0
        change_point_list = []
        spk_label_list = []
        while j < spk_sentence_num:
            cur_sentence_length = spk_sentence_list[j][2]
            cur_sentence_spk_id = spk_sentence_list[j][0]
            cur_sentence += spk_sentence_list[j][1]

            if cur_sentence_spk_id not in spk_map:
                spk_map[cur_sentence_spk_id] = spk_num
                spk_num += 1
            cur_sentence_spk_index = spk_map[cur_sentence_spk_id]
            if len(spk_label_list) != 0 and spk_label_list[-1] != cur_sentence_spk_index:
                change_point_list.append(count_sentence_length)
            spk_label_list.extend([cur_sentence_spk_index] * cur_sentence_length)

            count_sentence_length += cur_sentence_length
            if count_sentence_length >= sentence_length:
                break
            j += 1

        result_conversation_list.append({
            "utt_id": utt_id,
            "conversation_id": f"{utt_id}_{index + 1}",
            "sentence": cur_sentence,
            "change_point_list": change_point_list,
            "spk_num": spk_num,
            "spk_label_list": spk_label_list,
        })
        index += 1

        count_sentence_length = 0
        j = i + 1
        while j < spk_sentence_num:
            count_sentence_length += spk_sentence_list[j][2]
            if count_sentence_length >= sentence_shift:
                break
            j += 1
        i = j

    result_conversation_list = result_conversation_list[:-1]

    sentence = ""
    count_sentence_length = 0
    spk_label_list = []
    change_point_list = []
    spk_map = dict()
    spk_num = 0
    i = spk_sentence_num - 1
    while i >= 0:
        cur_spk_id = spk_sentence_list[i][0]
        cur_sentence = spk_sentence_list[i][1]
        cur_sentence_length = spk_sentence_list[i][2]

        if cur_spk_id not in spk_map:
            spk_map[cur_spk_id] = spk_num
            spk_num += 1
        cur_spk_index = spk_map[cur_spk_id]
        if len(spk_label_list) != 0 and spk_label_list[-1] != cur_spk_index:
            change_point_list.append(count_sentence_length)
        spk_label_list.extend([cur_spk_index] * cur_sentence_length)
        sentence = cur_sentence + sentence
        count_sentence_length += cur_sentence_length

        if count_sentence_length >= sentence_length:
            break
        i -= 1

    reversed(spk_label_list)

    result_conversation_list.append({
        "utt_id": utt_id,
        "conversation_id": f"{utt_id}_{index + 1}",
        "sentence": sentence,
        "change_point_list": sorted([len(sentence) - i for i in change_point_list]),
        "spk_num": spk_num,
        "spk_label_list": spk_label_list
    })

    return result_conversation_list


def main():
    args = get_args()
    logger.info(f"{args}")

    flag = args.flag
    trans7time_scp_file = args.trans7time_scp_file
    save_path = args.save_path
    sentence_length = args.sentence_length
    sentence_shift = args.sentence_shift

    trans7time_scp = load_wav_scp(trans7time_scp_file)
    total_dialogue_detection_results = []
    total_speaker_turn_detection_results = []
    for utt_id in tqdm.tqdm(trans7time_scp):
        trans7time_file = trans7time_scp[utt_id]
        trans7time_list = load_trans7time_list(trans7time_file)

        dialogue_detection_results = build_dialogue_detection_from_trans7time_shift_windows(
            utt_id, trans7time_list, sentence_length, sentence_shift
        )
        speaker_turn_detection_results = build_speaker_turn_detection_from_trans7time_shift_windows(
            utt_id, trans7time_list, sentence_length, sentence_shift
        )
        total_dialogue_detection_results.extend(dialogue_detection_results)
        total_speaker_turn_detection_results.extend(speaker_turn_detection_results)

    dialogue_detection_json_file = os.path.join(save_path, f"{flag}.dialogue_detection.json")
    speaker_turn_detection_json_file = os.path.join(save_path, f"{flag}.speaker_turn_detection.json")
    write_json_file(dialogue_detection_json_file, total_dialogue_detection_results)
    write_json_file(speaker_turn_detection_json_file, total_speaker_turn_detection_results)


if __name__ == '__main__':
    main()
