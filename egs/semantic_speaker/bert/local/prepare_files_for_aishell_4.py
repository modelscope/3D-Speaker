import argparse
import os
import textgrid
import tqdm

from speakerlab.utils.fileio import write_wav_scp, write_trans7time_list
from speakerlab.utils.utils import get_logger

logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser(
        description=f"Prepare files for aishell-4"
    )
    parser.add_argument(
        "--home_path", required=True,
        help="The home path which including ['train_L', 'train_M', 'train_S', 'test']"
    )
    parser.add_argument(
        "--save_path", required=True, help="The path to save files"
    )
    return parser.parse_args()


# Some Aishell-4 data have some meaningless sentence, which is not suitable.
train_offset_dict = {
    "20200616_M_R001S01C01": 5,
    "20200616_M_R001S02C01": 9.5,
    "20200616_M_R001S03C01": 6.5,
    "20200616_M_R001S04C01": 22.5,
    "20200616_M_R001S05C01": 24,
    "20200616_M_R001S06C01": 28.5,
    "20200616_M_R001S07C01": 12.5,
    "20200616_M_R001S08C01": 16.4,
    "20200617_M_R001S01C01": 38.7,
    "20200617_M_R001S02C01": 14,
    "20200617_M_R001S03C01": 14.5,
    "20200617_M_R001S04C01": 33.5,
    "20200617_M_R001S05C01": 31.5,
    "20200617_M_R001S06C01": 42.7,
    "20200617_M_R001S07C01": 34,
    "20200617_M_R001S08C01": 32.5,
    "20200618_M_R001S01C01": 27,
    "20200618_M_R001S02C01": 23.7,
    "20200618_M_R001S03C01": 24.4,
    "20200618_M_R001S04C01": 24.5,
    "20200618_M_R001S05C01": 38,
    "20200618_M_R001S06C01": 31,
    "20200618_M_R001S07C01": 41.2,
    "20200618_M_R001S08C01": 50.1,
    "20200620_M_R002S04C01": 43.8,
    "20200620_M_R002S05C01": 20.0,
    "20200620_M_R002S06C01": 53,
    "20200620_M_R002S07C01": 27.0,
    "20200620_M_R002S08C01": 26.9,
    "20200621_M_R002S01C01": 26.4,
    "20200621_M_R002S02C01": 20.5,
    "20200621_M_R002S03C01": 25.2,
    "20200621_M_R002S04C01": 29.4,
    "20200621_M_R002S05C01": 23.5,
    "20200621_M_R002S06C01": 16.2,
    "20200621_M_R002S07C01": 30,
    "20200621_M_R002S08C01": 65,
    "20200622_M_R002S01C01": 29.5,
    "20200622_M_R002S02C01": 34.05,
    "20200622_M_R002S03C01": 33,
    "20200622_M_R002S04C01": 17.5,
    "20200622_M_R002S05C01": 23,
    "20200622_M_R002S06C01": 21.5,
    "20200622_M_R002S07C01": 21.4,
    "20200622_M_R002S08C01": 17.2,
    "20200623_S_R001S01C01": 31.1,
    "20200623_S_R001S02C01": 35.2,
    "20200623_S_R001S03C01": 33.6,
    "20200623_S_R001S04C01": 19.45,
    "20200623_S_R001S05C01": 30,
    "20200623_S_R001S06C01": 19.0,
    "20200623_S_R001S07C01": 25.5,
    "20200623_S_R001S08C01": 16.35,
    "20200624_S_R001S01C01": 34.0,
    "20200624_S_R001S02C01": 20.0,
    "20200624_S_R001S03C01": 29.75,
    "20200624_S_R001S04C01": 30,
    "20200624_S_R001S05C01": 33,
    "20200624_S_R001S06C01": 26,
    "20200624_S_R001S07C01": 34.17,
    "20200624_S_R001S08C01": 27.26,
    "20200630_S_R001S01C01": 28.4,
    "20200630_S_R001S02C01": 36,
    "20200630_S_R001S03C01": 33,
    "20200630_S_R001S04C01": 33.5,
    "20200630_S_R001S05C01": 22.8,
    "20200630_S_R001S06C01": 43.8,
    "20200630_S_R001S07C01": 32.5,
    "20200630_S_R001S08C01": 40.25,
    "20200701_S_R001S01C01": 24.29,
    "20200701_S_R001S02C01": 27,
    "20200701_S_R001S03C01": 30.0,
    "20200701_S_R001S04C01": 26.8,
    "20200701_S_R001S05C01": 21.2,
    "20200701_S_R001S06C01": 32.2,
    "20200701_S_R001S07C01": 20.6,
    "20200701_S_R001S08C01": 24,
    "20200702_S_R001S01C01": 36,
    "20200702_S_R001S02C01": 42.5,
    "20200702_S_R001S03C01": 25.75,
    "20200702_S_R001S04C01": 44,
    "20200702_S_R001S05C01": 43.5,
    "20200702_S_R001S07C01": 22,
    "20200702_S_R001S08C01": 38.93,
    "20200703_M_R002S01C01": 33,
    "20200703_M_R002S02C01": 42.6,
    "20200703_M_R002S03C01": 32.75,
    "20200703_M_R002S04C01": 47,
    "20200703_M_R002S05C01": 31,
    "20200703_M_R002S06C01": 39.3,
    "20200703_M_R002S07C01": 30,
    "20200703_M_R002S08C01": 37,
    "20200704_M_R002S01C01": 42,
    "20200704_M_R002S02C01": 26,
    "20200704_M_R002S03C01": 25,
    "20200704_M_R002S04C01": 29,
    "20200704_M_R002S05C01": 26.3,
    "20200704_M_R002S06C01": 26.3,
    "20200704_M_R002S07C01": 30.0,
    "20200704_M_R002S08C01": 23,
    "20200705_M_R002S01C01": 46,
    "20200705_M_R002S02C01": 30,
    "20200705_M_R002S03C01": 34.5,
    "20200705_M_R002S04C01": 45,
    "20200705_M_R002S05C01": 39,
    "20200705_M_R002S06C01": 29,
    "20200705_M_R002S07C01": 23,
    "20200705_M_R002S08C01": 36.5,
    "20200706_L_R001S01C01": 48,
    "20200706_L_R001S02C01": 72,
    "20200706_L_R001S03C01": 68.5,
    "20200706_L_R001S04C01": 55.5,
    "20200706_L_R001S05C01": 50.25,
    "20200706_L_R001S06C01": 28.6,
    "20200706_L_R001S07C01": 82,
    "20200706_L_R001S08C01": 50,
    "20200707_L_R001S01C01": 86,
    "20200707_L_R001S02C01": 49,
    "20200707_L_R001S03C01": 44,
    "20200707_L_R001S04C01": 52.3,
    "20200707_L_R001S05C01": 58,
    "20200707_L_R001S06C01": 69,
    "20200707_L_R001S07C01": 56.5,
    "20200707_L_R001S08C01": 95.2870,
    "20200708_L_R002S01C01": 75.5,
    "20200708_L_R002S02C01": 46,
    "20200708_L_R002S03C01": 55,
    "20200708_L_R002S04C01": 45,
    "20200708_L_R002S05C01": 50,
    "20200708_L_R002S07C01": 31,
    "20200708_L_R002S08C01": 60.5,
    "20200709_L_R002S01C01": 50.15,
    "20200709_L_R002S03C01": 46.3,
    "20200709_L_R002S04C01": 45.4,
    "20200709_L_R002S05C01": 45,
    "20200709_L_R002S06C01": 43,
    "20200709_L_R002S07C01": 39.8,
    "20200709_L_R002S08C01": 48,
    "20200710_M_R002S01C01": 32,
    "20200710_M_R002S03C01": 40,
    "20200710_M_R002S04C01": 53.4,
    "20200710_M_R002S05C01": 35.0,
    "20200710_M_R002S06C01": 41,
    "20200710_M_R002S07C01": 75.8,
    "20200710_M_R002S08C01": 27.5,
    "20200712_M_R002S01C01": 26.7,
    "20200712_M_R002S02C01": 27,
    "20200712_M_R002S03C01": 28.6,
    "20200712_M_R002S04C01": 46,
    "20200712_M_R002S05C01": 49.34,
    "20200712_M_R002S06C01": 43.2,
    "20200712_M_R002S07C01": 39,
    "20200712_M_R002S08C01": 35.55,
    "20200713_M_R002S01C01": 56.5,
    "20200713_M_R002S02C01": 55,
    "20200713_M_R002S03C01": 49.4,
    "20200713_M_R002S04C01": 50.15,
    "20200713_M_R002S05C01": 42,
    "20200713_M_R002S06C01": 49,
    "20200713_M_R002S07C01": 43.25,
    "20200713_M_R002S08C01": 41,
    "20200714_M_R002S01C01": 37.7,
    "20200714_M_R002S02C01": 58.5,
    "20200714_M_R002S03C01": 45,
    "20200714_M_R002S04C01": 42.5,
    "20200714_M_R002S05C01": 56,
    "20200714_M_R002S06C01": 56.5,
    "20200714_M_R002S07C01": 52.8,
    "20200714_M_R002S08C01": 54,
    "20200715_M_R002S01C01": 32.5,
    "20200715_M_R002S02C01": 26,
    "20200715_M_R002S03C01": 35,
    "20200715_M_R002S04C01": 42,
    "20200715_M_R002S05C01": 33,
    "20200715_M_R002S06C01": 38,
    "20200715_M_R002S07C01": 29.63,
    "20200715_M_R002S08C01": 17.45,
    "20200805_S_R001S01C01": 24.5,
    "20200805_S_R001S03C01": 25,
    "20200805_S_R001S06C01": 27,
    "20200805_S_R001S08C01": 25,
    "20200806_S_R001S01C01": 21,
    "20200806_S_R001S05C01": 19.5,
    "20200806_S_R001S07C01": 20,
    "20200807_S_R001S02C01": 23.5,
    "20200807_S_R001S03C01": 21.5,
    "20200807_S_R001S04C01": 33.3,
    "20200807_S_R001S06C01": 19,
    "20200807_S_R001S07C01": 24,
    "20200807_S_R001S08C01": 26,
    "20200808_S_R001S02C01": 22,
}

test_offset_dict = {
    "L_R003S01C02": 23,
    "L_R003S02C02": 20.6,
    "L_R003S03C02": 23,
    "L_R003S04C02": 23,
    "L_R004S01C01": 12.5,
    "L_R004S02C01": 9.2,
    "L_R004S03C01": 12.5,
    "L_R004S06C01": 10.5,
    "M_R003S01C01": 14.48,
    "M_R003S02C01": 23,
    "M_R003S04C01": 15,
    "M_R003S05C01": 15,
    "S_R003S01C01": 19,
    "S_R003S02C01": 11.5,
    "S_R003S03C01": 18,
    "S_R003S04C01": 21.5,
    "S_R004S01C01": 17,
    "S_R004S02C01": 22.8,
    "S_R004S03C01": 20.5,
    "S_R004S04C01": 17,
}


def filter_sentence(sentence: str) -> str:
    bad_list = ["<sil>", "<$>", "<>", "<%>", "<#>", "&", "<->", "<_>"]
    for bad_str in bad_list:
        sentence = sentence.replace(bad_str, "")
    assert "<" not in sentence and ">" not in sentence, f"[{sentence}] have strange signals"
    return sentence


def filter_trans7time_list(utt_id, trans7time_list):
    result = list()
    if utt_id in train_offset_dict:
        offset = train_offset_dict[utt_id]
    else:
        offset = test_offset_dict[utt_id]
    for (spk_id, st, ed, content) in trans7time_list:
        if ed < offset:
            continue
        content = filter_sentence(content)
        if len(content) <= 0:
            continue
        result.append((
            spk_id, st, ed, content
        ))
    return result


def solve_trans7time_list_punctuation(trans7time_list):
    punc_list = ['，', '。', '？', '、', '！']

    trans7time_list = sorted(trans7time_list, key=lambda x: x[1])
    result_trans7time_list = []
    pre_spk_id = None
    pre_content = ""
    pre_st, pre_ed = 0.0, 0.0
    for idx, (spk_id, st, ed, content) in enumerate(trans7time_list):
        if pre_spk_id is None:
            pass
        elif pre_spk_id != spk_id:
            pre_final_ch = pre_content[-1]
            pre_final_punc = "" if pre_final_ch in punc_list else "。"
            result_trans7time_list.append((
                pre_spk_id, pre_st, pre_ed, pre_content + pre_final_punc
            ))
        else:
            pre_final_ch = pre_content[-1]
            pre_final_punc = "" if pre_final_ch in punc_list else "，"
            result_trans7time_list.append((
                pre_spk_id, pre_st, pre_ed, pre_content + pre_final_punc
            ))

        pre_spk_id = spk_id
        pre_content = content
        pre_st, pre_ed = st, ed

    pre_final_ch = pre_content[-1]
    pre_final_punc = "" if pre_final_ch in punc_list else "。"
    result_trans7time_list.append((
        pre_spk_id, pre_st, pre_ed, pre_content + pre_final_punc
    ))

    return result_trans7time_list


def correct_aishell_4_from_some_source(utt_id, textgrid_file):
    """
        It seems aishell_4 from different source have some mistakes so that the TextGrid cannot read it
    """
    if utt_id == "20200622_M_R002S07C01":
        logger.info(f"correct {utt_id}")
        result_lines = []
        with open(textgrid_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                if line.find("2104.492") >= 0:
                    line = line.replace("2104.492", "2187.436")
                result_lines.append(line)
        with open(textgrid_file, "w") as fw:
            for line in result_lines:
                fw.write(f"{line}")
    elif utt_id == "20200710_M_R002S06C01":
        logger.info(f"correct {utt_id}")
        result_lines = []
        with open(textgrid_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                if line.find("1836.689") >= 0:
                    line = line.replace("1836.689", "1846.773")
                result_lines.append(line)
        with open(textgrid_file, "w") as fw:
            for line in result_lines:
                fw.write(f"{line}")


def convert_textgrid_to_trans7time(textgrid_scp, save_path):
    trans7time_scp = dict()
    for utt_id in tqdm.tqdm(textgrid_scp):
        textgrid_file = textgrid_scp[utt_id]
        correct_aishell_4_from_some_source(utt_id, textgrid_file)
        tg = textgrid.TextGrid.fromFile(textgrid_file)
        trans7time_list = []
        for i in range((len(tg))):
            for j in range(len(tg[i])):
                cur_seg = tg[i][j]
                if cur_seg.mark:
                    trans7time_list.append((
                        tg[i].name, cur_seg.minTime, cur_seg.maxTime, cur_seg.mark.strip()
                    ))
        trans7time_list = sorted(trans7time_list, key=lambda x: x[1])
        trans7time_list = filter_trans7time_list(utt_id, trans7time_list)
        trans7time_list = solve_trans7time_list_punctuation(trans7time_list)
        trans7time_file = os.path.join(save_path, f"{utt_id}.trans7time")
        write_trans7time_list(trans7time_file, trans7time_list)
        trans7time_scp[utt_id] = trans7time_file
    return trans7time_scp


def main():
    args = get_args()
    logger.info(f"{args}")
    home_path = args.home_path
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_folders = ['train_L', 'train_M', 'train_S', 'test']

    textgrid_scp = dict()
    for base_folder in tqdm.tqdm(base_folders):
        logger.info(f"Start {base_folder}")
        cur_textgrid_scp = dict()

        cur_folder = os.path.join(home_path, base_folder)
        if not os.path.exists(cur_folder):
            logger.warning(f"{cur_folder} do not exist")
            continue

        # collect textgrid folder
        textgrid_folder = os.path.join(cur_folder, "TextGrid")
        textgrid_items = os.listdir(textgrid_folder)
        textgrid_scp_file = os.path.join(save_path, f"{base_folder}_textgrid.scp")
        trans7time_scp_file = os.path.join(save_path, f"{base_folder}_trans7time.scp")
        trans7time_save_path = os.path.join(save_path, f"{base_folder}_trans7time")
        os.makedirs(trans7time_save_path, exist_ok=True)
        for textgrid_item in textgrid_items:
            utt_id = textgrid_item.split(".")[0]
            extension = textgrid_item.split(".")[1]
            textgrid_file_path = os.path.join(textgrid_folder, textgrid_item)
            if extension == "TextGrid":
                cur_textgrid_scp[utt_id] = textgrid_file_path
                textgrid_scp[utt_id] = textgrid_file_path
        logger.info(f"Build {base_folder} TextGrid.scp finished")

        trans7time_scp = convert_textgrid_to_trans7time(cur_textgrid_scp, trans7time_save_path)

        write_wav_scp(textgrid_scp_file, cur_textgrid_scp)
        write_wav_scp(trans7time_scp_file, trans7time_scp)


if __name__ == '__main__':
    main()
