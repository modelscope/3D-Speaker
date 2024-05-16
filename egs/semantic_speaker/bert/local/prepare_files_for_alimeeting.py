import argparse
import os
import textgrid
import tqdm

from speakerlab.utils.fileio import write_wav_scp, write_trans7time_list
from speakerlab.utils.utils import get_logger

logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare files for alimeeting dataset"
    )
    parser.add_argument(
        "--home_path", required=True, help="The root path for alimeeting path"
    )
    parser.add_argument(
        "--save_path", required=True, help="The file path to save files"
    )
    return parser.parse_args()


def solve_textgrid(textgrid_file, utt_id):
    rttm_list = []
    trans7time_list = []
    tg = textgrid.TextGrid.fromFile(textgrid_file)

    for i in range(len(tg)):
        for j in range(len(tg[i])):
            cur_seg = tg[i][j]
            if cur_seg.mark:
                trans7time_list.append((
                    tg[i].name, cur_seg.minTime, cur_seg.maxTime, cur_seg.mark.strip()
                ))
                rttm_list.append((
                    utt_id, tg[i].name, cur_seg.minTime, cur_seg.maxTime
                ))
    trans7time_list = sorted(trans7time_list, key=lambda x: x[1])
    rttm_list = sorted(rttm_list, key=lambda x: x[2])

    return trans7time_list, rttm_list


def build_textgrid_wav_files(initial_path,
                             trans7time_scp_file, trans7time_save_path):
    logger.info(
        f"From initial_path = {initial_path}:\n"
        f"trans7time_scp_file: {trans7time_scp_file}"
    )

    textgrid_constant = "textgrid_dir"

    os.makedirs(trans7time_save_path, exist_ok=True)

    initial_textgrid_path = os.path.join(initial_path, textgrid_constant)

    textgrid_items = os.listdir(initial_textgrid_path)

    trans7time_scp = dict()
    for textgrid_item in tqdm.tqdm(textgrid_items):
        textgrid_file = os.path.join(initial_textgrid_path, textgrid_item)
        utt_id = textgrid_item[:11]
        trans7time_list, rttm_list = solve_textgrid(textgrid_file, utt_id)
        trans7time_file = os.path.join(trans7time_save_path, f"{utt_id}.trans7time")
        trans7time_scp[utt_id] = trans7time_file
        write_trans7time_list(trans7time_file, trans7time_list)

    write_wav_scp(trans7time_scp_file, trans7time_scp)


def main():
    args = get_args()

    home_path = args.home_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_home_path = os.path.join(home_path, "Train_Ali")
    if not os.path.exists(train_home_path):
        train_home_path = home_path
    train_near_path = os.path.join(train_home_path, "Train_Ali_near")
    train_far_path = os.path.join(train_home_path, "Train_Ali_far")
    train_near_trans7time_scp_file = os.path.join(save_path, "train_near_trans7time.scp")
    train_near_trans7time_save_path = os.path.join(save_path, "train_near_trans7time")
    train_far_trans7time_scp_file = os.path.join(save_path, "train_far_trans7time.scp")
    train_far_trans7time_save_path = os.path.join(save_path, "train_far_trans7time")

    eval_home_path = os.path.join(home_path, "Eval_Ali")
    eval_near_path = os.path.join(eval_home_path, "Eval_Ali_near")
    eval_far_path = os.path.join(eval_home_path, "Eval_Ali_far")
    eval_near_trans7time_scp_file = os.path.join(save_path, "eval_near_trans7time.scp")
    eval_near_trans7time_save_path = os.path.join(save_path, "eval_near_trans7time")
    eval_far_trans7time_scp_file = os.path.join(save_path, "eval_far_trans7time.scp")
    eval_far_trans7time_save_path = os.path.join(save_path, "eval_far_trans7time")

    test_home_path = os.path.join(home_path, "Test_Ali")
    test_near_path = os.path.join(test_home_path, "Test_Ali_near")
    test_far_path = os.path.join(test_home_path, "Test_Ali_far")
    test_near_trans7time_scp_file = os.path.join(save_path, "test_near_trans7time.scp")
    test_near_trans7time_save_path = os.path.join(save_path, "test_near_trans7time")
    test_far_trans7time_scp_file = os.path.join(save_path, "test_far_trans7time.scp")
    test_far_trans7time_save_path = os.path.join(save_path, "test_far_trans7time")

    build_textgrid_wav_files(
        train_near_path, train_near_trans7time_scp_file, train_near_trans7time_save_path
    )
    build_textgrid_wav_files(
        train_far_path, train_far_trans7time_scp_file, train_far_trans7time_save_path
    )
    build_textgrid_wav_files(
        eval_near_path, eval_near_trans7time_scp_file, eval_near_trans7time_save_path
    )
    build_textgrid_wav_files(
        eval_far_path, eval_far_trans7time_scp_file, eval_far_trans7time_save_path
    )
    build_textgrid_wav_files(
        test_near_path, test_near_trans7time_scp_file, test_near_trans7time_save_path
    )
    build_textgrid_wav_files(
        test_far_path, test_far_trans7time_scp_file, test_far_trans7time_save_path
    )


if __name__ == '__main__':
    main()
