import argparse
import tqdm

from speakerlab.utils.fileio import write_json_file, load_json_file
from speakerlab.utils.utils import get_logger

logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge multi json files"
    )
    parser.add_argument(
        "--initial_files", required=True, nargs="+", help="The initial json files"
    )
    parser.add_argument(
        "--result_json_file", required=True, help="The result json file"
    )
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(f"{args}")

    initial_files = args.initial_files
    result_json_file = args.result_json_file

    result_list = []
    conversation_id_set = set()
    for initial_file in tqdm.tqdm(initial_files):
        cur_list = load_json_file(initial_file)
        logger.info(f"initial_file = {initial_file}: {len(cur_list)} items")
        for item in cur_list:
            conversation_id = item['conversation_id']
            if conversation_id in conversation_id_set:
                raise ValueError(f"Conversation_id = {conversation_id}")
            conversation_id_set.add(conversation_id)
        result_list.extend(cur_list)

    logger.info(f"result_list: {len(result_list)}, set: {len(conversation_id_set)}")

    write_json_file(result_json_file, result_list)


if __name__ == '__main__':
    main()
