import argparse, pandas, os
from scipy.io import wavfile
import multiprocessing as mp

parser = argparse.ArgumentParser(description = "Extract Audio Clips.")
parser.add_argument('--csv_ori', type=str, default='val_orig.csv',help='Original CSV file.')
parser.add_argument('--audio_ori_dir', type=str, default='orig_videos/trainval',help='Original audio data dir.')
parser.add_argument('--audio_out_dir', type=str, default='clips_audios/val',help='Audio output dir.')
parser.add_argument('--nj', type=str, default=32, help='Number of workers.')
args = parser.parse_args()

if __name__ == '__main__':
    def extract_one_entity(entity, input_dir, output_dir):
        ins_data = df.get_group(entity)
        video_key = ins_data.iloc[0]['video_id']
        entity_id = ins_data.iloc[0]['entity_id']
        start = ins_data.iloc[0]['frame_timestamp']
        end = ins_data.iloc[-1]['frame_timestamp']
        assert entity_id == entity
        ins_dir = os.path.join(output_dir, video_key)
        os.makedirs(ins_dir, exist_ok=True)
        ins_path = os.path.join(ins_dir, entity_id+'.wav')
        audio_file = os.path.join(input_dir, video_key+'.wav')
        sr, audio = wavfile.read(audio_file)
        audio_start = int(float(start)*sr)
        audio_end = int(float(end)*sr)
        audio_data = audio[audio_start:audio_end]
        wavfile.write(ins_path, sr, audio_data)

    df = pandas.read_csv(args.csv_ori)
    def_neg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
    df_pos = df[df['label_id'] == 1]
    df = pandas.concat([df_pos, def_neg]).reset_index(drop=True)
    df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
    entity_list = df['entity_id'].unique().tolist()
    df = df.groupby('entity_id')

    pool = mp.Pool(args.nj)
    for entity in entity_list:
        pool.apply_async(extract_one_entity, args=(entity, args.audio_ori_dir, args.audio_out_dir))

    pool.close()
    pool.join()
