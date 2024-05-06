import argparse, pandas, glob, cv2, numpy, os
import multiprocessing as mp

parser = argparse.ArgumentParser(description = "Extract Video Clips.")
parser.add_argument('--csv_ori', type=str, default='val_orig.csv',help='Original CSV file.')
parser.add_argument('--video_ori_dir', type=str, default='orig_videos/trainval',help='Original video data dir.')
parser.add_argument('--video_out_dir', type=str, default='clips_videos/val',help='Video output dir.')
parser.add_argument('--nj', type=str, default=16, help='Number of workers.')
args = parser.parse_args()

if __name__ == '__main__':
    def extract_one_entity(entity, input_dir, output_dir):
        ins_data = df.get_group(entity)
        video_key = ins_data.iloc[0]['video_id']
        entity_id = ins_data.iloc[0]['entity_id']
        assert entity_id == entity
        video_file = glob.glob(os.path.join(input_dir, '{}.*'.format(video_key)))[0]
        V = cv2.VideoCapture(video_file)
        ins_dir = os.path.join(os.path.join(output_dir, video_key, entity_id))
        os.makedirs(ins_dir, exist_ok=True)
        for _, row in ins_data.iterrows():
            image_filename = os.path.join(ins_dir, str("%.2f"%row['frame_timestamp'])+'.jpg')
            V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
            _, frame = V.read()
            h = numpy.size(frame, 0)
            w = numpy.size(frame, 1)
            x1 = int(row['entity_box_x1'] * w)
            y1 = int(row['entity_box_y1'] * h)
            x2 = int(row['entity_box_x2'] * w)
            y2 = int(row['entity_box_y2'] * h)
            face = frame[y1:y2, x1:x2, :]
            cv2.imwrite(image_filename, face)

    df = pandas.read_csv(args.csv_ori)
    df_neg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
    df_pos = df[df['label_id'] == 1]
    df = pandas.concat([df_pos, df_neg]).reset_index(drop=True)
    df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
    entityList = df['entity_id'].unique().tolist()
    df = df.groupby('entity_id')

    pool = mp.Pool(args.nj)
    for entity in entityList:
        pool.apply_async(extract_one_entity, args=(entity, args.video_ori_dir, args.video_out_dir))

    pool.close()
    pool.join()
