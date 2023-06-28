# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import csv
import time
import logging
import pickle
import argparse
import torchaudio
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s]: %(message)s")
logger = logging.getLogger(__file__)

def get_utt2spk_dict(utt2spk, suffix=''):
    temp_dict={}
    with open(utt2spk,'r') as utt2spk_f:
        lines = utt2spk_f.readlines()
    for i in lines:
        i=i.strip().split()
        if suffix == '' or suffix is None:
            key_i = i[0]
            value_spk = i[1]
        else:
            key_i = i[0]+'_'+suffix
            value_spk = i[1]+'_'+suffix
        if key_i in temp_dict:
            raise ValueError('The key must be unique.')
        temp_dict[key_i]=value_spk
    return temp_dict


def get_wavscp_dict(wavscp, suffix=''):
    temp_dict={}
    with open(wavscp, 'r') as wavscp_f:
        lines = wavscp_f.readlines()
    for i in lines:
        i=i.strip().split()
        if suffix == '' or suffix is None:
            key_i = i[0]
        else:
            key_i = i[0]+'_'+suffix
        value_path = i[1]
        if key_i in temp_dict:
            raise ValueError('The key must be unique.')
        temp_dict[key_i]=value_path
    return temp_dict

def get_chunks(seg_dur, audio_id, audio_duration):
    num_chunks = max(1, int(audio_duration / seg_dur))

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(min(i * seg_dur + seg_dur, audio_duration))
        for i in range(num_chunks)
    ]

    return chunk_lst

def prepare_csv(wav_keys, seg_dur, wavscp_dict, utt2spk_dict, samplerate, random_segment=False, storage=None):
    error_case = 0
    spk2dur = {}
    entry = []
    logs = []
    for audio_id in wav_keys:
        if error_case > len(wav_keys) * 0.1:
            raise ValueError('More than 10% of data loads fail. Please check it')

        wav_file = wavscp_dict[audio_id]
        spk_id = utt2spk_dict[audio_id]

        try:
            signal, fs = torchaudio.load(wav_file)
        except Exception:
            str_log = 'Error loading: failed to open file %s.' % wav_file
            print(str_log)
            logs.append(str_log)
            error_case += 1
            continue
        if fs != samplerate:
            str_log = 'Error loading: unexpected file sample rate for %s: %d.'%(wav_file, fs)
            print(str_log)
            logs.append(str_log)
            error_case += 1
            continue
        if signal.shape[0] > 1:
            str_log = 'Error loading: unexpected multichannel file %s.'%wav_file
            print(str_log)
            logs.append(str_log)
            error_case += 1
            continue

        signal = signal.squeeze(0)
        audio_duration = signal.shape[0] / samplerate
        if not random_segment:
            uniq_chunks_list = get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]

                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    s,
                    e,
                    spk_id,
                ]
                entry.append(csv_line)
                if spk_id in spk2dur:
                    spk2dur[spk_id]+=seg_dur
                else:
                    spk2dur[spk_id]=seg_dur
        else:
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                0,
                str(audio_duration),
                spk_id,
            ]
            entry.append(csv_line)
            if spk_id in spk2dur:
                    spk2dur[spk_id]+=audio_duration
            else:
                spk2dur[spk_id]=audio_duration

    if storage is not None:
        storage.put((spk2dur, entry, logs))
    else:
        return spk2dur, entry, logs

def main(args):
    conf = {
        "seg_dur": args.seg_dur,
        "min_spkdur": args.min_spkdur,
        "random_seg": args.random_seg,
    }
    wavscp = args.data_dir + '/wav.scp'
    utt2spk = args.data_dir + '/utt2spk'
    if not os.path.exists(wavscp):
        raise FileNotFoundError('%s not found.' % wavscp)
    if not os.path.exists(utt2spk):
        raise FileNotFoundError('%s not found.' % utt2spk)

    save_opt = os.path.join(args.data_dir, 'opt.pkl')
    save_csv = os.path.join(args.data_dir, 'train.csv')
    save_log = os.path.join(args.data_dir, 'csv.log')

    if os.path.exists(save_opt) and os.path.exists(save_csv):
        with open(save_opt, "rb") as f:
            conf_old = pickle.load(f)
        if conf_old != conf:
            raise ValueError('An old different opt file is found. Please confirm and manually delete it.')
        else:
            logger.info('%s has been prepared ready.' % args.data_dir)
            sys.exit(0)

    logger.info("Starting preparation for %s." % args.data_dir)
    start_time = time.time()

    name = os.path.basename(args.data_dir)
    wavscp_dict = get_wavscp_dict(wavscp, name)
    utt2spk_dict = get_utt2spk_dict(utt2spk, name)

    spk2utt_dict={}
    for utt,spk in utt2spk_dict.items():
        if spk in spk2utt_dict:
            spk2utt_dict[spk].append(utt)
        else:
            spk2utt_dict[spk] = [utt]

    def err_call_back(err):
        print(f'Error occurs: {err}')

    pool = mp.Pool(args.nj)
    manager = mp.Manager()
    storage = manager.Queue()
    spk2dur_tot = {}
    entry_tot = []
    logs_tot = []
    csv_output = [["ID", "dur", "path", "start", "stop", "spk"]]
    for n in range(args.nj):
        wav_keys = list(wavscp_dict.keys())[n::args.nj]
        # prepare_csv(wav_keys, args.seg_dur, wavscp_dict, utt2spk_dict, args.sample_rate, args.random_seg, storage)
        pool.apply_async(
            prepare_csv,
            args=(wav_keys, args.seg_dur, wavscp_dict, utt2spk_dict, args.sample_rate, args.random_seg, storage),
            error_callback=err_call_back)
    pool.close()
    pool.join()

    if storage.empty():
        raise ValueError('No data is collected.')

    while not storage.empty():
        spk2dur, entry, logs = storage.get()
        for spk, dur in spk2dur.items():
            if spk in spk2dur_tot:
                spk2dur_tot[spk]+=dur
            else:
                spk2dur_tot[spk]=dur
        entry_tot += entry
        logs_tot += logs

    for seg in entry_tot:
        spk_id = seg[-1]
        if spk2dur_tot[spk_id] > args.min_spkdur:
            csv_output.append(seg)

    # Write the csv file
    with open(save_csv, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Write the log file
    if len(logs_tot)!=0:
        logger.warning('WARNING: some loading errors occur and please check the %s' % save_log)
        with open(save_log, mode="w") as csv_f:
            for line in logs_tot:
                csv_f.write(line)

    # Write the config file
    with open(save_opt, "wb") as f:
        pickle.dump(conf, f)
    logger.info('%s is prepared ready. It takes time %.2fs.' % (args.data_dir,time.time()-start_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default="data/vox2_dev",
                        help="train data dir")
    parser.add_argument('--seg_dur',
                        type=float,
                        default=4.0,
                        help="seg len")
    parser.add_argument('--sample_rate',
                        type=int,
                        default=16000,
                        help="sample rate")
    parser.add_argument('--min_spkdur',
                        type=float,
                        default=0.0,
                        help="min spk dur")
    parser.add_argument('--random_seg',
                        type=bool,
                        default=False,
                        help="crop the wav or not")
    parser.add_argument('--nj',
                        type=int,
                        default=8,
                        help="num of process")

    args = parser.parse_args()
    main(args)
