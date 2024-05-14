import os
import glob
import argparse
from scipy.io import wavfile
from tqdm import tqdm

def main(args):
    files = glob.glob(f'{args.musan_dir}/*/*/*.wav')
    if len(files) == 0:
        raise Exception('Musan files not found.')

    audlen = 16000*5
    audstr = 16000*3

    pbar = tqdm(total=len(files))
    pbar.set_description("Processing musan")

    with open(args.out_scp, 'w') as f:
        for idx, file in enumerate(files):
            fs,aud = wavfile.read(file)
            writedir = os.path.splitext(file.replace('musan', 'musan_split'))[0]
            os.makedirs(writedir, exist_ok=True)
            for st in range(0,len(aud)-audlen,audstr):
                wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])
                f.write('{} {}\n'.format('_'.join(writedir.split('/')[-4:]), writedir+'/%05d.wav'%(st/fs)))
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--musan_dir',
                        type=str,
                        default="",
                        help="musan dir")
    parser.add_argument('--out_scp',
                        type=str,
                        default="",
                        help="out scp")
    args = parser.parse_args()
    main(args)
