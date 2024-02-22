# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script uses pretrained models to perform speaker visual embeddings extracting.
This script use following open source models:
    1. Face detection: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
    2. Active speaker detection: TalkNet, https://github.com/TaoRuijie/TalkNet-ASD
    3. Face quality assessment: https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
    4. Face recognition: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
Processing pipeline: 
    1. Face detection (input: video frames)
    2. Active speaker detection (input: consecutive face frames, audio)
    3. Face quality assessment (input: video frames)
    4. Face recognition (input: video frames)
"""


import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os, time, torch, cv2, pickle, python_speech_features

import vision_tools.face_detection as face_detection
import vision_tools.active_speaker_detection as active_speaker_detection
import vision_tools.face_recognition as face_recognition
import vision_tools.face_quality_assessment as face_quality_assessment


class VisionProcesser():
    def __init__(
        self, 
        video_file_path, 
        audio_file_path, 
        audio_vad, 
        out_feat_path, 
        onnx_dir, 
        conf, 
        device='cpu', 
        device_id=0, 
        out_video_path=None
        ):
        # read audio data and check the samplerate.
        fs, self.audio = wavfile.read(audio_file_path)
        assert fs == 16000, '[ERROR]: Samplerate of wav must be 16000'
        # convert time interval to integer sampling point interval.
        audio_vad = [[int(i*16000), int(j*16000)] for (i, j) in audio_vad]
        self.video_id = os.path.basename(video_file_path).rsplit('.', 1)[0]

        # read video data
        self.cap = cv2.VideoCapture(video_file_path)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print('video %s info: w: {}, h: {}, count: {}, fps: {}'.format(w, h, self.count, self.fps) % self.video_id)

        # initial vision models
        self.face_detector = face_detection.Predictor(onnx_dir, device, device_id)
        self.speaker_detector = active_speaker_detection.ASDTalknet(onnx_dir, device, device_id)
        self.face_quality_evaluator = face_quality_assessment.FaceQualityAssess(onnx_dir, device, device_id)
        self.face_embs_extractor = face_recognition.FaceRecIR101(onnx_dir, device, device_id)

        # store facial feats along with the necessary information.
        self.active_facial_embs = {'frameI':np.empty((0,), dtype=int), 'feat':np.empty((0, 512), dtype=np.float32)}

        self.audio_vad = audio_vad
        self.out_video_path = out_video_path
        self.out_feat_path = out_feat_path

        self.min_track = conf['min_track']
        self.num_failed_det = conf['num_failed_det']
        self.crop_scale = conf['crop_scale']
        self.min_face_size = conf['min_face_size']
        self.face_det_stride = conf['face_det_stride']
        self.shot_stride = conf['shot_stride']

        if self.out_video_path is not None:
            # save the active face detection results video (for debugging).
            self.v_out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(w), int(h)))

        # record the time spent by each module.
        self.elapsed_time = {'faceTime':[], 'trackTime':[], 'cropTime':[],'asdTime':[], 'visTime':[], 'featTime':[]}

    def run(self):
        frames, face_det_frames = [], []
        for [audio_sample_st, audio_sample_ed] in self.audio_vad:
            # frame_st and frame_ed are the starting and ending frames of current interval.
            frame_st, frame_ed = int(audio_sample_st/640), int(audio_sample_ed/640)
            num_frames = frame_ed - frame_st + 1
            # go to frame 'frame_st'.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_st)
            index = 0
            for _ in range(num_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                if index % self.face_det_stride==0:
                    face_det_frames.append(frame)
                frames.append(frame)
                if (index + 1) % self.shot_stride==0:
                    audio = self.audio[(frame_st + index + 1 - self.shot_stride)*640:(frame_st + index + 1)*640]
                    self.process_one_shot(frames, face_det_frames, audio, frame_st + index + 1 - self.shot_stride)
                    frames, face_det_frames = [], []
                index += 1
            if len(frames) != 0:
                audio = self.audio[(frame_st + index - len(frames))*640:(frame_st + index)*640]
                self.process_one_shot(frames, face_det_frames, audio, frame_st + index - len(frames))
                frames, face_det_frames = [], []

        self.cap.release()
        if self.out_video_path is not None:
            self.v_out.release()

        active_facial_embs = {'embeddings':self.active_facial_embs['feat'], 'times': self.active_facial_embs['frameI']*0.04}
        pickle.dump(active_facial_embs, open(self.out_feat_path, 'wb'))

        # print elapsed time
        all_elapsed_time = 0
        for k in self.elapsed_time:
            all_elapsed_time += sum(self.elapsed_time[k])
            self.elapsed_time[k] = sum(self.elapsed_time[k])
        elapsed_time_msg = 'The total processing time for %s is %.2fs, including' % (self.video_id, all_elapsed_time)
        for k in self.elapsed_time:
            elapsed_time_msg += ' %s %.2fs,'%(k, self.elapsed_time[k])
        print(elapsed_time_msg[:-1]+'.')

    def process_one_shot(self, frames, face_det_frames, audio, frame_st=None):
        curTime = time.time()
        dets = self.face_detection(face_det_frames)
        faceTime = time.time()

        allTracks, vidTracks = [], []
        allTracks.extend(self.track_shot(dets))
        trackTime = time.time()

        for ii, track in enumerate(allTracks):
            vidTracks.append(self.crop_video(track, frames, audio))
        cropTime = time.time()

        scores = self.evaluate_asd(vidTracks)
        asdTime = time.time()

        active_facial_embs = self.evaluate_fr(frames, vidTracks, scores)
        self.active_facial_embs['frameI'] = np.append(self.active_facial_embs['frameI'], active_facial_embs['frameI'] + frame_st)
        self.active_facial_embs['feat'] = np.append(self.active_facial_embs['feat'], active_facial_embs['feat'], axis=0)
        featTime = time.time()

        if self.out_video_path is not None:
            self.visualization(frames, vidTracks, scores)
        visTime = time.time()

        self.elapsed_time['faceTime'].append(faceTime-curTime)
        self.elapsed_time['trackTime'].append(trackTime-faceTime)
        self.elapsed_time['cropTime'].append(cropTime-trackTime)
        self.elapsed_time['asdTime'].append(asdTime-cropTime)
        self.elapsed_time['featTime'].append(featTime-asdTime)
        if self.out_video_path is not None:
            self.elapsed_time['visTime'].append(visTime-featTime)

    def face_detection(self, frames):
        dets = []
        for fidx, image in enumerate(frames):
            image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, _, probs = self.face_detector(image_input, top_k=10, prob_threshold=0.9)
            bboxes = torch.cat([bboxes, probs.reshape(-1, 1)], dim=-1)
            dets.append([])
            for bbox in bboxes:
                frame_idex = fidx * self.face_det_stride
                dets[-1].append({'frame':frame_idex, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        return dets

    def bb_intersection_over_union(self, boxA, boxB, evalCol=False):
        # IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self, scene_faces):
        # Face tracking
        tracks = []
        while True:   # continuously search for consecutive faces.
            track = []
            for frame_faces in scene_faces:
                for face in frame_faces:
                    if track == []:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:  # the face does not interrupt for 'num_failed_det' frame.
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        # minimum IOU between consecutive face.
                        if iou > 0.5:
                            track.append(face)
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if track == []:
                break
            elif len(track) > 1 and track[-1]['frame'] - track[0]['frame'] + 1 >= self.min_track:
                frame_num = np.array([ f['frame'] for f in track ])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                frameI = np.arange(frame_num[0], frame_num[-1]+1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn  = interp1d(frame_num, bboxes[:,ij]) # missing boxes can be filled by interpolation.
                    bboxesI.append(interpfn(frameI))
                bboxesI  = np.stack(bboxesI, axis=1)
                if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > self.min_face_size:  # need face size > min_face_size
                    tracks.append({'frame':frameI,'bbox':bboxesI})
        return tracks

    def crop_video(self, track, frames, audio):
        # crop the face clips
        crop_frames = []
        dets = {'x':[], 'y':[], 's':[]}
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y
        for fidx, frame in enumerate(track['frame']):
            cs  = self.crop_scale
            bs  = dets['s'][fidx]   # detection box size
            bsi = int(bs * (1 + 2 * cs))  # pad videos by this amount 
            image = frames[frame]
            frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = dets['y'][fidx] + bsi  # BBox center Y
            mx  = dets['x'][fidx] + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            crop_frames.append(cv2.resize(face, (224, 224)))
        cropaudio = audio[track['frame'][0]*640:(track['frame'][-1]+1)*640]
        return {'track':track, 'proc_track':dets, 'data':[crop_frames, cropaudio]}

    def evaluate_asd(self, tracks):
        # active speaker detection by pretrained TalkNet
        all_scores = []
        for ins in tracks:
            video, audio = ins['data']
            audio_feature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
            video_feature = []
            for frame in video:
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                video_feature.append(face)
            video_feature = np.array(video_feature)
            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
            audio_feature = audio_feature[:int(round(length * 100)),:]
            video_feature = video_feature[:int(round(length * 25)),:,:]
            audio_feature = np.expand_dims(audio_feature, axis=0).astype(np.float32)
            video_feature = np.expand_dims(video_feature, axis=0).astype(np.float32)
            score = self.speaker_detector(audio_feature, video_feature)
            all_score = np.round(score, 1).astype(float)
            all_scores.append(all_score)	
        return all_scores

    def evaluate_fr(self, frames, tracks, scores):
        # extract high-quality facial embeddings 
        faces = [[] for i in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = np.mean(s)
                bbox = track['track']['bbox'][fidx]
                face = frames[frame][max(int(bbox[1]), 0):min(int(bbox[3]), frames[frame].shape[0]), max(int(bbox[0]), 0):min(int(bbox[2]), frames[frame].shape[1])]
                faces[frame].append({'track':tidx, 'score':float(s), 'facedata':face})

        active_facial_embs={'frameI':np.empty((0,), dtype=int), 'feat':np.empty((0, 512), dtype=np.float32)}
        for fidx in range(len(faces)):
            if fidx % self.face_det_stride != 0:
                continue
            active_face = None
            active_face_num = 0
            for face in faces[fidx]:
                if face['score'] > 0:
                    active_face = face['facedata']
                    active_face_num += 1
            # process frames containing only one active face.
            if active_face_num == 1:
                # quality assessment
                face_quality_score = self.face_quality_evaluator(active_face)
                if face_quality_score < 0.7:
                    continue
                feature = self.face_embs_extractor(active_face)
                active_facial_embs['frameI'] = np.append(active_facial_embs['frameI'], fidx)
                active_facial_embs['feat'] = np.append(active_facial_embs['feat'], feature, axis=0)
        return active_facial_embs

    def visualization(self, frames, tracks, scores):
        faces = [[] for i in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
                s = np.mean(s)
                faces[frame].append({'track':tidx, 'score':float(s),'bbox':track['track']['bbox'][fidx]})

        colorDict = {0: 0, 1: 255}
        for fidx, image in enumerate(frames):
            for face in faces[fidx]:
                clr = colorDict[int((face['score'] >= 0))]
                txt = round(face['score'], 1)
                cv2.rectangle(image, (int(face['bbox'][0]), int(face['bbox'][1])), (int(face['bbox'][2]), int(face['bbox'][3])),(0,clr,255-clr),10)
                cv2.putText(image,'%s'%(txt), (int(face['bbox'][0]), int(face['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
            self.v_out.write(image)
