#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
webui
'''

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import shutil
import librosa
import python_speech_features
import time
from LIA_Model import LIA_Model
import os
from tqdm import tqdm
import argparse
import numpy as np
from torchvision import transforms
from templates import *
import argparse
import shutil
import librosa
import python_speech_features
import importlib.util
import time
import os
import time
import numpy as np
import asyncio
from flask import Flask,json,request,jsonify,send_from_directory,render_template
from flask_cors import CORS
import openai
import gtts
import tempfile
from moviepy.editor import *
from dotenv import load_dotenv

# template_path = os.path.join(os.getcwd(), 'AniTalker/code/templates')
load_dotenv()

app = Flask(__name__,static_folder='static')

# CORS(app)
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5000"}})
openai.api_key = "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"
def getResponseFromGPT(text:str)->str:
    return openai.ChatCompletion.create(
        model="gpt-4o",  # 使用正確的 ChatCompletion 端點

        messages=[
            {"role": "system", "content": "你是一個可聊天英文文法檢查助理。"},
            {"role": "user", "content": text}
        ]
    )["choices"][0]["message"]["content"]


def tts(text:str)->str:
    audio = gtts.gTTS(text,lang="en-us")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    audio_path = os.path.join(os.getcwd(),"/AniTalker/audio/",temp_file.name)
    audio.save(audio_path)
    return audio_path

def check_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    if package_spec is None:
        print(f"{package_name} is not installed.")
        return False
    else:
        print(f"{package_name} is installed.")
        return True

def frames_to_video(input_path, audio_path, output_path, fps=25):
    image_files = [os.path.join(input_path, img) for img in sorted(os.listdir(input_path))]
    clips = [ImageClip(m).set_duration(1/fps) for m in image_files]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256
    return img / 255.0

def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]
    return imgs_norm

def saved_image(img_tensor, img_path):
    toPIL = transforms.ToPILImage()
    img = toPIL(img_tensor.detach().cpu().squeeze(0))  # 使用squeeze(0)来移除批次维度
    img.save(img_path)
    
def main(args):
    print(f"asdad{os.getcwd()}")
    frames_result_saved_path = os.path.join(args.result_path, 'frames')
    os.makedirs(frames_result_saved_path, exist_ok=True)
    test_image_name = os.path.splitext(os.path.basename(args.test_image_path))[0]
    print(f"image show{test_image_name}")
    audio_name = os.path.splitext(os.path.basename(args.test_audio_path))[0]
    predicted_video_256_path = os.path.join(args.result_path,  f'{test_image_name}-{audio_name}.mp4')
    predicted_video_512_path = os.path.join(args.result_path,  f'{test_image_name}-{audio_name}_SR.mp4')
 
    #======Loading Stage 1 model=========
    lia = LIA_Model(motion_dim=args.motion_dim, fusion_type='weighted_sum')
    lia.load_lightning_model(args.stage1_checkpoint_path)
    lia.to(args.device)
    #============================

    conf = ffhq256_autoenc()
    conf.seed = args.seed
    conf.decoder_layers = args.decoder_layers
    conf.infer_type = args.infer_type
    conf.motion_dim = args.motion_dim
    
    if args.infer_type == 'mfcc_full_control':
        conf.face_location=True
        conf.face_scale=True
        conf.mfcc = True
    elif args.infer_type == 'mfcc_pose_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = True
    elif args.infer_type == 'hubert_pose_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = False
    elif args.infer_type == 'hubert_audio_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = False
    elif args.infer_type == 'hubert_full_control':
        conf.face_location=True
        conf.face_scale=True
        conf.mfcc = False
    else: 
        print('Type NOT Found!')
        exit(0)

    print(f"test_image_path: {args.test_image_path}")

    if not os.path.exists(args.test_image_path):
        print(f'{args.test_image_path} does not exist!')
        exit(0)
    
    if not os.path.exists(args.test_audio_path):
        print(f'{args.test_audio_path} does not exist!')
        exit(0)
    
    img_source = img_preprocessing(args.test_image_path, args.image_size).to(args.device)
    one_shot_lia_start, one_shot_lia_direction, feats = lia.get_start_direction_code(img_source, img_source, img_source, img_source)

    #======Loading Stage 2 model=========
    model = LitModel(conf)
    state = torch.load(args.stage2_checkpoint_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.ema_model.eval()
    model.ema_model.to(args.device)
    #=================================
    
    #======Audio Input=========
    if conf.infer_type.startswith('mfcc'):
        # MFCC features
        wav, sr = librosa.load(args.test_audio_path, sr=16000)
        input_values = python_speech_features.mfcc(signal=wav, samplerate=sr, numcep=13, winlen=0.025, winstep=0.01)
        d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
        d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
        audio_driven_obj = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
        frame_start, frame_end = 0, int(audio_driven_obj.shape[0]/4)
        audio_start, audio_end = int(frame_start * 4), int(frame_end * 4) # The video frame is fixed to 25 hz and the audio is fixed to 100 hz
        
        audio_driven = torch.Tensor(audio_driven_obj[audio_start:audio_end,:]).unsqueeze(0).float().to(args.device)
        
    elif conf.infer_type.startswith('hubert'):
        # Hubert features
        if not os.path.exists(args.test_hubert_path):
            
            if not check_package_installed('transformers'):
                print('Please install transformers module first.')
                exit(0)
            hubert_model_path = os.getcwd()+'/AniTalker/ckpts/chinese-hubert-large'
            print(hubert_model_path)
            if not os.path.exists(hubert_model_path):
                print('Please download the hubert weight into the ckpts path first.')
                exit(0)
            print('You did not extract the audio features in advance, extracting online now, which will increase processing delay')

            start_time = time.time()

            # load hubert model
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            audio_model = HubertModel.from_pretrained(hubert_model_path).to(args.device)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
            audio_model.feature_extractor._freeze_parameters()
            audio_model.eval()

            # hubert model forward pass
            audio, sr = librosa.load(args.test_audio_path, sr=16000)
            input_values = feature_extractor(audio, sampling_rate=16000, padding=True, do_normalize=True, return_tensors="pt").input_values
            input_values = input_values.to(args.device)
            ws_feats = []
            with torch.no_grad():
                outputs = audio_model(input_values, output_hidden_states=True)
                for i in range(len(outputs.hidden_states)):
                    ws_feats.append(outputs.hidden_states[i].detach().cpu().numpy())
                ws_feat_obj = np.array(ws_feats)
                ws_feat_obj = np.squeeze(ws_feat_obj, 1)
                ws_feat_obj = np.pad(ws_feat_obj, ((0, 0), (0, 1), (0, 0)), 'edge') # align the audio length with video frame
            
            execution_time = time.time() - start_time
            print(f"Extraction Audio Feature: {execution_time:.2f} Seconds")

            audio_driven_obj = ws_feat_obj
        else:
            print(f'Using audio feature from path: {args.test_hubert_path}')
            audio_driven_obj = np.load(args.test_hubert_path)

        frame_start, frame_end = 0, int(audio_driven_obj.shape[1]/2)
        audio_start, audio_end = int(frame_start * 2), int(frame_end * 2) # The video frame is fixed to 25 hz and the audio is fixed to 50 hz
        
        audio_driven = torch.Tensor(audio_driven_obj[:,audio_start:audio_end,:]).unsqueeze(0).float().to(args.device)
    #============================
    
    # Diffusion Noise
    noisyT = torch.randn((1,frame_end, args.motion_dim)).to(args.device)
    
    #======Inputs for Attribute Control=========
    if os.path.exists(args.pose_driven_path):
        pose_obj = np.load(args.pose_driven_path)

        if len(pose_obj.shape) != 2:
            print('please check your pose information. The shape must be like (T, 3).')
            exit(0)
        if pose_obj.shape[1] != 3:
            print('please check your pose information. The shape must be like (T, 3).')
            exit(0)

        if pose_obj.shape[0] >= frame_end:
            pose_obj = pose_obj[:frame_end,:]
        else:
            padding = np.tile(pose_obj[-1, :], (frame_end - pose_obj.shape[0], 1))
            pose_obj = np.vstack((pose_obj, padding))
            
        pose_signal = torch.Tensor(pose_obj).unsqueeze(0).to(args.device) / 90 # 90 is for normalization here
    else:
        yaw_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.pose_yaw
        pitch_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.pose_pitch
        roll_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.pose_roll
        pose_signal = torch.cat((yaw_signal, pitch_signal, roll_signal), dim=-1)
    
    pose_signal = torch.clamp(pose_signal, -1, 1)

    face_location_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.face_location
    face_scae_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.face_scale
    #===========================================

    start_time = time.time()

    #======Diffusion Denosing Process=========
    generated_directions = model.render(one_shot_lia_start, one_shot_lia_direction, audio_driven, face_location_signal, face_scae_signal, pose_signal, noisyT, args.step_T, control_flag=args.control_flag)
    #=========================================
    
    execution_time = time.time() - start_time
    print(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

    generated_directions = generated_directions.detach().cpu().numpy()
    
    start_time = time.time()
    #======Rendering images frame-by-frame=========
    for pred_index in tqdm(range(generated_directions.shape[1])):
        ori_img_recon = lia.render(one_shot_lia_start, torch.Tensor(generated_directions[:,pred_index,:]).to(args.device), feats)
        ori_img_recon = ori_img_recon.clamp(-1, 1)
        wav_pred = (ori_img_recon.detach() + 1) / 2
        saved_image(wav_pred, os.path.join(frames_result_saved_path, "%06d.png"%(pred_index)))
    #==============================================
    
    execution_time = time.time() - start_time
    print(f"Renderer Model: {execution_time:.2f} Seconds")

    frames_to_video(frames_result_saved_path, args.test_audio_path, predicted_video_256_path)
    
    shutil.rmtree(frames_result_saved_path)
    
    # Enhancer
    if args.face_sr and check_package_installed('gfpgan'):
        from face_sr.face_enhancer import enhancer_list
        import imageio

        # Super-resolution
        imageio.mimsave(predicted_video_512_path+'.tmp.mp4', enhancer_list(predicted_video_256_path, method='gfpgan', bg_upsampler=None), fps=float(25))
        
        # Merge audio and video
        video_clip = VideoFileClip(predicted_video_512_path+'.tmp.mp4')
        audio_clip = AudioFileClip(predicted_video_256_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(predicted_video_512_path, codec='libx264', audio_codec='aac')
        
        os.remove(predicted_video_512_path+'.tmp.mp4')
    
    if args.face_sr:
        return predicted_video_256_path, predicted_video_512_path
    else:
        return predicted_video_256_path, predicted_video_256_path

def generate_video(uploaded_img, uploaded_audio, infer_type, 
        pose_yaw, pose_pitch, pose_roll, face_location, face_scale, step_T, device, face_sr, seed, face_crop):
    if uploaded_img is None or uploaded_audio is None:
        return None,None,"Error: Input image or audio file is empty. Please check and upload both files."
    
    model_mapping = {
        "mfcc_pose_only": os.getcwd()+"/AniTalker/ckpts/stage2_pose_only_mfcc.ckpt",
        "mfcc_full_control": os.getcwd()+"/AniTalker/ckpts/stage2_more_controllable_mfcc.ckpt",
        "hubert_audio_only": os.getcwd()+"/AniTalker/ckpts/stage2_audio_only_hubert.ckpt",
        "hubert_pose_only": os.getcwd()+"/AniTalker/ckpts/stage2_pose_only_hubert.ckpt",
        "hubert_full_control": os.getcwd()+"/AniTalker/ckpts/stage2_full_control_hubert.ckpt",
    }

    if face_crop:
        from data_preprocess.crop_image2 import crop_image
        print("==> croping source_img")
        crop_path = os.path.join(os.path.dirname(uploaded_img), 'crop_'+os.path.basename(uploaded_img))
        try:
            crop_image(uploaded_img, crop_path)
            if os.path.exists(crop_path):
                uploaded_img = crop_path
        except:
            print('==> crop image failed, use original source for animate')

    stage2_checkpoint_path = model_mapping.get(infer_type, "default_checkpoint.ckpt")
    try:
        args = argparse.Namespace(
            infer_type=infer_type,
            test_image_path=uploaded_img,
            test_audio_path=uploaded_audio,
            test_hubert_path='',
            result_path=os.getcwd()+'/',
            stage1_checkpoint_path=os.getcwd()+'/AniTalker/ckpts/stage1.ckpt',
            stage2_checkpoint_path=stage2_checkpoint_path,
            seed=seed,
            control_flag=True,
            pose_yaw=pose_yaw,
            pose_pitch=pose_pitch,
            pose_roll=pose_roll,
            face_location=face_location,
            pose_driven_path='not_supported_in_this_mode',
            face_scale=face_scale,
            step_T=step_T,
            image_size=256,
            device=device,
            motion_dim=20,
            decoder_layers=2,
            face_sr=face_sr
        )

        # Save the uploaded audio to the expected path
        # shutil.copy(uploaded_audio, args.test_audio_path)

        # Run the main function
        output_256_video_path, output_512_video_path = main(args)

        # Check if the output video file exists
        if not os.path.exists(output_256_video_path):
            return None,None, "Error: Video generation failed. Please check your inputs and try again."
        if output_256_video_path == output_512_video_path:
            return output_256_video_path, output_256_video_path, "Video (256*256 only) generated successfully!"
        return output_256_video_path, output_512_video_path, "Video generated successfully!"

    except Exception as e:
        return None, None,f"Error: An unexpected error occurred - {str(e)}"

default_values = {
    "pose_yaw": 0,
    "pose_pitch": 0,
    "pose_roll": 0,
    "face_location": 0.5,
    "face_scale": 0.5,
    "step_T": 50,
    "seed": 0,
    "device": "cuda"
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/file/<filename>')
def serve_video(filename):
    # 檢查影片是否存在，避免找不到文件的錯誤
    print(os.getcwd()+ filename)
    if os.path.exists(os.path.join(os.getcwd(), filename)):
        return send_from_directory(os.getcwd(), filename,mimetype='video/mp4')
    else:
        return "File not found", 404

@app.route("/api/generate_video",methods=["POST"])
def generate():
    json_data = request.get_json()
    uploaded_img = json_data.get("img")
    prompt = json_data.get("text")
    res = getResponseFromGPT(prompt)
    uploaded_audio = tts(res)

    inputs={
        "uploaded_img":uploaded_img, 
        "uploaded_audio":uploaded_audio, 
        "infer_type":'hubert_audio_only',
        "pose_yaw":1, 
        "pose_pitch":1, 
        "pose_roll":1, 
        "face_location":0, 
        "face_scale":0, 
        "step_T":50, 
        "device":"cpu", 
        "face_sr":False, 
        "seed":0,
        "face_crop":False
    }
    print("Inputs passed to generate_video:", inputs)

    video_256,video_512,msg = generate_video(**inputs)
    if video_256 is None:
        print("Error: video_256 is None. No video was generated.")
    print(f"video_256: {video_256}, video_512: {video_512}, msg: {msg}")

    return jsonify({"video_path":video_256.replace(os.getcwd(),"http://127.0.0.1:5000/file"),"msg":msg,"status":200,"result":res})

# @app.route('/api/translate', methods=['POST'])
# def translate():
#     json_data = request.get_json()
#     print(f"Text to be translated: {json_data}")

#     # 調用翻譯 API
#     target_url = 'https://api.mymemory.translated.net/get'
#     params = {
#         'q': json_data,
#         'langpair': 'en|zh'
#     }
#     try:
#         response = request.get(target_url, params=params)
#         response_data = response.json()
#         translated_text = response_data['responseData']['translatedText']
#         return jsonify({"translatedText": translated_text})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
