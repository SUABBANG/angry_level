import librosa
import librosa.display
import numpy as np
import torch
import pickle
import pandas as pd
import evaluate
import joblib
import json

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from transformers import AutoFeatureExtractor
from datasets import Dataset, DatasetDict
from flask_cors import CORS
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

from werkzeug.utils import secure_filename

from flask import Flask, jsonify, request, flash, redirect

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
accuracy = evaluate.load("accuracy")

def calculate_pitch(audio_path):
    import librosa
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    import numpy as np

    pitch_list=[] 
    audio_sample, sampling_rate = librosa.load(audio_path, sr = None)

    S = np.abs(librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sampling_rate)

    shape = np.shape(pitches)
    nb_samples = shape[0]
    nb_windows = shape[1]

    for i in range(0, nb_windows):
        index = magnitudes[:,i].argmax()
        pitch = pitches[index,i]
        pitch_list.append(pitch)
        mean_pitch = np.mean(pitch_list)
    return mean_pitch

def calculate_speech_rate(audio_path):
    import librosa

    audio, sr = librosa.load(audio_path)

    non_silent_intervals = librosa.effects.split(audio, top_db=20)

    speech_durations = [librosa.get_duration(y=audio[start:end], sr=sr) for start, end in non_silent_intervals]

    average_speech_rate = len(speech_durations) / sum(speech_durations)
  
    return average_speech_rate

def calculate_decibel(audio_path):
    # 오디오 파일을 로드합니다.
    audio, sr = librosa.load(audio_path)

    # 시간-주파수 분석을 수행합니다.
    stft = np.abs(librosa.stft(audio))

    # 파워 스펙트럼을 계산합니다.
    power_spec = librosa.power_to_db(stft**2)

    # 평균 데시벨 값을 계산합니다.
    max_db = np.max(power_spec)
    
    return max_db

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    global ok_result
    ok_result = [i for i in range(len(predictions)) if predictions[i] == eval_pred.label_ids[i]]
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def preprocess_function(examples):
    audio_arrays = [x for x in examples["audio"]]

    inputs = feature_extractor(
        audio_arrays, sampling_rate=16000, padding_value=0.0,do_normalize = True,truncation=True,max_length=80000
    )
    return inputs

def load_audio(audio_dir):
    sample_rate=48000

    #파일 로드
    waveform, _ = librosa.load(audio_dir, sr=sample_rate)

    
    return waveform

def predict_emotion_probabilities(audio_path, mel_array,trained_model):
    trained_model = trained_model
    
    inputs = torch.tensor(mel_array).unsqueeze(0)
    
    logits = trained_model(inputs).logits
    predicted_probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    return predicted_probabilities

def predict_emotion(audio_path,mel_array,trained_model):
    trained_model = trained_model
    
    inputs = torch.tensor(mel_array).unsqueeze(0)
    
    logits = trained_model(inputs).logits
    predicted_label = torch.argmax(logits, dim=1).item()
    
    return predicted_label
 
label2id = {'분노':'1','기쁨':'0'}
id2label = {'1':'분노','0':'기쁨'}
num_labels = len(id2label)

           
# 모델 
# 1 model load
trained_model = AutoModelForAudioClassification.from_pretrained(
"C:/Users/user/angry_level/model/checkpoint-3120", num_labels=num_labels, label2id=label2id, id2label=id2label
) 
# C:/Users/user/angry_level/checkpoint-3120

# 2 model load
filename = 'C:/Users/user/angry_level/model/xgb_model_ver4.sav'
loaded_model = pickle.load(open(filename, 'rb'))

from werkzeug.utils import secure_filename
import audioread

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = './files/'
ALLOWED_EXTENSIONS = {'wav'}

app=Flask(__name__)

# main update
@app.route('/main', methods=['GET','POST'])

def main():
    print(request.method) # GET or POST
    # UPLOAD SOUND FILE
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        print(f.filename)
        global filename
        filename = f.filename + '.wav'
        if filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(filename):
            print(filename)
            f.save(UPLOAD_FOLDER + secure_filename(filename))
            sound_file = UPLOAD_FOLDER + filename
            print(sound_file)
            
 
    waveform = load_audio(sound_file) # audio   
    
    # 2 predict
    predicted_probabilities = predict_emotion_probabilities(sound_file, waveform,trained_model)
    label_ratios = {
    '0 단계': predicted_probabilities[0],
    '분노': predicted_probabilities[1],
    }   
    predicted_label = predict_emotion(sound_file,waveform,trained_model)    
    
    print("predicted_label: ",predicted_label)

    if predicted_label == 1 : #angry label
        print("It is classified as a black consumer.")
        decibel = calculate_decibel(sound_file)
        print("decibel:", decibel)
        pitch = calculate_pitch(sound_file)
        print("pitch:", pitch)
        speech_rate = calculate_speech_rate(sound_file)
        print("speech_rate:", speech_rate)
        
        data = np.array([[pitch,decibel,speech_rate]])
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        
        print(data)
        
        predicted_label = loaded_model.predict(data)
        
        predicted_probabilities = loaded_model.predict_proba(data)
        anger_ratios = predicted_probabilities[0]

        print("1st Stage of Anger:", float(anger_ratios[0]))
        print("2nd Stage of Anger:", float(anger_ratios[1]))
        print("3rd Stage of Anger:", float(anger_ratios[2]))
        
        if predicted_label == 0:
            #1
            answer = "It is a black consumer. It is classified into 1st stages of anger."
            result_lavel =1
        elif predicted_label == 1:
            #2
            answer = "It is a black consumer. It is classified into 2nd stages of anger."
            result_lavel =2
        else:
            #3
            answer = "It is a black consumer. It is classified into 3rd stages of anger."
            result_lavel =3
        
        data = {'sentence': answer, "1st": round(float(anger_ratios[0]),3), "2nd": round(float(anger_ratios[1]),3),"3rd": round(float(anger_ratios[2]),3)}
    else:
        ratio_list = []
        answer = "It is not classified as a black consumer."
        for label, ratio in label_ratios.items():
            print(ratio)
            ratio_list.append(float(ratio))
            print(ratio_list)
        result_lavel = 0
        data = {'sentence': answer, "Normal": round(ratio_list[0],3), "black": round(float(ratio_list[1]),3)}
    print("result_level",result_lavel)
    return {'result_lavel':result_lavel}

if __name__ == "__main__":
    app.run(debug=True)

# 추가
CORS(
    app, 
    resources={r'*': {'origins': '*'}}, 
    supports_credentials=True)

