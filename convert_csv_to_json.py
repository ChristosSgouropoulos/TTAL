import pandas as pd
import librosa
import json 

csv_path = "data/train.csv"
audio_path = "/data/audiocaps/audio"
output_json = "data/train.json"
df = pd.read_csv(csv_path)
list_dict = []
for i, row in df.iterrows():
    youtube_id = row['youtube_id']
    wav_path = audio_path + "/" + youtube_id +f"_{row['start_time']}.wav"
    duration = librosa.get_duration(filename=wav_path)
    captions = row['caption']
    list_dict.append({'captions':captions, 'location': wav_path, 'duration':duration})

with open(output_json, 'w') as f:
    json.dump(list_dict, f, indent=4)




csv_path = "data/val.csv"
audio_path = "/data/audiocaps/audio"
output_json = "data/val.json"
df = pd.read_csv(csv_path)
list_dict = []
for i, row in df.iterrows():
    youtube_id = row['youtube_id']
    wav_path = audio_path + "/" + youtube_id +f"_{row['start_time']}.wav"
    duration = librosa.get_duration(filename=wav_path)
    captions = row['caption']
    list_dict.append({'captions':captions, 'location': wav_path, 'duration':duration})

with open(output_json, 'w') as f:
    json.dump(list_dict, f, indent=4)




csv_path = "data/test.csv"
audio_path = "/data/audiocaps/audio"
output_json = "data/test.json"
df = pd.read_csv(csv_path)
list_dict = []
for i, row in df.iterrows():
    youtube_id = row['youtube_id']
    wav_path = audio_path + "/" + youtube_id +f"_{row['start_time']}.wav"
    duration = librosa.get_duration(filename=wav_path)
    captions = row['caption']
    list_dict.append({'captions':captions, 'location': wav_path, 'duration':duration})

with open(output_json, 'w') as f:
    json.dump(list_dict, f, indent=4)