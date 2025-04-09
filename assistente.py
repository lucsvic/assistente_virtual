# ===============================================
# Assistente Virtual
# Descrição: Este script é parte de uma assistente virtual genérica, responsável por executar
#            tarefas automatizadas, responder comandos ou auxiliar o usuário conforme necessário.
#            categorizando automaticamente por temas ou tipos de conteúdo.
# OBS: Este script pode ser integrado a fluxos automatizados de organização pessoal ou profissional.
# ===============================================


import pyttsx3#Conversão de texto para fala
import speech_recognition as sr
from playsound import playsound
import random
import datetime

import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set()
from modules import commandos
comandos = commandos.comandos
respostas = commandos.respostas

hour = datetime.datetime.now().strftime("%H:%M")
date = datetime.date.today().strftime("%d/%B/%Y")

nome_assistente = 'assistente'

chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'

def search(frase):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + frase)


#search("Computador")
MODEL_TYPES = ['EMOÇÃO']


#Função para carregar modelo pré-treinado. Futura implementação
def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = sorted(list(['neutra','calma','feliz','triste','nervosa','medo','nojo','surpreso']))
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

model_type = 'EMOÇÃO'
loaded_model = load_model_by_name(model_type)

def predict_sound(AUDIO, SAMPLE_RATE, plot = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr=SAMPLE_RATE)
    #print(wav_data.shape)

    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end = True, pad_value = 0)

    for i, data in enumerate(splitted_audio_data.numpy()):
        mfccs_features = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1 ,-1)
        #print(mfccs_scaled_features.shape)
        mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]
        #print(mfccs_scaled_features.shape)

        predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)
        predictions = predictions.argmax(axis = 1)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)
    count_results = [[results.count(x), x] for x in set(results)]
    return max(count_results)[1]

#Futura implementação para detecção de emoções
#emocao = predict_sound('./sound.mp3',loaded_model[2],plot=True)

def speak(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate',200)
    engine.setProperty('volume',1)
    engine.say(audio)
    engine.runAndWait()


def listen_microphone():
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source, duration=0.8)
        print('Ouvindo:')
        audio = microfone.listen(source)
        with open('recordings/speech.wav','wb') as f:
            f.write(audio.get_wav_data())
    try:
        frase = microfone.recognize_google(audio, language = 'pt-BR')
        print(frase)
    except sr.UnknownValueError:
        frase = ''
        print('erro')
    return frase


def trocar_bipe(tipo):
    if tipo == 'assistente':
        playsound("notification.wav")
    elif tipo == 'erro':
        playsound("notification2.wav")

assistente_ativa = False
tempo_inicial = time.time()

while True:
    tempo_atual = time.time()

    result = listen_microphone()

    if assistente_ativa:

        if result.lower() in comandos[0]:
            speak('Até agora minhas fuções são ' + respostas[0])

        elif 'encerrar' in result.lower().split():
            speak(respostas[4][random.randint(0,len(respostas[4])-1)])
        else:
            trocar_bipe('erro')
            speak(respostas[5][random.randint(0,len(respostas[5])-1) ])
            assistente_ativa = False
    else:
        if nome_assistente in result.lower().split():
            print('Estou ouvindo')
            trocar_bipe('assistente')
            assistente_ativa = True
