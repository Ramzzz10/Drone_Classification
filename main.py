import pyaudio
import numpy as np
import xgboost as xgb
import librosa
# import signal
# import soundfile as sf

model = xgb.Booster(model_file='/Users/midasxlr/Desktop/Drone_Classification/code and models/XGBoost')


def record_sound(min_duration=0.5, sr=44100, frames_per_buffer=512):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer=frames_per_buffer)
    frames = []
    recorded_duration = 0

    try:
        while recorded_duration < min_duration:
            data = stream.read(frames_per_buffer)
            frames.append(np.frombuffer(data, dtype=np.float32))
            recorded_duration += frames_per_buffer / sr
    except OSError as e:
        print("Произошла ошибка записи звука:", e)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)

def extract_features(audio, sr, duration=1, offset=0.25):
    features = []

    # Получаем количество сегментов, если аудиофайл короче 1 секунды
    num_segments = int(np.ceil(len(audio) / (sr * duration)))

    for i in range(num_segments):
        start = int(sr * duration * i)
        end = int(min(len(audio), sr * duration * (i + 1)))

        segment = audio[start:end]  # Выбираем i-й сегмент аудио

        # Извлекаем признаки из текущего сегмента
        mfccs = librosa.feature.mfcc(y=segment, sr=sr)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        mel = librosa.feature.melspectrogram(y=segment, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, fmin=100.0, n_bands=6)

        # Добавляем средние значения признаков сегмента в список признаков
        features.append(np.mean(mfccs, axis=1))
        features.append(np.mean(chroma, axis=1))
        features.append(np.mean(mel, axis=1))
        features.append(np.mean(contrast, axis=1))

    # Конкатенируем признаки из всех сегментов
    if len(features) > 0:
        return np.concatenate(features)
    else:
        return None

def detect_drone_sound(sound_data, sr):
    features = extract_features(sound_data, sr)
    dmatrix = xgb.DMatrix(features.reshape(1, -1))
    probability = model.predict(dmatrix)[0]
    return probability >= 0.5, probability


def main():
    print("Начинаем прослушивание...")

    recorded_sounds = []

    # # Обработка сигнала SIGINT (Ctrl+C)
    # def signal_handler(sig, frame):
    #     print("\nПрерывание программы.")
    #     # Сохраняем записанные звуки
    #     for idx, sound in enumerate(recorded_sounds):
    #         sf.write(f"sound_{idx}.wav", sound, RATE, format='wav', subtype='PCM_16')
    #     exit(0)
    #
    # signal.signal(signal.SIGINT, signal_handler)

    CHUNK_SIZE = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    while True:
        try:
            sound_data = record_sound(min_duration=0.5, sr=RATE, frames_per_buffer=CHUNK_SIZE)
            recorded_sounds.append(sound_data)
            sr = RATE
            is_drone, probability = detect_drone_sound(sound_data, sr)
            if is_drone:
                print(f"Обнаружен звук дрона! Вероятность: {probability:.2f}")
            else:
                print("Звука дрона не обнаружено.")
        except KeyboardInterrupt:
            print("\nПрерывание программы.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()

























