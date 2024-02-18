import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from transformers import pipeline

from time import sleep
from sys import platform
from pydub import AudioSegment
import simpleaudio as sa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    
    parser.add_argument("--file", default=None, type=str,
                        help="audiofile to transcribe")

    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    
    scam_model = pipeline(task="text-classification", model="phishbot/ScamLLM", top_k=None)
    transcription = ['']
    
    
    audio = AudioSegment.from_mp3(args.file)
    chunk_length_ms = 2 * 1000  # 30 seconds in milliseconds

    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    print("Model loaded.")
    
    now =  datetime.utcnow()
    for i, chunk in enumerate(chunks): 
        print(i)
        source = chunk.export('temp_chunk.wav', format="wav")
        with sr.AudioFile('temp_chunk.wav') as source:
        # play_obj = sa.play_buffer(source.read(), num_channels=1, bytes_per_sample=2, sample_rate=44100)
            recorder.adjust_for_ambient_noise(source)
            audio_data = recorder.record(source)
            try:
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                if text and phrase_complete:
                    transcription.append(text)
                elif text:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                is_scam = scam_model(".".join(transcription)) 
                for dictionary in is_scam[0]:
                    if dictionary['label'] == "LABEL_1": 
                        print("Is scam: ", dictionary['score'])
                        if dictionary['score'] > 0.6:
                            print("Scam detected")
                sleep(0.25)
            except sr.UnknownValueError:
                print(f"Chunk {i} could not be understood.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                
    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
