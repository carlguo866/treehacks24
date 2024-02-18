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

import streamlit as st
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
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    


    # The last time a recording was retrieved from the queue.
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # if 'linux' in platform:
    mic_name = "MacBook Pro Microphone"
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
        return
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break

    if 'clicked' not in st.session_state:
        st.session_state['clicked'] = False

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    
    scam_model = pipeline(task="text-classification", model="phishbot/ScamLLM", top_k=None)
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)


    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.")
    print(source.device_index)
    
    st.markdown("# Geras")
    st.markdown("Geras is a AI-powered tool that prevents the elderly from being scammed. We use OpenAI's Whisper API to transcribe the audio and another text classification LLM to detect if the person is being scammed.")

    st.markdown("Let's give it a try. Please click the button below to record.")

    # print_info = False

    refreshable_box = st.empty()
    def record_button():
        st.write("Recording... Text below:")
        st.session_state.clicked = True

    break_loop = False
    def stop_button():
        st.write("Stopped recording.")
        break_loop = True
        st.session_state.clicked = False
        refreshable_box.empty()
        return
    
    st.button('Record', on_click=record_button) 
    st.button("Stop", on_click=stop_button)
    while True:
        try:
            
            if break_loop: 
                break
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if st.session_state.clicked:
                    if text and phrase_complete:
                        transcription.append(text)
                    elif text:
                        transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                with refreshable_box.container(): 
                    for line in transcription:
                        st.write(line)
                    # Flush stdout.
                    # print('', end='', flush=True)

                    # Infinite loops are bad for processors, must sleep.
                    # inputs = ".".join(transcription) 
                    # if len(inputs) > 520: 
                    #     inputs = inputs[len(inputs) - 520:]
                    is_scam = scam_model(".".join(transcription) ) 
                    for dictionary in is_scam[0]:
                        if dictionary['label'] == "LABEL_1": 
                            # if print_info:
                                st.write("Is scam: ", dictionary['score'])
                                if dictionary['score'] > 0.6:
                                    st.markdown(f"<h1 style='color: red;'>Scam detected</h1>", unsafe_allow_html=True)

                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()