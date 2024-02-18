
import pyaudio
import wave
import numpy as np
from time import sleep, time
import torchaudio
import torch


import pandas as pd
import json
import math

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import streamlit as st
dropout_fn = nn.Dropout1d
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Number of audio channels
RATE = 16000              # Bit rate
TARGET_SAMPLE_RATE = 16000
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 15      # Record duration
NUM_SAMPLES = 16000 * 15  # Number of samples
WAVE_OUTPUT_FILENAME = "output.wav"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    
    
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y
    

class Janus(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=True,
    ):
        super().__init__()

        # Linear encoder
        self.encoder = nn.Linear(d_input, d_model)
        self.prenorm = prenorm

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.002))
            )
            self.norms.append(RMSNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        #print(x.shape)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    

def get_torchaudio_data(file_path):
    signal, sr = torchaudio.load(file_path)
    signal = signal.to(device)
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE).to(device)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    if signal.shape[1] > NUM_SAMPLES:
        signal = signal[:, :NUM_SAMPLES]
    elif signal.shape[1] < NUM_SAMPLES:
        padding = NUM_SAMPLES - signal.shape[1]
        signal = torch.cat([signal, torch.zeros(1, padding).to(device)], dim=1)
    signal = torchaudio.transforms.Spectrogram()(signal)
    
    return torch.squeeze(signal.permute(0, 2, 1)).to(device)
    
def main(): 
    audio = pyaudio.PyAudio()
    

    model_config = dict(
        d_model=256,
        n_layers=4,
        dropout=0.2,
        d_input=201, #input dim
        d_output=2, #num classes
        prenorm=True,
    )
    d_model = model_config['d_model']
    n_layers = model_config['n_layers']

    model = Janus(**model_config).to(device)
    model.load_state_dict(torch.load("Janus_256_4.pt", map_location=torch.device('cpu')))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")
    
    
    # Start recording
   



    
         
    st.markdown("# Geras")
    st.markdown("Geras is a AI-powered tool that prevents the elderly from being scammed. We use OpenAI's Whisper API to transcribe the audio and another text classification LLM to detect if the person is being scammed.")

    st.markdown("Let's give it a try. Please click the button below to record.")
    
    if 'clicked' not in st.session_state:
        st.session_state['clicked'] = False
    
    def record_button():
        st.write("Recording... Text below:")
        print("recording...")
        st.session_state.clicked = True 
        
    break_loop = False
    def stop_button():
        st.write("Stopped recording.")
        break_loop = True
        print_info = False
        st.session_state.clicked = False
        
        return
    
    st.button('Record', on_click=record_button) 
    
    st.button("Stop", on_click=stop_button)


    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = [] 
    time_start = time()
    og_time = time() 
    while True:
        try: 
            if break_loop:
                break
            
            if not st.session_state.clicked: 
                continue
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.append(data)
            if len(frames) >= RATE / CHUNK * RECORD_SECONDS: 
                frames = frames[1:]
                
            if time() - time_start > 1: 
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                data = get_torchaudio_data(WAVE_OUTPUT_FILENAME)
                data = data.unsqueeze(0)
                model.eval() 
                with torch.no_grad():
                    output = model(data)
                    score = output.log_softmax(dim=1).exp().argmax(dim=1)
                    if time() - og_time > 6:
                        st.write("Is scam: ", score)
                        if score > 0.5:
                            st.markdown(f"<h1 style='color: red;'>Scam detected</h1>", unsafe_allow_html=True)

                        
                time_start = time()
                    
                
        except KeyboardInterrupt:
            break
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    print("finished recording")
    # Stop recording

    

if __name__ == "__main__":
    main()
