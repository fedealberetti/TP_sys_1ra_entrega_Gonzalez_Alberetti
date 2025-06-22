from audio.generadores import *
from audio.io_audio import *
from audio.procesamiento import filtronorma, funcRI, ir_a_log, sintetizar_respuesta_impulso
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

from utils import espectro
from utils import espectroSine_sweep
from utils.graficofrecuencia import graficar_espectro, graficar_espectro_inv, graficar_espectro_sweep
from utils.graficotemporal import graficar_dominio_temporal

fs=44100

senialporbandas = filtronorma("convolucion_FI_con_Sinesweepaula.wav", fs,'octava', order=4)
