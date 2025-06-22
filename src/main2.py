import numpy as np
import soundfile as sf
from utils.irsint import irsint
from utils.ruido_rosa import ruido_rosa
from utils.hilbert import hilbert_transform
from utils.schroeder import schroeder
from utils.lognorm import lognorm
from utils.grafico import grafico1
from utils.grafico import grafico2
from utils.filter import filtro
from utils.moving_average import moving_average
from utils.least_sqrs import regresion_lineal_iso3382
from utils.ir import ir
from utils.procesamiento import param_edt, param_c80, param_d50
import matplotlib.pyplot as plt

t60 = {31.25:7.54,
       62.5:8.14,
       125:7.85,
       250:8.29,
       500:8.4,
       1000:7.71,
       2000:6.03,
       4000:4.03,
       8000:2
       }

# Sintetizo la IR del recinto
ir = irsint(t60,44100,True)
fs = 44100
#prepend_samples = int(fs*0.05) 
#ir = np.concatenate([np.zeros(prepend_samples),ir])

# Calculamos la energía de la IR sintetizada
ir_power = np.mean(ir**2)
SNR_dB = 60
noise_power = ir_power / (10**(SNR_dB / 10))

# Generamos el ruido rosa
ruido = ruido_rosa(9)

# Escalamos el ruido rosa a la potencia deseada
ruido_power = np.mean(ruido**2)
factor = np.sqrt(noise_power / ruido_power)
ruido_escalado = ruido * factor

# Extendemos IR para que dure lo mismo que el ruido
extention = np.zeros(len(ruido) - len(ir))
ir = np.concatenate([ir,extention])

# Señal con ruido de fondo
ir_ruido = ir + ruido_escalado

filtro = filtro(ir_ruido,44100,'tercio')
ir_ruido = filtro[19]

#ir_ruido, fs = sf.read('convolucion_FI_con_Sinesweepaula1000hz.wav')
#ir_ruido, fs = sf.read('ir.wav')

# Filtro la envolvente
ma = moving_average(ir_ruido,481)

# Se calcula Schroeder para la envolvente
sch = schroeder(ma)  # En escala logarítmica

# Paso las señales a escala logarítmica
ir_log = lognorm(ir_ruido)
t = np.linspace(0,len(sch) / fs,len(sch))
ma_log = lognorm(ma)

# Grafico
grafico2(ir_log,sch,"IR","Schroeder","T60")

m,b,y = regresion_lineal_iso3382(t,sch)

edt = m
t60 = param_edt(m,b)
#t60 = edt * 6
c50 = param_c80(ir_ruido)
c80 = param_c80(ir_ruido,t=80)
d50 = param_d50(ir_ruido)

print(f'T60: {t60:.2f}s\nEDT: {edt:.2f}s\nC50: {c50:.2f}dB\nC80: {c80:.2f}dB\nD50: {d50:.2f}\n')
