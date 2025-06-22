import numpy as np
import pandas as pd
from scipy import signal
from audio.generadores import *
import soundfile as sf
import matplotlib.pyplot as plt
from audio.io_audio import *

def normalizar_RI(senal):
    """Normaliza la señal al rango [-1, 1]"""
    max_val = np.max(np.abs(senal))
    return senal / max_val if max_val > 0 else senal

import numpy as np

def sintetizar_respuesta_impulso(t60_por_banda, fs, duracion, frecuencias_centrales, SNR_dB=60):
    """
    Sintetiza una respuesta al impulso multibanda y agrega ruido rosa con fade out.

    Parámetros:
        t60_por_banda (list): Lista de T60 por banda (en segundos).
        fs (int): Frecuencia de muestreo (Hz).
        duracion (float): Duración total de la señal de salida (en segundos).
        frecuencias_centrales (list): Lista de frecuencias centrales (Hz).
        SNR_dB (float): Relación señal-ruido en dB (por defecto 60 dB).

    Retorna:
        tuple: (vector de tiempo, señal final normalizada)
    """
    if len(frecuencias_centrales) != len(t60_por_banda):
        raise ValueError("El número de frecuencias y valores T60 debe coincidir.")

    # Tiempo y señal base
    t = np.arange(0, duracion, 1/fs)
    senal_total = np.zeros_like(t)

    # Generar IR como suma de componentes de cada banda
    for i, fc in enumerate(frecuencias_centrales):
        tau = -np.log(10**(-3)) / t60_por_banda[i]
        componente = np.exp(-tau * t) * np.cos(2 * np.pi * fc * t)
        senal_total += componente

    # Insertar 0.05 s de silencio al inicio
    prepend_samples = int(0.05 * fs)
    ir_desplazada = np.concatenate([np.zeros(prepend_samples), senal_total])

    # Ajustar longitud final
    if len(ir_desplazada) < len(t):
        ir_extendida = np.concatenate([ir_desplazada, np.zeros(len(t) - len(ir_desplazada))])
    else:
        ir_extendida = ir_desplazada[:len(t)]

    # Calcular potencia de la IR
    ir_power = np.mean(ir_extendida**2)
    noise_power = ir_power / (10**(SNR_dB / 10))

    # Generar ruido rosa del mismo largo
    duracion_ruido = len(ir_extendida) / fs
    ruido, _ = generar_ruido_rosa(duracion_ruido, fs=fs)
    ruido = ruido[:len(ir_extendida)].astype(np.float64)  # ← Conversión explícita a float

    # Aplicar rampa de fade out (último 10% de la señal)
    n = len(ruido)
    fade_len = int(0.10 * n)
    ventana = np.ones(n)
    ventana[-fade_len:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))  # Hann invertida

    ruido *= ventana


    # Escalar ruido para respetar el SNR
    ruido_power = np.mean(ruido**2)
    factor_ruido = np.sqrt(noise_power / ruido_power)
    ruido_escalado = ruido * factor_ruido

    # Sumar ruido a la IR extendida
    senal_con_ruido = ir_extendida + ruido_escalado

    # Normalizar señal final
    senal_normalizada = normalizar_RI(senal_con_ruido)

    return t, senal_normalizada




def funcRI(fi, sine):
    datosfi, fs = sf.read(fi)  # Cargar el archivo de respuesta al impulso
    datossine, fs = sf.read(sine)  # Cargar el archivo de señal de entrada (sine sweep)
    # Realizar la convolución
    RI = signal.fftconvolve(datossine, datosfi, mode='full')
    # Normalizar la señal resultante y grabarla como un archivo WAV   
    return RI




def filtronorma(signal_in, fs, fc, tipo='octava'):
    """
    Aplica filtro bandpass Butterworth de octava o tercio de octava en la frecuencia fc
    según IEC 61260, sobre la señal de entrada.

    Parámetros:
        signal_in (np.array): Señal a filtrar (1D).
        fs (int): Frecuencia de muestreo.
        fc (float): Frecuencia central de la banda a filtrar.
        tipo (str): 'octava' o 'tercio' (por defecto 'octava').

    Retorna:
        np.array: Señal filtrada.
    """
    if tipo == 'octava':
        G = 1.0 / 2.0
    elif tipo == 'tercio':
        G = 1.0 / 6.0
    else:
        raise ValueError("El parámetro 'tipo' debe ser 'octava' o 'tercio'.")

    factor = 2 ** G
    nyquist = fs / 2

    lower = fc / factor
    upper = fc * factor

    if upper > nyquist:
        raise ValueError(f"La frecuencia superior {upper} Hz excede el límite de Nyquist ({nyquist} Hz).")

    # Orden dinámico del filtro (puede ajustarse)
    if fc <= 200:
        orden = 8
    elif fc < 1000:
        orden = 6
    else:
        orden = 4

    sos = signal.iirfilter(
        N=orden,
        Wn=[lower, upper],
        btype='band',
        analog=False,
        ftype='butter',
        fs=fs,
        output='sos'
    )

    filtrada = signal.sosfilt(sos, signal_in)
    return filtrada

def ir_a_log(signal, fs = 44100):
    
    """
    Convierte una respuesta al impulso (RI) a una escala logarítmica normalizada.
    
    La función aplica la siguiente fórmula:
        R(t) = 20 * log10( A(t) / A(t)_max )
    
    Parámetros:
        impulse_response : array-like
            Arreglo que contiene la respuesta al impulso.

    Retorna:
        np.array: Arreglo con la señal transformada a escala logarítmica (dB).
    """
    # Se asegura que impulse_response sea un arreglo de NumPy
     # Señal con valores absolutos
    signal_abs = np.abs(signal)

    # Valor máximo de la señal
    signal_max = np.max(np.abs(signal))

    # Si la energía de la señal es nula
    if signal_max == 0:
        return np.full_like(signal, -np.inf)

    # Cociente entre el valor absoluto y el máximo
    signal_norm = signal_abs / signal_max
    signal_norm_segura = np.clip(signal_norm, 1e-12, None)

    # Transformación a escala logarítmica
    r = 20 * np.log10(signal_norm_segura)


    return r
    return irlog


def filtro(banda='octava'):
    '''
    Genera filtros de cada frecuencia centra de la respuesta al impulso.
    Genera un archivo wav por cada frecuencia filtada.

    Nota: si los archivos wav ya existen, serán reemplazadas.

    Parámetros
    ----------
    ir: Ruta
        Ruta donde se encuentra la respuesta al impulso en formato wav.
    return: Lista
        Lista con los arrays correspondientes a los filtros por banda de octava.
    '''
    # Se ingresa la respuesta al impulso como archivo wav
    audio, fs = sf.read('ir.wav')

    # Lista donde se guardan las señales filtradas
    filtros = []
    frecuencias = []

    if banda == 'octava':
        # Bandas por octava
        frecuencias = [31.25,62.5,125,250,500,1000,2000,4000,8000]
        G = 1.0/2.0

    else:
        # Banda tercio por octava
        frecuencias = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
               250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
               3150, 4000, 5000, 6300, 8000, 10000, 12500]
        G = 1.0/6.0

    for fi in frecuencias:
        #Selección de octava - G = 1.0/2.0 / 1/3 de Octava - G=1.0/6.0
        fo = 0
        factor = np.power(2, G)
        centerFrequency_Hz = fi 

        #Calculo los extremos de la banda a partir de la frecuencia central
        lowerCutoffFrequency_Hz=centerFrequency_Hz/factor;
        upperCutoffFrequency_Hz=centerFrequency_Hz*factor;

        # El orden del filtro varía con la frecuencia central

        # Orden 8 para frecuencias menores a 200Hz
        if fi <= 200:
            fo = 8
        # Orden 6 para frecuencias entre 200Hz y 1kHz
        if fi > 200 and fi < 1000:
            fo = 6
        # Orden 4 para frecuencias mayores a 1kHz
        if fi >= 1000:
            fo = 4

        # Extraemos los coeficientes del filtro 
        b,a = signal.iirfilter(fo, [2*np.pi*lowerCutoffFrequency_Hz,2*np.pi*upperCutoffFrequency_Hz],
                                    rs=60, btype='band', analog=True,
                                    ftype='butter') 

        # para aplicar el filtro es más óptimo
        sos = signal.iirfilter(fo, [lowerCutoffFrequency_Hz,upperCutoffFrequency_Hz],
                                    rs=60, btype='band', analog=False,
                                    ftype='butter', fs=fs, output='sos') 

        w, h = signal.freqs(b,a)

        # Aplicando filtro al audio
        filt = signal.sosfilt(sos, audio)

        # Se guarda la señal filtrada en una lista
        filtros.append(filt)

        # Se genera un archivo .wav correspondiente a la señal filtrada en fi
        sf.write(f'filtro_{fi}Hz.wav',filt,fs)

    return filtros



def aplicar_transformada_hilbert(senal):
    """
    Aplica la transformada de Hilbert a la señal de entrada.
    
    Parámetros:
      senal: np.array
          Señal de entrada en el dominio temporal.
    
    Retorna:
      analitica: np.array
          Señal analítica, donde la parte real es la señal original y 
          la parte imaginaria es la transformada de Hilbert.
    """
    # Calcular la señal analítica utilizando la función hilbert de SciPy.
    analitica = signal.hilbert(senal)
    return normalizar_RI(np.abs(analitica))
def promedio_movil_convolucion(senal, L):
    """
    Aplica el promedio móvil a la señal x usando convolución.
    
    Parámetros:
      senal: np.array
         Señal de entrada.
      L: int
         Tamaño de la ventana del promedio.
    
    Retorna:
      y: np.array
         Señal filtrada.
    """
    # tiempoi = time.start()
    # El kernel del filtro es un vector de tamaño L con valores 1/L
    kernel = np.ones(L) / L
    # La opción 'same' asegura que la salida tenga el mismo tamaño que la señal de entrada
    y = np.convolve(senal, kernel, mode='same')
    #tiempof = time.end()
    #print(f"Tiempo de ejecución: {tiempof - tiempoi:.6f} segundos") 
    return y
def promedio_movil_bucle(x, L):
    """
    Aplica el promedio móvil a la señal x usando bucles.
    
    Parámetros:
      x: np.array
         Señal de entrada.
      L: int
         Tamaño de la ventana del promedio.
    
    Retorna:
      y: np.array
         Señal filtrada.
    """
    tiempoi = time.start()
    N = len(x)
    y = np.zeros_like(x, dtype=float)
    
    for i in range(N):
        if i < L - 1:
            # Para índices iniciales, promediamos las muestras disponibles
            y[i] = np.sum(x[:i+1]) / (i+1)
        else:
            # Cuando hay suficientes muestras, usamos la ventana completa
            y[i] = np.sum(x[i-L+1:i+1]) / L
    tiempof = time.end()
    print(f"Tiempo de ejecución: {tiempof - tiempoi:.6f} segundos")
    # Normalizar la señal resultante
    return y

def schroeder(ir):
    """
    Calcula la curva de Schroeder (energía acumulada inversa) y la normaliza.
    
    Parámetros:
        ir (np.array): Respuesta al impulso
        
    Retorna:
        np.array: Curva de Schroeder en dB
    """
    energy = ir**2
    sch = np.cumsum(energy[::-1])[::-1]  # Suma acumulada inversa
    
    # Manejar caso de señal con energía cero
    if np.max(sch) <= 0:
        return np.zeros_like(sch)
    
    sch /= np.max(sch)  # Normalizar
    sch_dB = 10 * np.log10(sch + 1e-12)  # Evitar log(0)
    return sch_dB

def schroeder_integral(ir, dt=1.0):
    """
    Calcula la integral de Schroeder a partir de una respuesta al impulso (ir).
    
    Parámetros:
        ir: np.array - Respuesta al impulso
        dt: float - Intervalo de tiempo entre muestras (1/fs)
        
    Retorna:
        tuple: (energy_decay, energy_decay_db)
    """
    squared_ir = np.square(ir)
    
    # Calcular la suma acumulada hacia atrás
    energy_decay = np.flip(np.cumsum(np.flip(squared_ir))) * dt
    
    # Normalizar
    max_energy = np.max(energy_decay)
    if max_energy <= 0:
        return energy_decay, np.zeros_like(energy_decay)
    
    energy_decay_norm = energy_decay / max_energy
    energy_decay_db = 10 * np.log10(energy_decay_norm + 1e-12)
    
    return energy_decay, energy_decay_db

def lundeby(ir, fs, margin=10, perc_tail=10, offset=5, max_iter=20, tol=1e-3):
    """
    Estima el "extremo superior" de la integral de Schroeder usando el método Lundeby iterativo,
    siguiendo el siguiente esquema:
    
      1. Se calcula la curva de energía (integral de Schroeder) en dB.
      2. Se estima el nivel de ruido (promedio de los últimos 'perc_tail'% de la curva).
      3. Se selecciona la parte de la curva comprendida entre 0 dB (punto izquierdo) y
         (nivel de ruido + offset) dB (punto derecho, típicamente 5–10 dB por encima del ruido).
      4. Se realiza una regresión lineal sobre esa porción para obtener la pendiente y el intercepto.
      5. Se calcula el tiempo de intersección de la línea de regresión con el nivel de ruido.
      6. Se redefine una ventana local (mínimo 10% de la RI) a partir del punto de cruce (punto 7)
         para actualizar la estimación del ruido.
      7. Se repiten los pasos 4 a 6 hasta que la variación en t_intersect sea menor que 'tol'.
    
    Parámetros modificados:
      ir       : np.array
                 Respuesta al impulso.
      fs       : int
                 Frecuencia de muestreo (Hz).
      margin   : float
                 Margen en dB (por ejemplo, 10 dB) que se utiliza para definir el rango
                 de la regresión.
      perc_tail: float
                 Porcentaje de la cola de la señal que se usa para estimar el nivel de ruido (e.g., 10%).
      offset   : float
                 Offset en dB que se suma al nivel de ruido para definir el punto derecho de la regresión (5–10 dB).
      max_iter : int
                 Número máximo de iteraciones.
      tol      : float
                 Tolerancia para determinar la convergencia en el tiempo de intersección.
    
    Retorna:
      ir_truncated : np.array
                     Respuesta al impulso truncada en el punto de intersección con el ruido.
    """
    # Crear vector de tiempo basado en la longitud de la señal y fs
    t = np.arange(len(ir)) / fs
    dt = t[1] - t[0]  # Intervalo de tiempo entre muestras
    
    # Calcular la integral de Schroeder en dB
    _, energy_decay_db = schroeder_integral(ir, dt=dt)
    
    # (Paso 2) Estimar el nivel de ruido a partir del final (último perc_tail% de la curva)
    num_tail = max(int(len(energy_decay_db) * perc_tail / 100), 1)
    noise_level = np.mean(energy_decay_db[-num_tail:])
    
    t_intersect_old = None
    t_intersect = 0  # Inicializar con valor por defecto
    
    for iteration in range(max_iter):
        # (Paso 3) Se define el rango para la regresión:
        #      - Punto izquierdo: 0 dB.
        #      - Punto derecho: noise_level + offset (usualmente 5–10 dB por encima del ruido).
        upper_bound = 0  # 0 dB (punto inicial)
        lower_bound = noise_level + offset  # Umbral inferior para la selección
        indices = np.where((energy_decay_db <= upper_bound) & (energy_decay_db >= lower_bound))[0]
        
        if len(indices) < 2:
            print(f"Iteración {iteration}: No hay suficientes puntos para ajustar; se aborta el proceso.")
            break
        
        # (Paso 4) Ajuste lineal (regresión) sobre los puntos de la curva seleccionada.
        t_fit = t[indices]
        dB_fit = energy_decay_db[indices]
        slope, intercept = np.polyfit(t_fit, dB_fit, 1)
        
        # (Paso 5) Calcular el punto de cruce: se resuelve slope*t + intercept = noise_level.
        t_intersect = (noise_level - intercept) / slope
        
        # Comprobación de convergencia
        if t_intersect_old is not None and np.abs(t_intersect - t_intersect_old) < tol:
            break
        t_intersect_old = t_intersect
        
        # (Paso 7) Actualizar la estimación de ruido:
        # Se define una ventana local a partir del punto de cruce que abarque al menos el 10% de la RI.
        idx_start = np.searchsorted(t, t_intersect)
        idx_end = min(idx_start + max(int(0.1 * len(t)), 1), len(t))
        if idx_start < idx_end:
            noise_level = np.mean(energy_decay_db[idx_start:idx_end])
    
    # Truncar la señal en el punto de intersección
    idx_intersect = int(t_intersect * fs)
    return ir[:idx_intersect] if idx_intersect < len(ir) else ir
    
  

def regresion_lineal_iso3382(x, y):
    """
    Realiza una regresión lineal por mínimos cuadrados siguiendo las fórmulas
    típicamente encontradas en el Anexo C de ISO 3382:2008.

    Args:
        x_data (array-like): Datos de la variable independiente (e.g., tiempo).
        y_data (array-like): Datos de la variable dependiente (e.g., nivel de sonido en dB).

    Returns:
        tuple: Una tupla que contiene:
            - pendiente (float): La pendiente 'm' de la línea de regresión.
            - ordenada_origen (float): La ordenada al origen 'b' de la línea de regresión.
            - y_pred (numpy.ndarray): Los valores 'y' predichos por la línea de regresión.
    """
    #n = len(x_data)

    #if n == 0:
    #    raise ValueError("Los datos de entrada no pueden estar vacíos.")
    #if n != len(y_data):
    #    raise ValueError("x_data y y_data deben tener la misma longitud.")
    #if n < 2:
    #    raise ValueError("Se necesitan al menos 2 puntos para realizar una regresión lineal.")

    ## Convertir a arrays de NumPy para facilitar las operaciones
    #x = np.array(x_data)
    #y = np.array(y_data)

    n = len(x)

    # Calcular las sumatorias necesarias
    sum_xy = np.sum(x * y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x**2)

    # Calcular la pendiente (m)
    # m = (n * sum(xi*yi) - sum(xi) * sum(yi)) / (n * sum(xi^2) - (sum(xi))^2)
    numerador_m = n * sum_xy - sum_x * sum_y
    denominador_m = n * sum_x_squared - sum_x**2

    if denominador_m == 0:
        raise ValueError("No se puede calcular la pendiente (denominador es cero). Esto puede ocurrir si todos los valores de x son iguales.")

    pendiente = numerador_m / denominador_m

    # Calcular la ordenada al origen (b)
    # b = mean(y) - m * mean(x)
    media_x = np.mean(x)
    media_y = np.mean(y)

    ordenada_origen = media_y - pendiente * media_x

    # Calcular los valores y predichos
    y_pred = pendiente * x + ordenada_origen

    return pendiente, ordenada_origen, y_pred

def param_edt(signal, fs=44100):
    """
    Calcula el tiempo de decaimiento temprano (EDT) de la respuesta al impulso.
    
    Parámetros:
        signal: np.array - Respuesta al impulso
        fs: int - Frecuencia de muestreo (Hz)
        
    Retorna:
        float: Valor de EDT en segundos
    """
    # Calcular la curva de Schroeder
    sch_db = schroeder(signal)
    
    # Encontrar el rango para la regresión (primeros 10 dB de decaimiento)
    max_db = np.max(sch_db)
    start_idx = np.argmax(sch_db >= (max_db - 0.1))
    end_idx = np.argmax(sch_db <= (max_db - 10.0))
    
    # Ajustar índices si no se encuentra el punto final
    if end_idx == 0:
        end_idx = len(sch_db) - 1
    
    # Crear vector de tiempo
    t = np.arange(len(signal)) / fs
    
    # Extraer segmento para regresión
    x = t[start_idx:end_idx]
    y = sch_db[start_idx:end_idx]
    
    # Realizar regresión lineal
    if len(x) < 2:
        return 0.0  # No hay suficientes puntos para regresión
    
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calcular EDT
    return -60 / slope


# param_c80 
def param_c80(signal, fs=44100, t=80):
    t_sec = t / 1000.0  # Convertir a segundos
    n_samples = int(fs * t_sec)  # Convertir a entero
    # Asegurar que no exceda la longitud
    if n_samples > len(signal):
        n_samples = len(signal)
    pre80 = signal[:n_samples]
    post80 = signal[n_samples:]
    
    energia_pre80 = np.sum(pre80**2)
    energia_post80 = np.sum(post80**2)
    
    if energia_post80 == 0:
        return 0.0
    
    return 10 * np.log10(energia_pre80 / energia_post80)

# param_d50 
def param_d50(signal, fs=44100):
    t_sec = 0.05
    n_samples = int(fs * t_sec)  # Convertir a entero
    if n_samples > len(signal):
        n_samples = len(signal)
    pre50 = signal[:n_samples]
    
    energia_pre50 = np.sum(pre50**2)
    energia_total = np.sum(signal**2)
    
    if energia_total == 0:
        return 0.0
    
    return energia_pre50 / energia_total