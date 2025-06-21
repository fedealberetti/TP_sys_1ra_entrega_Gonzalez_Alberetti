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

def sintetizar_respuesta_impulso(t60_por_banda, fs, duracion, frecuencias_centrales):
    """
    Sintetiza una respuesta al impulso para bandas específicas
    
    Parámetros:
    t60_por_banda: lista de tiempos T60 por banda (segundos)
    fs: frecuencia de muestreo (Hz)
    duracion: duración total de la RI (segundos)
    frecuencias_centrales: lista de frecuencias centrales (Hz)
    
    Retorna:
    t: vector de tiempo
    senal: respuesta al impulso normalizada
    """
    # Validar entrada
    if len(frecuencias_centrales) != len(t60_por_banda):
        raise ValueError("El número de bandas y valores T60 debe coincidir")
    
    t = np.arange(0, duracion, 1/fs)  # Vector de tiempo
    senal_total = np.zeros_like(t)     # Inicializar señal de salida
    
    # Generar componentes para cada banda
    for i, fc in enumerate(frecuencias_centrales):
        tau_i = -np.log(10**(-3)) / t60_por_banda[i]  # Constante de decaimiento
        componente = np.exp(-tau_i * t) * np.cos(2 * np.pi * fc * t)
        senal_total += componente
    
    return t, normalizar_RI(senal_total)



def funcRI(fi, sine):
    datosfi, fs = sf.read(fi)  # Cargar el archivo de respuesta al impulso
    datossine, fs = sf.read(sine)  # Cargar el archivo de señal de entrada (sine sweep)
    # Realizar la convolución
    RI = signal.fftconvolve(datossine, datosfi, mode='full')
    # Normalizar la señal resultante y grabarla como un archivo WAV   
    return RI




def filtronorma(senial, fs, tipo='octava', order=4):
    audiodata, fs = sf.read(senial)
    """
    Aplica filtros de octava o tercio de octava según norma IEC 61260.
    
    Parámetros:
        audiodata (array): Señal de audio.
        fs (float): Frecuencia de muestreo (Hz).
        band_type (str): 'octave' o 'third-octave'.
        order (int): Orden del filtro (recomendado: 4).
        zero_phase (bool): Si True, usa filtrado sin fase (filtfilt).
    
    Retorna:
        dict: {frecuencia_central: señal_filtrada}
    """
    # Definir frecuencias centrales según IEC 61260
    if tipo == 'octava':
        centers = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        G = 1.0  # Ancho de banda para octava
    elif tipo == 'third-octave':
        centers = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
                   250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
                   3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        G = 1.0 / 3.0  # Ancho de banda para tercio
    else:
        raise ValueError("band_type debe ser 'octave' o 'third-octave'")

    factor = 2 ** (G / 2)  # Cálculo del factor de banda
    nyquist = fs / 2
    filtroporbandas = {}

    for fc in centers:
        # Calcular frecuencias de corte
        lower = fc / factor
        upper = fc * factor
        
        # Saltar bandas que exceden Nyquist
        if upper > nyquist:
            continue
        
        # Diseñar filtro Butterworth en formato SOS
        sos = signal.iirfilter(
            N=order, 
            Wn=[lower, upper], 
            btype='band', 
            analog=False, 
            ftype='butter', 
            fs=fs, 
            output='sos'
        )
        
 
        filtered_data = signal.sosfilt(sos, audiodata)
        
        filtroporbandas[fc] = filtered_data

    return filtroporbandas

def ir_a_log(archivo):
    fs=44100  # Frecuencia de muestreo por defecto, se puede ajustar según el archivo
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
    array, fs = sf.read(archivo)
    audiodata = np.clip(array, 1e-12, None)
    # Se obtiene el valor máximo absoluto para la normalización
    A_max = np.max(np.abs(audiodata))

    # Evitar división por cero: si A_max es 0, se devuelve un array con -inf (logaritmo indefinido)
    if A_max == 0:
        return np.full_like(audiodata, -np.inf)

    # Se aplica la fórmula de conversión a escala logarítmica
    irlog = 20 * np.log10(np.abs(audiodata) / A_max)


    return irlog


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
    return analitica
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
    tiempoi = time.start()
    # El kernel del filtro es un vector de tamaño L con valores 1/L
    kernel = np.ones(L) / L
    # La opción 'same' asegura que la salida tenga el mismo tamaño que la señal de entrada
    y = np.convolve(senal, kernel, mode='same')
    tiempof = time.end()
    print(f"Tiempo de ejecución: {tiempof - tiempoi:.6f} segundos") 
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
def schroeder_integral(ir, dt=1.0):
    """
    Calcula la integral de Schroeder a partir de una respuesta al impulso (ir).
    
    Parámetros:
      ir: np.array
          Respuesta al impulso.
      dt: float, opcional
          Intervalo de muestreo (tiempo entre muestras). Por defecto 1.0.
    
    Retorna:
      energy_decay: np.array
          Curva de energía integrada (acumulada hacia adelante en el tiempo).
      energy_decay_db: np.array
          Versión en decibelios de la energía integrada, normalizada a la energía inicial.
    """
    # 1. Calcular la energía instantánea (la respuesta al impulso elevada al cuadrado)
    squared_ir = np.square(ir)
    
    # 2. Calcular la suma acumulada hacia atrás (integración inversa)
    # Se voltea el vector, se realiza la suma acumulada y se vuelve a girar.
    energy_decay = np.flip(np.cumsum(np.flip(squared_ir))) * dt
    
    # 3. Normalizar la curva: se divide por el valor máximo (energía total)
    energy_decay_norm = energy_decay / np.max(energy_decay)
    
    # 4. Convertir a decibelios; se añade un pequeño valor para evitar log(0)
    energy_decay_db = 10 * np.log10(energy_decay_norm + 1e-12)
    
    return energy_decay, energy_decay_db

def lundeby_extremo_integral(ir, t, margin=10, perc_tail=10, offset=5, max_iter=20, tol=1e-3):
    
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
    
    Parámetros:
      ir       : np.array
                 Respuesta al impulso.
      t        : np.array
                 Vector de tiempo correspondiente.
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
      t_intersect     : float
                        Tiempo en el que la regresión se cruza con el nivel de ruido.
      slope           : float
                        Pendiente de la regresión.
      intercept       : float
                        Intercepto de la regresión.
      noise_level     : float
                        Nivel de ruido final (en dB).
      energy_decay_db : np.array
                        Curva de decaimiento en dB (resultado de la integral de Schroeder).
    """
    dt = t[1] - t[0]
    _, energy_decay_db = schroeder_integral(ir, dt=dt)
    
    # (Paso 2) Estimar el nivel de ruido a partir del final (último perc_tail% de la curva)
    num_tail = max(int(len(energy_decay_db) * perc_tail / 100), 1)
    noise_level = np.mean(energy_decay_db[-num_tail:])
    
    t_intersect_old = None
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
        
        # Se podrían implementar más subdivisiones o calcular nuevos RMS locales (pasos 5–6 adicionales)
        # según se requiera alcanzar mayor precisión.
    
    return t_intersect, slope, intercept, noise_level, energy_decay_db

# Ejemplo de uso:

# Generamos una respuesta al impulso simulada: decaimiento exponencial con un poco de ruido.
# t = np.linspace(0, 3, 3000)  # 3 segundos con 3000 muestras
# ir = np.exp(-2*t) + 0.05 * np.random.randn(len(t))

# Aplicamos el método Lundeby para estimar el extremo superior de la integral
# t_int, slope, intercept, noise_level, energy_decay_db = lundeby_extremo_integral(
#     ir, t, margin=10, perc_tail=10, offset=5, max_iter=20, tol=1e-3
# )

# print("Tiempo de intersección (extremo superior):", t_int)
# print("Pendiente de la regresión:", slope)
# print("Intercepto:", intercept)
# print("Nivel de ruido estimado (dB):", noise_level)

# Se grafica la curva de decaimiento y el punto de intersección
# plt.figure(figsize=(10, 6))
# plt.plot(t, energy_decay_db, label="Curva de decaimiento (dB)")
# plt.axhline(y=noise_level, color='gray', linestyle='--', label="Nivel de ruido estimado")
# plt.axvline(x=t_int, color='red', linestyle='--', label=f"t_intersect = {t_int:.3f} s")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Energía (dB)")
# plt.title("Integral de Schroeder con límite superior según Lundeby")
# plt.legend()
# plt.show()