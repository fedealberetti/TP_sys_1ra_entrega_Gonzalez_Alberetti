import numpy as np
import pandas as pd
from scipy import signal
from generadores import *
import soundfile as sf
import matplotlib.pyplot as plt
from io_audio import guardar_wav, reproducir_audio
from generadores import normalizar_senial


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

def ir(grabacion,filtro_inverso,fs=44100,norm=True):
    '''
    Obtiene la respuesta al impulso de un recinto ingresando la grabación del 
    sine-sweep en el lugar y su respectivo filtro inverso.

    Parámetros
    ----------
    grabacion: ruta .wav
        Registro del sine-sweep en el recinto.
    filtro_inverso: ruta .wav
        Filtro inverso del sine-sweep grabado en el recinto.
    norm: Bool
        Si es True, la señal del return será normalizada.
    return: Numpy Array
        Devuelve la respuesta al impulso del recinto.
    '''
    # Se cargan los archivos de audio del sweep en el recinto y el filtro inverso
    k, fs = sf.read('filtro_inverso.wav')
    y, fs = sf.read('grabacion.wav')

    #k = filtro_inverso
    #y = grabacion

    #if y.ndim > 1:
    #    y = y.flatten()

    # Se aplica la transformada de Fourier en ámbas señales
    n = len(y) + len(k) - 1
    K = np.fft.fft(k,n=n)
    Y = np.fft.fft(y,n=n)

    # Se multiplican las señales
    H = Y * K

    # Tranformada inversa de Fourier de h
    h = np.fft.ifft(H)
    h = np.real(h)

    # Se obtiene el valor máximo de h
    h_max = np.argmax(abs(h))

    # Recortá una ventana desde el pico, por ejemplo 3 segundos
    duracion_segundos = 30
    muestras = int(fs * duracion_segundos)
    inicio = h_max 
    fin = h_max + muestras

    # Asegurate de no salirte del array
    h = h[inicio:fin] if fin < len(h) else h[inicio:]

    # Normalización (opcional)
    if norm == True:
        h /= np.max(np.abs(h))
    
    # Normalización y formato float32
    h_audio = h / np.max(np.abs(h))
    h_audio = h_audio.astype(np.float32)

    sf.write('impulse_response.wav',h_audio,fs)

    return h

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

def param_edt(m,b):
    '''
    Calcula el tiempo de decaimiento temprano de la respuesta al impulso de un recinto.

    Parámetros:
    -----------
    m: float
        Pendiente de la recta obtenida por regresión lineal.
    b: float
        Ordenada al origen de la recta obtenida por regresión lineal.
    return: float
        Devuelve el valor del EDT.
    '''
    dB_ini = 0     # Inicio del decaimiento (nivel máximo)
    dB_fin = -10   # Nivel a alcanzar

    t0 = (dB_ini - b) / m
    t10 = (dB_fin - b) / m

    edt = 6 * (t10 - t0)

    return edt

def param_t60(m,b,metodo='t30'):
    """
    Calcula T60 extrapolado a partir de una recta ya ajustada.

    Parámetros:
    -----------
    m: float
        Pendiente de la recta obtenida por regresión lineal.
    b: float
        Ordenada al origen de la recta obtenida por regresión lineal.
    metodo: string
        Metodo de transposición para el T60: 't10','t20','t30' (por defecto).
    return: float
        Devuelve el valor del T60.
    """
    if metodo == 't10':
        dB_ini, dB_fin = -5, -15
    elif metodo == 't20':
        dB_ini, dB_fin = -5, -25
    elif metodo == 't30':
        dB_ini, dB_fin = -5, -35
    else:
        raise ValueError("Método inválido. Usa 't10', 't20' o 't30'.")

    t_ini = (dB_ini - b) / m
    t_fin = (dB_fin - b) / m
    tramo = t_fin - t_ini
    multiplicador = 60 / abs(dB_fin - dB_ini)

    t60 = tramo * multiplicador

    return t60

def param_c80(signal,t=50,fs=44100):
    '''
    Calcula el C80 de la respuesta al impulso ingresada con la posibilidad
    de que el usuario elija el tiempo que quiere tomar según el recinto.

    Parámetros:
    -----------
    signal: Numpy Array
        Corresponde a la respuesta al impulso del recinto.
    t: int
        Valor en milisegundos que requiera el usuario (50ms por defecto).
    fs: int
        Frecuencia de muestreo correspondiente a la respuesta al impulso.
    return: float
        Devuelve el valor del C80.
    '''
    # C80 o "claridad"

    # Se pasa de milisegundos a segundos
    t = t / 1000

    # Se recorta la IR hasta el extremo superior
    pre80 = signal[:int(fs*t)]
    post80 = signal[int(fs*t):]

    # Calcula la energía del primer tramo
    energia_pre80 = np.sum(pre80**2)

    # Calcula la energía del segundo tramo
    energia_post80 = np.sum(post80**2)

    if energia_post80 == 0:
        return 0

    # Se calcula el C80
    c80 = 10 * np.log10(energia_pre80 / energia_post80)

    return c80

def param_d50(signal,fs=44100):
    '''
    Calcula el D50 de la respuesta al impulso ingresada.

    Parámetros:
    -----------
    signal: Numpy Array
        Corresponde a la respuesta al impulso del recinto
    fs: int
        Frecuencia de muestreo correspondiente a la respuesta al impulso.
    return: floar
        Devuelve el valor del D50.
    '''
    ## D50 o "definición"

    t = 0.05

    # Se recorta la IR hasta el extremo superior
    pre50 = signal[:int(fs*t)]

    # Calcula la energía del primer tramo
    energia_pre50 = np.sum(pre50**2)

    # Calcula la energía total de la señal
    energia_signal= np.sum(signal**2)

    if energia_signal == 0:
        return 0

    # Se calcula el C80
    d50 = energia_pre50 / energia_signal

    return d50
