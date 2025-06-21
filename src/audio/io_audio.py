from scipy.io.wavfile import write
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


from ipywidgets import Button
from tkinter import Tk, filedialog
from IPython.display import clear_output, display
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np


def guardar_wav(nombre_archivo, señal, fs=44100):
    """
    Guarda la señal como archivo WAV
    
    Args:
        nombre_archivo (str): Ruta del archivo de salida
        señal (np.array): Señal de audio
        fs (int): Frecuencia de muestreo
    """
    
    # Guardar archivo WAV
    
    write(nombre_archivo, fs, (señal * 32767).astype(np.int16))
    
    #write(nombre_archivo, fs, señal)

def reproducir_audio(señal, fs=44100):
    """Reproduce la señal usando sounddevice"""
    import sounddevice as sd  # Import local para mejor portabilidad
    sd.play(señal, fs)
    sd.wait()

def reproducir_y_grabar(senal, fs=44100):
    """
    Reproduce una señal y graba al mismo tiempo.
    
    :param senal: Señal a reproducir (array NumPy)
    :param fs: Frecuencia de muestreo
    :return: Señal grabada
    """
    duracion = len(senal) / fs
    print(f"Reproduciendo y grabando por {duracion:.2f} segundos...")

    # Reproducir y grabar en simultáneo
    grabado = sd.playrec(senal, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    print("Grabación terminada.")
    return grabado.flatten()


def grabador(T,wav=False,fs=44100):
    '''
    Reproduce un sine-sweep y registra el audio tomado por el dispositivo 
    seleccionado por el usuario en el tiempo elegido. El registro
    se guarda en el archivo 'grabacion.wav'

    Nota: si 'grabacion.wav' existe, este será sobreescrito.

    Parámetros
    ----------
    fs: int
        Frecuencia de muestreo a la que se quiere grabar.
    return: Numpy Array
        Devuelve el array correspondiente a la grabación.
    '''
    ## Selección del dispositivo de audio
    sentinela = ""
    while sentinela != "s":
        # Imprime los dispositivos disponibles
        print(f"\n{sd.query_devices()}")

        # Ingreso de los dispositivos i/o del usuario
        entrada = int(input("\nSeleccione el dispositivo de entrada: "))
        salida = int(input("Seleccione el dispositivo de salida: "))
        sd.default.device = (entrada,salida)  # type: ignore

        # Se imprimen los dispositivos seleccionados
        print(f"\n{sd.query_devices()}")
        
        # Confirmación de los datos
        sentinela = str(input("\n¿Desea guardar los cambios? [S/n]: ")).lower()

    ## Grabador y reproductor

    # Carga del audio del sweep
    sweep, fs = sf.read('sweep.wav')
    t = T*fs

    # Extendemos el sweep según el tiempo que decida el usuario
    extend = np.zeros(t - len(sweep))
    sweep = np.concatenate([sweep,extend])

    # Grabación y reproducción
    print("Grabando...")
    grabacion = sd.playrec(sweep,channels=1)
    sd.wait()

    # Se guarda la grabación en un archivo 'grabacion.wav'
    if wav == True:
        sf.write('grabacion.wav',grabacion,fs)

    return grabacion, fs

def medir_latencia():
    """
    Estima la latencia del sistema de audio usando una señal de impulso.

    :return: Latencia estimada en segundos y en muestras
    :rtype: tuple[float, int]
    """
    fs = 44100
    dur = 1.0  # duración de 1 segundo

    # Crear señal de prueba: impulso en la muestra 100
    senal = np.zeros(int(fs * dur))
    senal[100] = 1.0

    # Reproducir y grabar usando función ya definida
    grabado = reproducir_y_grabar(senal, fs)

    # Aplanar por si quedó con forma (N,1)
    grabado = grabado.flatten()

    # Calcular correlación cruzada
    corr = correlate(grabado, senal)
    delay_samples = np.argmax(corr) - len(senal) + 1
    latencia_segundos = delay_samples / fs

    # Mostrar resultado
    print(f"Latencia estimada: {latencia_segundos:.4f} segundos ({delay_samples} muestras)")

    # Graficar
    t = np.linspace(0, dur, len(senal))
    plt.figure(figsize=(12, 4))
    plt.plot(t, senal, label="Señal emitida")
    plt.plot(t, grabado, label="Señal grabada")
    plt.title(f"Latencia estimada: {latencia_segundos*1000:.2f} ms")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Escuchar señal grabada
    reproducir_audio(grabado, fs)

    return latencia_segundos, delay_samples

# Lista global para almacenar los archivos seleccionados
files = []

def select_and_load_files(b):
    clear_output()  # Limpia la salida en Jupyter
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    root.call('wm', 'attributes', '.', '-topmost', True)  # Levanta el diálogo al frente

    # Permite seleccionar múltiples archivos WAV
    selected_files = filedialog.askopenfilenames(
        title="Seleccione archivos WAV",
        filetypes=[("Archivos WAV", ".wav"), ("Todos los archivos", ".*")]
    )
    
    # Agregar los archivos seleccionados a la lista global
    files.extend(selected_files)
    
    # Mostrar la lista de archivos seleccionados
    print("Archivos seleccionados:")
    for archivo in files:
        print(f"- {archivo}")
    
    # Procesar cada archivo WAV seleccionado
    for archivo in files:
        try:
            # Cargar el archivo WAV
            audio_data, sample_rate = sf.read(archivo)
            
            # Mostrar información del archivo
            print(f"\nArchivo: {archivo}")
            print(f"Frecuencia de muestreo: {sample_rate} Hz")
            print(f"Duración: {len(audio_data) / sample_rate:.2f} segundos")
            
            # Graficar la señal de audio
            time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
            plt.figure(figsize=(10, 4))
            plt.plot(time, audio_data, color='blue')
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud")
            plt.title(f"Señal de: {archivo.split('/')[-1]}")
            plt.grid(True)
            plt.show()
        
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")

# Crear el botón interactivo
fileselect = Button(description="Seleccione archivos WAV")
fileselect.on_click(select_and_load_files)

# Mostrar el botón en la celda de Jupyter
display(fileselect)

# La variable 'files' contiene la lista de archivos seleccionados
files

select_and_load_files(fileselect)  # Llamar a la función para cargar archivos al inicio
