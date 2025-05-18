import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io.wavfile import write

def adquisicion_y_reproduccion(duracion, fs=44100, dispositivo_entrada=None, dispositivo_salida=None):
    """
    Reproduce y graba audio simultáneamente con verificación de simultaneidad.
    
    Parámetros:
    -----------
    duracion : float
        Duración en segundos de la grabación/reproducción
    fs : int
        Frecuencia de muestreo (Hz)
    dispositivo_entrada : int/str
        ID del dispositivo de entrada
    dispositivo_salida : int/str
        ID del dispositivo de salida
    
    Retorna:
    --------
    señal_grabada : numpy.ndarray
        Señal grabada con metadatos de tiempo
    latencia : float
        Latencia medida en milisegundos
    """
    
    # 1. Configuración inicial
    print("\n=== Dispositivos disponibles ===")
    print(sd.query_devices())
    
    # 2. Generar señal de prueba (tono + pulso de inicio)
    muestras_totales = int(fs * duracion)
    t = np.linspace(0, duracion, muestras_totales, False)
    tono = 0.5 * np.sin(2 * np.pi * 440 * t)
    pulso_inicio = np.zeros_like(t)
    pulso_inicio[:100] = 1.0  # Pulso de 100 muestras al inicio
    señal_reproducida = tono + pulso_inicio
    
    # 3. Buffer para grabación
    señal_grabada = np.zeros((muestras_totales, 1), dtype='float32')
    
    # 4. Callback para procesamiento en tiempo real
    posicion = 0
    tiempo_inicio = None
    
    def callback(indata, outdata, frames, tiempo, status):
        nonlocal posicion, tiempo_inicio
        if tiempo_inicio is None:
            tiempo_inicio = time.time()
        
        # Manejo seguro del final de la señal
        disponible = len(señal_reproducida) - posicion
        if disponible <= 0:
            raise sd.CallbackStop
        frames_a_usar = min(frames, disponible)
        
        # Procesamiento de audio
        outdata[:frames_a_usar] = señal_reproducida[posicion:posicion+frames_a_usar].reshape(-1,1)
        outdata[frames_a_usar:] = 0  # Silencio para el resto del buffer
        señal_grabada[posicion:posicion+frames_a_usar] = indata[:frames_a_usar]
        posicion += frames_a_usar
        
        # Detener cuando llegamos al final
        if posicion >= len(señal_reproducida):
            raise sd.CallbackStop
    
    # 5. Ejecutar stream
    with sd.Stream(
        device=(dispositivo_entrada, dispositivo_salida),
        samplerate=fs,
        channels=1,
        dtype='float32',
        callback=callback,
        blocksize=1024
    ):
        print(f"\nIniciando prueba ({duracion}s)...")
        start_time = time.time()
        while posicion < len(señal_reproducida) and (time.time() - start_time) < duracion + 5:  # Timeout de seguridad
            sd.sleep(100)
    
    # 6. Calcular latencia
    latencia = (time.time() - tiempo_inicio - duracion) * 1000  # ms
    
    # 7. Análisis de simultaneidad
    def detectar_pulso(señal, umbral=0.3):
        detecciones = np.where(señal > umbral)[0]
        return detecciones[0] if len(detecciones) > 0 else 0
    
    inicio_teorico = 0
    inicio_real = detectar_pulso(señal_grabada)/fs
    desfase = (inicio_real - inicio_teorico) * 1000  # ms
    
    # 8. Visualización
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(señal_reproducida[:1000], label='Señal reproducida')
    plt.title('Señal de prueba (primeros 1000 puntos)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(señal_grabada[:1000], label='Señal grabada')
    plt.title(f'Señal grabada (Latencia total: {latencia:.2f} ms | Desfase: {desfase:.2f} ms)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 9. Guardar grabación
    write('grabacion_simultanea.wav', fs, señal_grabada)
    
    print("\n=== Resultados ===")
    print(f"Latencia total del sistema: {latencia:.2f} ms")
    print(f"Desfase inicio grabación: {desfase:.2f} ms")
    
    return señal_grabada, latencia

# Ejemplo de uso con dispositivos específicos para macOS
if __name__ == "__main__":
    # Configuración recomendada para macOS
    señal, latencia = adquisicion_y_reproduccion(
        duracion=3,
        fs=44100,
        dispositivo_entrada=1,  # BlackHole como entrada
        dispositivo_salida=1     # BlackHole como salida
    )