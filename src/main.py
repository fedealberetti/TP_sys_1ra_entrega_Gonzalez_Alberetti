from audio.generadores import generar_ruido_rosa
from audio.io_audio import guardar_wav, reproducir_audio

def main():
    # Generar 5 segundos de ruido rosa
    t=5
    señal = generar_ruido_rosa(t)
    
    # Guardar archivo
    guardar_wav("ruido_rosa.wav", señal)
    
    # Reproducir audio
    print("Reproduciendo ruido rosa durante" ,t, "segundos...")
    reproducir_audio(señal)

if __name__ == "__main__":
    main()