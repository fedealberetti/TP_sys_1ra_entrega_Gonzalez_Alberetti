import matplotlib.pyplot as plt
from utils.espectro import cargar_espectro  # type: ignore # Importamos la función


def graficar_espectro_inv():
    """
    Grafica el espectro de frecuencia usando los datos cargados desde espectro.py.
    """
    frecuencias, nivel_db = cargar_espectro()  # Obtenemos los datos
    
    # Configuración del gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(frecuencias, nivel_db, color='blue', linewidth=1, label='Espectro Audacity')
    plt.title("Espectro de Frecuencia")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Nivel (dB)")
    plt.xscale('log')  # Escala logarítmica para el eje de frecuencia
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


