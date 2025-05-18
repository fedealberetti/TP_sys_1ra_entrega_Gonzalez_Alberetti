import matplotlib.pyplot as plt
import numpy as np

def graficar_dominio_temporal(señal, frecuencia_muestreo=1.0):
    """
    Visualiza el dominio temporal de una señal dada como array/list.
    """
    tiempo = np.arange(len(señal)) / frecuencia_muestreo
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, señal)
    plt.title('Dominio Temporal de la Señal')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

