import tkinter as tk
from tkinter import filedialog, messagebox
from src.audio.generadores import *

# Crear la ventana principal
root = tk.Tk()
root.title("Medición IR - Proyecto de Audio")
root.geometry("500x400")

# Título
titulo = tk.Label(root, text="¿Qué desea hacer?", font=("Arial", 16))
titulo.pack(pady=20)

# Funciones vacías para conectar luego
def grabar_reproducir():
    print("→ Grabando tu propia RI")

def cargar_archivos():
    print("→ Cargando archivos")

def sintetizar_ir():
    print("→ Sintetizando IR")

# Botones principales
tk.Button(root, text=" Graba tu propia RI", command=grabar_reproducir, width=30).pack(pady=10)
tk.Button(root, text=" Cargar archivos .wav de sine sweep y FI", command=cargar_archivos, width=30).pack(pady=10)
tk.Button(root, text=" Sintetizar IR a partir de RT60 por bandas", command=sintetizar_ir, width=30).pack(pady=10)

# Iniciar la interfaz
root.mainloop()

print("Programa de medición de parámetros acústicos ISO 3382")
print("-----------------------------------------------------")

opcion = int(input("-Medir recinto (1)\n-Generar IR (2)\n-Cargar archivos (3)\n-Salir (4)\n\n--> "))

# Medir recinto
if opcion == 1:
    fs = int(input("Ingrese la frecuencia de muestreo: ")
    T = int(input("Ingrese la duración del audio sine-sweep: ")
    sweep = generar_sine_sweep(20,20000,T,fs)# Devuelve sweep, filtro_inverso y fs
    sweep_norm, t_norm = normalizar_senial(sweep,t)
    grabacion, fs = grabador(15,True)  # Reproduce sweep.wav mientras graba
    ir = ir(grabacion,filtro_inverso,True)  # Se obtiene la IR de la grabación

