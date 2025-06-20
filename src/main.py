import tkinter as tk
from tkinter import filedialog, messagebox

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
