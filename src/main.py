import tkinter as tk
from tkinter import messagebox, filedialog
from audio.generadores import *
from audio.io_audio import *
from audio.procesamiento import sintetizar_respuesta_impulso, funcRI
from utils.graficotemporal import graficar_dominio_temporal
import numpy as np
from scipy.io import wavfile  # <-- para cargar wav

# --- Datos precargados ---
banda8vas = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
tiempox8va = [6.98, 5.45, 4.49, 5.15, 5.43, 4.78, 3.6, 2.51, 1.45, 0.98]

frecuencias_tercios = [
    25, 31, 39, 50, 62, 79, 99, 125, 157, 198,
    250, 315, 397, 500, 630, 794, 1000, 1260, 1587, 2000,
    2520, 3175, 4000, 5040, 6350, 8000, 10079, 12699, 16000
]
t60_tercios = [
    38.89, 36.02, 29.22, 20.60, 22.32, 23.35, 22.36, 18.38, 18.60, 16.15,
    14.81, 14.29, 13.51, 11.79, 10.66, 10.03, 9.09, 7.85, 6.53, 5.60,
    4.68, 3.66, 2.83, 2.13, 1.56, 1.10, 0.84, 0.64, 0.56
]

fs = 44100
ventanas_abiertas = []
respuesta_generada = None

# --- Función para cargar WAV ---
def cargar_wav(ruta):
    fs_wav, data = wavfile.read(ruta)
    if fs_wav != fs:
        messagebox.showwarning("Aviso", f"La frecuencia de muestreo del archivo ({fs_wav}) no coincide con la esperada ({fs}).")
    # Convertir a mono si es estéreo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    # Normalizar si es int16
    if data.dtype == 'int16':
        data = data.astype(np.float32) / 32768.0
    return data

def cerrar_ventanas_secundarias():
    for v in ventanas_abiertas:
        try:
            v.destroy()
        except:
            pass
    ventanas_abiertas.clear()

def acciones_posteriores(nombre_sugerido):
    ventana_acciones = tk.Toplevel(root)
    ventanas_abiertas.append(ventana_acciones)
    ventana_acciones.title("Acciones posteriores")

    def guardar():
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")], initialfile=nombre_sugerido)
        if ruta_guardado:
            guardar_wav(ruta_guardado, respuesta_generada, fs)
            messagebox.showinfo("Guardado", f"Archivo guardado con éxito:\n{ruta_guardado}")

    def reproducir():
        reproducir_audio(respuesta_generada, fs)
        messagebox.showinfo("Reproducción", "Reproducción finalizada.")

    def graficar():
        graficar_dominio_temporal(respuesta_generada, fs)
        messagebox.showinfo("Gráfico", "Gráfico generado.")

    tk.Button(ventana_acciones, text="Guardar WAV", command=guardar).pack(pady=5)
    tk.Button(ventana_acciones, text="Reproducir", command=reproducir).pack(pady=5)
    tk.Button(ventana_acciones, text="Graficar", command=graficar).pack(pady=5)
    tk.Button(ventana_acciones, text="Volver al inicio", command=lambda: [cerrar_ventanas_secundarias(), ventana_acciones.destroy()]).pack(pady=10)

def cargar_ri_desde_wav():
    global respuesta_generada
    archivo_ri = filedialog.askopenfilename(title="Seleccione el archivo de Respuesta al Impulso", filetypes=[("WAV files", "*.wav")])
    if not archivo_ri:
        return
    
    respuesta_generada = cargar_wav(archivo_ri)
    messagebox.showinfo("Archivo cargado", f"RI cargada correctamente:\n{archivo_ri.split('/')[-1]}")
    messagebox.showinfo("Procesamiento", "Archivo cargado. Se procede a mostrar opciones de procesamiento de tu respuesta al impulso.")
    acciones_posteriores("respuesta_impulso.wav")

def generar_ir_sintetizada(t60_por_banda, frecuencias, duracion_total, nombre_archivo):
    global respuesta_generada
    _, respuesta_generada = sintetizar_respuesta_impulso(t60_por_banda, fs, duracion_total, frecuencias)
    messagebox.showinfo("Listo", "Respuesta al impulso sintetizada generada (no se guardó todavía).")
    acciones_posteriores(nombre_archivo)

def mostrar_info_y_generar(tipo):
    ventana = tk.Toplevel(root)
    ventanas_abiertas.append(ventana)
    ventana.title(f"Síntesis por {tipo}")

    if tipo == "octava":
        frecs = banda8vas
        t60s = tiempox8va
        duracion = 8.0
        nombre = "RI_R1_Reactor_Hall.wav"
    else:
        frecs = frecuencias_tercios
        t60s = t60_tercios
        duracion = 40.0
        nombre = "RI_Tercios_Octava.wav"

    tk.Label(ventana, text="Frecuencia central (Hz) - T60 (s)", font=("Arial", 12, "bold")).pack(pady=10)
    texto = "\n".join([f"{f} Hz - {t} s" for f, t in zip(frecs, t60s)])
    tk.Message(ventana, text=texto, width=400).pack(pady=10)

    tk.Button(ventana, text="Generar Respuesta al Impulso", command=lambda: generar_ir_sintetizada(t60s, frecs, duracion, nombre)).pack(pady=10)
    tk.Button(ventana, text="Volver al inicio", command=lambda: [cerrar_ventanas_secundarias(), ventana.destroy()]).pack(pady=5)

def sintetizar_ir():
    menu = tk.Toplevel(root)
    ventanas_abiertas.append(menu)
    menu.title("Tipo de bandas")

    tk.Label(menu, text="Seleccione el tipo de bandas para sintetizar la RI:", font=("Arial", 12)).pack(pady=10)
    tk.Button(menu, text="Octava", command=lambda: mostrar_info_y_generar("octava"), width=20).pack(pady=5)
    tk.Button(menu, text="Tercio de octava", command=lambda: mostrar_info_y_generar("tercio"), width=20).pack(pady=5)
    tk.Button(menu, text="Volver al inicio", command=lambda: [cerrar_ventanas_secundarias(), menu.destroy()]).pack(pady=10)

def cargar_y_convolucionar():
    global respuesta_generada
    archivo_sweep = filedialog.askopenfilename(title="Seleccione el archivo grabado (sine sweep)", filetypes=[("WAV files", "*.wav")])
    if not archivo_sweep:
        return
    archivo_filtro = filedialog.askopenfilename(title="Seleccione el archivo del filtro inverso", filetypes=[("WAV files", "*.wav")])
    if not archivo_filtro:
        return

    messagebox.showinfo("Archivos cargados", f"Sweep: {archivo_sweep.split('/')[-1]}\nFI: {archivo_filtro.split('/')[-1]}\n\nSe procederá a la convolución.")
    respuesta_generada = funcRI(archivo_filtro, archivo_sweep)
    respuesta_generada = normalizar_senial(respuesta_generada, np.linspace(0, len(respuesta_generada)/fs, len(respuesta_generada)))[0]

    messagebox.showinfo("Proceso completo", "Respuesta al impulso obtenida por convolución.")
    acciones_posteriores("RI_desde_archivos.wav")

def grabar_reproducir():
    ventana_grabar = tk.Toplevel(root)
    ventanas_abiertas.append(ventana_grabar)
    ventana_grabar.title("Grabar tu propia RI")

    tk.Label(ventana_grabar, text="Ingrese la duración del sine sweep en segundos:").pack(pady=10)
    entrada_duracion = tk.Entry(ventana_grabar)
    entrada_duracion.pack(pady=5)

    def iniciar_proceso():
        global respuesta_generada
        try:
            duracion = float(entrada_duracion.get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese una duración válida (número).")
            return

        sine, vec_t, R, K, L, w1 = generar_sine_sweep(20, 20000, duracion, fs)
        normalizar_senial(sine, vec_t)
        guardar_wav("sine_sweep.wav", sine)

        filtinv = generar_fi(sine, vec_t, R, K, L, w1)
        normalizar_senial(filtinv, vec_t)
        guardar_wav("filtro_inverso.wav", filtinv)

        messagebox.showinfo("Listo", f"Sine sweep de {duracion} segundos y su filtro inverso fueron generados.")
        messagebox.showinfo("Atención", "Se reproducirá el sweep y se grabará la respuesta en el recinto.")

        grabado = reproducir_y_grabar(sine, fs)
        guardar_wav("grabacion_usuario.wav", grabado)

        respuesta_generada = funcRI("filtro_inverso.wav", "grabacion_usuario.wav")
        respuesta_generada = normalizar_senial(respuesta_generada, vec_t)[0]
        messagebox.showinfo("Proceso completo", "Respuesta al impulso obtenida por convolución.")
        acciones_posteriores("RI_usuario.wav")

    tk.Button(ventana_grabar, text="Iniciar", command=iniciar_proceso).pack(pady=10)
    tk.Button(ventana_grabar, text="Volver al inicio", command=lambda: [cerrar_ventanas_secundarias(), ventana_grabar.destroy()]).pack(pady=10)

# GUI principal
root = tk.Tk()
root.title("Medición IR - Proyecto de Audio")
root.geometry("500x450")

titulo = tk.Label(root, text="¿Qué desea hacer?", font=("Arial", 16))
titulo.pack(pady=20)

tk.Button(root, text="Graba tu propia RI", command=grabar_reproducir, width=30).pack(pady=10)
tk.Button(root, text="Cargar sine sweep y filtro inverso", command=cargar_y_convolucionar, width=30).pack(pady=10)
tk.Button(root, text="Cargar tu propia RI (archivo WAV)", command=cargar_ri_desde_wav, width=30).pack(pady=10)
tk.Button(root, text="Sintetizar RI a partir de RT60 por bandas", command=sintetizar_ir, width=30).pack(pady=10)

root.mainloop()
