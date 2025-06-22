import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from audio.generadores import *
from audio.io_audio import *
from audio.procesamiento import *
from utils.graficotemporal import graficar_dominio_temporal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import csv

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
    if len(data.shape) > 1:
        data = data.mean(axis=1)
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
    tk.Button(ventana_acciones, text="Análisis avanzado", command=ventana_analisis_avanzado).pack(pady=5)
    tk.Button(ventana_acciones, text="Volver al inicio", command=lambda: [cerrar_ventanas_secundarias(), ventana_acciones.destroy()]).pack(pady=10)

# --- Función para mostrar resultados en tabla ---
def mostrar_resultados_tabla(resultados):
    ventana_resultado = tk.Toplevel()
    ventana_resultado.title("Parámetros calculados por banda")

    frame = tk.Frame(ventana_resultado)
    frame.pack(fill="both", expand=True)

    tabla = ttk.Treeview(frame, show="headings")

    # --- Columnas dinámicas ---
    columnas = ["Parámetro"]
    nombres_banda = [nombre for nombre, _ in resultados]
    columnas.extend(nombres_banda)
    tabla["columns"] = columnas

    for col in columnas:
        tabla.heading(col, text=col)
        tabla.column(col, anchor="center", minwidth=80, stretch=True)

    # --- Estilo visual con bordes y encabezados destacados ---
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
    style.configure("Treeview", font=("Helvetica", 10), rowheight=25)
    style.map("Treeview")

    # --- Parámetros en el orden deseado ---
    parametros_orden = ["T60", "EDT", "C50", "C80", "D50"]

    # --- Organizar resultados por secciones ---
    secciones = {
        "General": [],
        "Tercio de Octava": [],
        "Octava": []
    }

    for nombre_banda, datos in resultados:
        if nombre_banda.startswith("Tercio"):
            secciones["Tercio de Octava"].append((nombre_banda, datos))
        elif nombre_banda.startswith("Octava"):
            secciones["Octava"].append((nombre_banda, datos))
        else:
            secciones["General"].append((nombre_banda, datos))

    def agregar_filas(nombre_seccion, lista_datos):
        # --- Separador visual de secciones ---
        tabla.insert("", "end", values=(nombre_seccion.upper(), *[""] * (len(columnas) - 1)))

        # Agregar filas por cada parámetro en orden
        for param in parametros_orden:
            fila = [param]
            for nombre_banda, datos in lista_datos:
                valor = datos.get(param, "")
                if isinstance(valor, float):
                    fila.append(f"{valor:.2f}")
                else:
                    fila.append(str(valor))
            tabla.insert("", "end", values=fila)

    for nombre_seccion in secciones:
        agregar_filas(nombre_seccion, secciones[nombre_seccion])

    # --- Scroll y empaque ---
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tabla.yview)
    tabla.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    tabla.pack(fill="both", expand=True)

    # --- Exportar CSV ---
    def exportar_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columnas)
                for row in tabla.get_children():
                    writer.writerow(tabla.item(row)["values"])
            messagebox.showinfo("Exportación", "Archivo CSV exportado correctamente.")

    tk.Button(ventana_resultado, text="Exportar como CSV", command=exportar_csv).pack(pady=10)



def calcular_parametros_avanzados(opciones):
    # Aplicar suavizado
    if opciones["metodo"].get() == "hilbert":
        smooth = aplicar_transformada_hilbert(respuesta_generada)
    else:
        smooth = promedio_movil(respuesta_generada, opciones["L"].get())

    # Aplicar Lundeby si está seleccionado
    if opciones["usar_lundeby"].get():
        cruce, smooth = lundeby(smooth)
    else:
        cruce = None

    resultados = []
    nyquist = fs / 2

    # --- Función interna para análisis de una banda ---
    def procesar_banda(senal_smoothed, nombre_banda, cruce=None):
        banda_resultados = {}

        if opciones["hacer_schroeder"].get():
            try:
                sch = schroeder(senal_smoothed, cruce)
                banda_resultados["Schroeder"] = sch

                if opciones["hacer_regresion"].get():
                    tiempo = np.linspace(0, len(sch) / fs, len(sch))
                    pendiente, ordenada, _ = regresion_lineal_iso3382(tiempo, sch)
                    banda_resultados["Regresión"] = {
                        "pendiente": pendiente,
                        "ordenada": ordenada
                    }
            except Exception as e:
                print(f"Error en Schroeder/Regresión para {nombre_banda}: {str(e)}")
                banda_resultados["Schroeder"] = "Error"
                if opciones["hacer_regresion"].get():
                    banda_resultados["Regresión"] = "Error"

        
        
        if opciones["t60"].get():
            try:
                pendiente = banda_resultados.get("Regresión", {}).get("pendiente", None)
                if pendiente is not None:
                    banda_resultados["T60"] = param_t60(pendiente, ordenada)
                else:
                    banda_resultados["T60"] = "Falta pendiente"
            except Exception as e:
                print(f"Error calculando T60 para {nombre_banda}: {str(e)}")
                banda_resultados["T60"] = "Error"

        if opciones["edt"].get():
            try:
                pendiente = banda_resultados.get("Regresión", {}).get("pendiente", None)
                if pendiente is not None:
                    banda_resultados["EDT"] = param_edt(pendiente, fs)
                else:
                    banda_resultados["EDT"] = "Falta pendiente"
            except Exception as e:
                print(f"Error calculando EDT para {nombre_banda}: {str(e)}")
                banda_resultados["EDT"] = "Error"

        if opciones["c50"].get():
            try:
                banda_resultados["C50"] = param_c80(senal_smoothed, t=50, fs=fs)
            except Exception as e:
                print(f"Error calculando C50 para {nombre_banda}: {str(e)}")
                banda_resultados["C50"] = "Error"

        if opciones["c80"].get():
            try:
                banda_resultados["C80"] = param_c80(senal_smoothed, t=80, fs=fs)
            except Exception as e:
                print(f"Error calculando C80 para {nombre_banda}: {str(e)}")
                banda_resultados["C80"] = "Error"

        if opciones["d50"].get():
            try:
                banda_resultados["D50"] = param_d50(senal_smoothed, fs=fs)
            except Exception as e:
                print(f"Error calculando D50 para {nombre_banda}: {str(e)}")
                banda_resultados["D50"] = "Error"

        resultados.append((nombre_banda, banda_resultados))

    # Procesar señal general
    if opciones["general"].get():
        procesar_banda(smooth, "General", cruce)

    # Procesar bandas de octava
    if opciones["octava"].get():
        frecuencias_octava_filtradas = [fc for fc in banda8vas if fc * np.sqrt(2) <= nyquist]
        for fc in frecuencias_octava_filtradas:
            try:
                señal_filtrada = filtronorma(smooth, fs, fc, 'octava')
                procesar_banda(señal_filtrada, f"Octava {fc}Hz", cruce)
            except Exception as e:
                print(f"Error procesando octava {fc}Hz: {str(e)}")

    # Procesar bandas de tercio de octava
    if opciones["tercio"].get():
        factor_tercio = 2 ** (1/6)
        frecuencias_tercio_filtradas = [fc for fc in frecuencias_tercios if fc * factor_tercio <= nyquist]
        for fc in frecuencias_tercio_filtradas:
            try:
                señal_filtrada = filtronorma(smooth, fs, fc, 'tercio')
                procesar_banda(señal_filtrada, f"Tercio {fc}Hz", cruce)
            except Exception as e:
                print(f"Error procesando tercio {fc}Hz: {str(e)}")

    # Mostrar resultados
    mostrar_resultados_tabla(resultados)



# --- Ventana análisis avanzado ---
def ventana_analisis_avanzado():
    ventana = tk.Toplevel()
    ventana.title("Análisis Avanzado")

    # Variables de control
    opciones = {
        "metodo": tk.StringVar(value="movil"),
        "usar_lundeby": tk.BooleanVar(value=True),
        "hacer_schroeder": tk.BooleanVar(value=True),
        "hacer_regresion": tk.BooleanVar(value=True),
        "L": tk.IntVar(value=481),
        "t60": tk.BooleanVar(value=True),
        "edt": tk.BooleanVar(value=True),
        "c50": tk.BooleanVar(value=True),
        "c80": tk.BooleanVar(value=True),
        "d50": tk.BooleanVar(value=False),
        "general": tk.BooleanVar(value=True),
        "octava": tk.BooleanVar(value=False),
        "tercio": tk.BooleanVar(value=False)
    }

    # --- Método de suavizado ---
    frame_suavizado = tk.LabelFrame(ventana, text="Método de Suavizado", padx=10, pady=5)
    frame_suavizado.pack(fill="x", padx=10, pady=10)

    def actualizar_visibilidad_L():
        if opciones["metodo"].get() == "movil":
            entrada_L_label.pack(anchor="w", padx=20)
            entrada_L.pack(anchor="w", padx=20, pady=(0, 5))
        else:
            entrada_L_label.pack_forget()
            entrada_L.pack_forget()

    tk.Radiobutton(frame_suavizado, text="Promedio Móvil", variable=opciones["metodo"],
                   value="movil", command=actualizar_visibilidad_L).pack(anchor="w")
    tk.Radiobutton(frame_suavizado, text="Transformada de Hilbert", variable=opciones["metodo"],
                   value="hilbert", command=actualizar_visibilidad_L).pack(anchor="w")

    entrada_L_label = tk.Label(frame_suavizado, text="Ancho de Ventana (L):")
    entrada_L = tk.Entry(frame_suavizado, textvariable=opciones["L"], width=10)
    actualizar_visibilidad_L()

    # --- Otras opciones ---
    tk.Checkbutton(ventana, text="Usar método de Lundeby", 
                   variable=opciones["usar_lundeby"]).pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(ventana, text="Incluir análisis Schroeder", 
                   variable=opciones["hacer_schroeder"]).pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(ventana, text="Incluir regresión lineal", 
                   variable=opciones["hacer_regresion"]).pack(anchor="w", padx=10, pady=5)

    # --- Parámetros acústicos ---
    tk.Label(ventana, text="Parámetros a calcular:").pack(anchor="w", padx=10, pady=(10,0))
    tk.Checkbutton(ventana, text="T60", variable=opciones["t60"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="EDT", variable=opciones["edt"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="C50", variable=opciones["c50"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="C80", variable=opciones["c80"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="D50", variable=opciones["d50"]).pack(anchor="w", padx=20)

    # --- Bandas de frecuencia ---
    tk.Label(ventana, text="Bandas de frecuencia:").pack(anchor="w", padx=10, pady=(10,0))
    tk.Checkbutton(ventana, text="Señal General", variable=opciones["general"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="Bandas de Octava", variable=opciones["octava"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="Bandas de Tercio", variable=opciones["tercio"]).pack(anchor="w", padx=20)

    # --- Botón de ejecución ---
    tk.Button(ventana, text="Calcular Parámetros", 
              command=lambda: calcular_parametros_avanzados(opciones)).pack(pady=20)
def hacer_schroeder(opciones):
    metodo = opciones["metodo"].get()
    if metodo == "hilbert":
        señal = aplicar_transformada_hilbert(respuesta_generada)
    else:
        señal = promedio_movil_convolucion(respuesta_generada, opciones["L"].get())

    if opciones["usar_lundeby"].get():
        señal = lundeby(señal, fs)

    if opciones["ver_schroeder"].get():
        mostrar_schroeder(señal)



# # --- Ventana parámetros acústicos con selección octava, tercio y general ---
# def ventana_parametros():
#     ventana = tk.Toplevel()
#     ventana.title("Parámetros acústicos")
#     seleccion = {
#         "edt": tk.BooleanVar(value=True),
#         "c80": tk.BooleanVar(value=True),
#         "d50": tk.BooleanVar(value=False),
#         "t60": tk.BooleanVar(value=True),
#         "general": tk.BooleanVar(value=True),
#         "octava": tk.BooleanVar(value=False),
#         "tercio": tk.BooleanVar(value=False),
#     }

#     # Parámetros acústicos
#     tk.Checkbutton(ventana, text="EDT", variable=seleccion["edt"]).pack(anchor="w", padx=10)
#     tk.Checkbutton(ventana, text="C80", variable=seleccion["c80"]).pack(anchor="w", padx=10)
#     tk.Checkbutton(ventana, text="D50", variable=seleccion["d50"]).pack(anchor="w", padx=10)
#     tk.Checkbutton(ventana, text="T60 (T10, T20, T30)", variable=seleccion["t60"]).pack(anchor="w", padx=10)

#     # Opciones de análisis general, octava y tercio
#     tk.Label(ventana, text="Opciones de análisis:").pack(anchor="w", padx=10, pady=(10,0))
#     tk.Checkbutton(ventana, text="Señal General (sin filtrar)", variable=seleccion["general"]).pack(anchor="w", padx=20)
#     tk.Checkbutton(ventana, text="Bandas de Octava", variable=seleccion["octava"]).pack(anchor="w", padx=20)
#     tk.Checkbutton(ventana, text="Bandas de Tercio de Octava", variable=seleccion["tercio"]).pack(anchor="w", padx=20)

#     tk.Button(ventana, text="Calcular parámetros", command=lambda: calcular_parametros(respuesta_generada, seleccion)).pack(pady=10)



# --- Ventana parámetros acústicos con selección octava, tercio y general ---
def ventana_parametros():
    ventana = tk.Toplevel()
    ventana.title("Parámetros acústicos")
    seleccion = {
        "edt": tk.BooleanVar(value=True),
        "c80": tk.BooleanVar(value=True),
        "d50": tk.BooleanVar(value=False),
        "t60": tk.BooleanVar(value=True),
        "general": tk.BooleanVar(value=True),
        "octava": tk.BooleanVar(value=False),
        "tercio": tk.BooleanVar(value=False),
    }

    # Parámetros acústicos
    tk.Checkbutton(ventana, text="EDT", variable=seleccion["edt"]).pack(anchor="w", padx=10)
    tk.Checkbutton(ventana, text="C80", variable=seleccion["c80"]).pack(anchor="w", padx=10)
    tk.Checkbutton(ventana, text="D50", variable=seleccion["d50"]).pack(anchor="w", padx=10)
    tk.Checkbutton(ventana, text="T60 (T10, T20, T30)", variable=seleccion["t60"]).pack(anchor="w", padx=10)

    # Opciones de análisis general, octava y tercio
    tk.Label(ventana, text="Opciones de análisis:").pack(anchor="w", padx=10, pady=(10,0))
    tk.Checkbutton(ventana, text="Señal General (sin filtrar)", variable=seleccion["general"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="Bandas de Octava", variable=seleccion["octava"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="Bandas de Tercio de Octava", variable=seleccion["tercio"]).pack(anchor="w", padx=20)

    tk.Button(ventana, text="Calcular parámetros", command=lambda: calcular_parametros(respuesta_generada, seleccion)).pack(pady=10)


def calcular_parametros(signal, seleccion):
    resultados = []

    def extraer_valor(x):
        if isinstance(x, (np.ndarray, list)):
            return float(x[0]) if len(x) > 0 else 0.0
        return float(x)

    def calc_parametros_senal(senal, etiqueta):
        res = [f"\n--- {etiqueta} ---"]
        if seleccion["edt"].get():
            res.append(f"EDT: {extraer_valor(param_edt(senal)):.2f} s")
        if seleccion["c80"].get():
            res.append(f"C80: {extraer_valor(param_c80(senal, fs)):.2f} dB")
        if seleccion["d50"].get():
            res.append(f"D50: {extraer_valor(param_d50(senal, fs)):.2f}")
        if seleccion["t60"].get():
            res.append("T60 (T10, T20, T30): implementar funciones específicas")
        return "\n".join(res)

    # Análisis general sin filtrado
    if seleccion["general"].get():
        resultados.append(calc_parametros_senal(signal, "Señal general"))

    # Análisis por bandas de octava
    if seleccion["octava"].get():
        frecuencias_octava = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for fc in frecuencias_octava:
            senal_filtrada = filtronorma(signal, fs, fc, tipo='octava')
            resultados.append(calc_parametros_senal(senal_filtrada, f"Octava {fc} Hz"))

    # Análisis por bandas de tercio de octava
    if seleccion["tercio"].get():
        frecuencias_tercio = [
            12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100,
            125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
            12500, 16000, 20000
        ]
        for fc in frecuencias_tercio:
            senal_filtrada = filtronorma(signal, fs, fc, tipo='tercio')
            resultados.append(calc_parametros_senal(senal_filtrada, f"Tercio {fc} Hz"))

    messagebox.showinfo("Resultados", "\n".join(resultados))


def mostrar_schroeder(signal):
    energia = np.cumsum(signal[::-1] ** 2)[::-1]
    energia_db = 10 * np.log10(energia / np.max(energia))
    tiempo = np.linspace(0, len(signal) / fs, len(signal))
    plt.figure()
    plt.plot(tiempo, energia_db)
    plt.title("Curva de Schroeder")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Nivel (dB)")
    plt.grid()
    plt.show()


    
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
