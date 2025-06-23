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
import re

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

def mostrar_resultados_tabla(resultados):
    ventana_resultado = tk.Toplevel()
    ventana_resultado.title("Parámetros calculados por banda")

    frame = tk.Frame(ventana_resultado)
    frame.pack(fill="both", expand=True)

    # Crear tabla SIN encabezados visibles
    tabla = ttk.Treeview(frame, columns=[], show="")  # ← encabezados ocultos

    # Columnas dinámicas (solo internas, no visibles como encabezado)
    columnas = ["Parámetro"]
    nombres_banda = [nombre for nombre, _ in resultados]
    columnas.extend(nombres_banda)
    tabla["columns"] = columnas

    for col in columnas:
        tabla.column(col, anchor="center", minwidth=60, stretch=True)

    # Estilo visual
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Treeview", font=("Helvetica", 10), rowheight=25)
    style.map("Treeview", background=[("selected", "#ececec")])

    style.configure("Seccion.Treeview", background="#d9e1f2", font=("Helvetica", 10, "bold"))

    # Parámetros en orden deseado
    parametros_orden = ["T60", "EDT", "C50", "C80", "D50"]

    # Organizar resultados por secciones
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

    def extraer_frecuencia(nombre_banda):
        match = re.search(r"(\d+\.?\d*)", nombre_banda)
        return match.group(1) + " Hz" if match else ""

    def agregar_filas(nombre_seccion, lista_datos):
        # Fila con nombre de la sección
        iid = tabla.insert("", "end", values=(nombre_seccion.upper(), *[""] * (len(columnas) - 1)))
        tabla.item(iid, tags=("seccion",))

        # Fila de frecuencias (solo para Octava y Tercio)
        if nombre_seccion != "General":
            fila_frec = ["Frecuencia"]
            for nombre_banda, _ in lista_datos:
                fila_frec.append(extraer_frecuencia(nombre_banda))
            tabla.insert("", "end", values=fila_frec)

        # Filas con los parámetros
        for param in parametros_orden:
            fila = [param]
            for _, datos in lista_datos:
                valor = datos.get(param, "")
                if isinstance(valor, float):
                    fila.append(f"{valor:.2f}")
                else:
                    fila.append(str(valor))
            tabla.insert("", "end", values=fila)

    for nombre_seccion in secciones:
        agregar_filas(nombre_seccion, secciones[nombre_seccion])

    tabla.tag_configure("seccion", background="#d9e1f2", font=("Helvetica", 10, "bold"))

    # Scroll vertical
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tabla.yview)
    tabla.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    tabla.pack(fill="both", expand=True)

    # Exportar a CSV
    def exportar_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                # Exportar también las columnas si lo deseás
                writer.writerow(columnas)
                for row in tabla.get_children():
                    writer.writerow(tabla.item(row)["values"])
            messagebox.showinfo("Exportación", "Archivo CSV exportado correctamente.")

    tk.Button(ventana_resultado, text="Exportar como CSV", command=exportar_csv).pack(pady=10)

def calcular_parametros_avanzados(opciones):
    resultados = []
    nyquist = fs / 2

    def procesar_banda(senal_filtrada, nombre_banda):
        banda_resultados = {}

        # Suavizado
        if opciones["metodo"].get() == "hilbert":
            smooth = aplicar_transformada_hilbert(senal_filtrada)
        else:
            smooth = promedio_movil(senal_filtrada, opciones["L"].get())

        # Lundeby
        if opciones["usar_lundeby"].get():
            margin_db = opciones.get("margen_db", tk.DoubleVar(value=50)).get()
            cruce = lundeby(smooth, fs, margin_db=margin_db)
            print(f"{nombre_banda} - cruce Lundeby: {cruce}")
        else:
            cruce = None

        # Schroeder
        if opciones["hacer_schroeder"].get():
            try:
                sch = schroeder(smooth, cruce)
                banda_resultados["Schroeder"] = sch

                if opciones["hacer_regresion"].get():
                    tiempo = np.linspace(0, len(sch) / fs, len(sch))
                    sch_db = sch  # ya está en dB

                    # Regresión EDT: de 0 a -10 dB
                    mask_edt = (sch_db <= 0) & (sch_db >= -10)
                    m_edt, b_edt = np.polyfit(tiempo[mask_edt], sch_db[mask_edt], 1)

                    # Regresión T60 (T30): de -5 a -35 dB
                    mask_t60 = (sch_db <= -5) & (sch_db >= -35)
                    m_t60, b_t60 = np.polyfit(tiempo[mask_t60], sch_db[mask_t60], 1)

                    # Rectas ajustadas
                    y_edt = m_edt * tiempo + b_edt
                    y_t60 = m_t60 * tiempo + b_t60

                    banda_resultados["Regresión"] = {
                        "pendiente_t60": m_t60,
                        "ordenada_t60": b_t60,
                        "pendiente_edt": m_edt,
                        "ordenada_edt": b_edt,
                        "tiempo": tiempo,
                        "y_t60": y_t60,
                        "y_edt": y_edt
                    }
            except Exception as e:
                print(f"Error en Schroeder/Regresión para {nombre_banda}: {str(e)}")
                banda_resultados["Schroeder"] = "Error"
                banda_resultados["Regresión"] = "Error"

        # --- Parámetros ---
        reg = banda_resultados.get("Regresión", {})
        m_t60 = reg.get("pendiente_t60")
        b_t60 = reg.get("ordenada_t60")
        m_edt = reg.get("pendiente_edt")
        b_edt = reg.get("ordenada_edt")

        if opciones["t60"].get():
            try:
                if m_t60 is not None:
                    banda_resultados["T60"] = param_t60(m_t60, b_t60)
                else:
                    banda_resultados["T60"] = "Falta pendiente"
            except Exception as e:
                print(f"Error calculando T60 para {nombre_banda}: {str(e)}")
                banda_resultados["T60"] = "Error"

        if opciones["edt"].get():
            try:
                if m_edt is not None:
                    banda_resultados["EDT"] = param_edt(m_edt, b_edt)
                else:
                    banda_resultados["EDT"] = "Falta pendiente"
            except Exception as e:
                print(f"Error calculando EDT para {nombre_banda}: {str(e)}")
                banda_resultados["EDT"] = "Error"

        try:
            idx_max = np.argmax(np.abs(senal_filtrada))
            senal_alineada = senal_filtrada[idx_max:]
        except Exception as e:
            print(f"Error alineando señal para {nombre_banda}: {str(e)}")
            senal_alineada = senal_filtrada

        if opciones["c50"].get():
            try:
                banda_resultados["C50"] = param_c80(senal_alineada, t=50, fs=fs)
            except Exception as e:
                print(f"Error calculando C50 para {nombre_banda}: {str(e)}")
                banda_resultados["C50"] = "Error"

        if opciones["c80"].get():
            try:
                banda_resultados["C80"] = param_c80(senal_alineada, t=80, fs=fs)
            except Exception as e:
                print(f"Error calculando C80 para {nombre_banda}: {str(e)}")
                banda_resultados["C80"] = "Error"

        if opciones["d50"].get():
            try:
                banda_resultados["D50"] = param_d50(senal_alineada, fs=fs)
                print(banda_resultados["D50"])
            except Exception as e:
                print(f"Error calculando D50 para {nombre_banda}: {str(e)}")
                banda_resultados["D50"] = "Error"

        resultados.append((nombre_banda, banda_resultados))

        if nombre_banda == "General" and opciones.get("graficar", tk.BooleanVar(value=False)).get():
            fig, ax = plt.subplots(figsize=(10, 5))

            t_smooth = np.linspace(0, len(smooth) / fs, len(smooth))
            smooth_db = ir_a_log(smooth)
            ax.plot(t_smooth, smooth_db, label="Suavizado (log)", color="blue")

            if isinstance(sch, np.ndarray):
                t_sch = np.linspace(0, len(sch) / fs, len(sch))
                ax.plot(t_sch, sch, label="Schroeder", color="red")

            if isinstance(reg, dict):
                ax.plot(reg["tiempo"], reg["y_edt"], label="Regresión EDT", linestyle="--", color="green")
                ax.plot(reg["tiempo"], reg["y_t60"], label="Regresión T60", linestyle="--", color="black")

            ax.set_xlabel("Tiempo [s]")
            ax.set_ylabel("Nivel [dB]")
            ax.set_title("Curvas - General")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

    if opciones["general"].get():
        procesar_banda(respuesta_generada, "General")

    if opciones["octava"].get():
        frecuencias_octava_filtradas = [fc for fc in banda8vas if fc * np.sqrt(2) <= nyquist]
        for fc in frecuencias_octava_filtradas:
            try:
                senal_filtrada = filtronorma(respuesta_generada, fs, fc, 'octava')
                procesar_banda(senal_filtrada, f"Octava {fc}Hz")
            except Exception as e:
                print(f"Error procesando octava {fc}Hz: {str(e)}")

    if opciones["tercio"].get():
        factor_tercio = 2 ** (1/6)
        frecuencias_tercio_filtradas = [fc for fc in frecuencias_tercios if fc * factor_tercio <= nyquist]
        for fc in frecuencias_tercio_filtradas:
            try:
                senal_filtrada = filtronorma(respuesta_generada, fs, fc, 'tercio')
                procesar_banda(senal_filtrada, f"Tercio {fc}Hz")
            except Exception as e:
                print(f"Error procesando tercio {fc}Hz: {str(e)}")

    mostrar_resultados_tabla(resultados)




# --- Ventana análisis avanzado ---
def ventana_analisis_avanzado():
    ventana = tk.Toplevel()
    ventana.title("Análisis Avanzado")

    # --- Variables de control ---
    opciones = {
        "metodo": tk.StringVar(value="movil"),
        "usar_lundeby": tk.BooleanVar(value=True),
        "margen_db": tk.DoubleVar(value=50),
        "hacer_schroeder": tk.BooleanVar(value=True),
        "hacer_regresion": tk.BooleanVar(value=True),
        "L": tk.IntVar(value=187),
        "t60": tk.BooleanVar(value=True),
        "edt": tk.BooleanVar(value=True),
        "c50": tk.BooleanVar(value=True),
        "c80": tk.BooleanVar(value=True),
        "d50": tk.BooleanVar(value=True),  # ← activado por defecto
        "general": tk.BooleanVar(value=True),
        "octava": tk.BooleanVar(value=True),  # ← activado por defecto
        "tercio": tk.BooleanVar(value=True),  # ← activado por defecto
        "graficar": tk.BooleanVar(value=True),  # ← opción nueva
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
    frame_opciones = tk.LabelFrame(ventana, text="Otras Opciones", padx=10, pady=5)
    frame_opciones.pack(fill="x", padx=10, pady=5)

    def actualizar_margen_lundeby():
        if opciones["usar_lundeby"].get():
            margen_label.pack(anchor="w", padx=20)
            margen_entry.pack(anchor="w", padx=20, pady=(0, 5))
        else:
            margen_label.pack_forget()
            margen_entry.pack_forget()

    tk.Checkbutton(frame_opciones, text="Usar método de Lundeby",
                   variable=opciones["usar_lundeby"], command=actualizar_margen_lundeby).pack(anchor="w")

    margen_label = tk.Label(frame_opciones, text="Margen (dB) para Lundeby:")
    margen_entry = tk.Entry(frame_opciones, textvariable=opciones["margen_db"], width=10)
    actualizar_margen_lundeby()

    tk.Checkbutton(frame_opciones, text="Incluir análisis Schroeder",
                   variable=opciones["hacer_schroeder"]).pack(anchor="w", padx=10)
    tk.Checkbutton(frame_opciones, text="Incluir regresión lineal",
                   variable=opciones["hacer_regresion"]).pack(anchor="w", padx=10)

    # --- Parámetros acústicos ---
    frame_parametros = tk.LabelFrame(ventana, text="Parámetros a calcular", padx=10, pady=5)
    frame_parametros.pack(fill="x", padx=10, pady=5)

    for texto, clave in [("T60", "t60"), ("EDT", "edt"), ("C50", "c50"),
                         ("C80", "c80"), ("D50", "d50")]:
        tk.Checkbutton(frame_parametros, text=texto, variable=opciones[clave]).pack(anchor="w", padx=20)

    # --- Bandas de frecuencia ---
    frame_bandas = tk.LabelFrame(ventana, text="Bandas de Frecuencia", padx=10, pady=5)
    frame_bandas.pack(fill="x", padx=10, pady=5)

    for texto, clave in [("Señal General", "general"), 
                         ("Bandas de Octava", "octava"), 
                         ("Bandas de Tercio", "tercio")]:
        tk.Checkbutton(frame_bandas, text=texto, variable=opciones[clave]).pack(anchor="w", padx=20)

    # --- Generar gráfico ---
    tk.Checkbutton(ventana, text="Mostrar gráfico para señal general", 
                   variable=opciones["graficar"]).pack(anchor="w", padx=10, pady=10)

    # --- Botón de ejecución ---
    tk.Button(ventana, text="Calcular Parámetros",
              command=lambda: calcular_parametros_avanzados(opciones)).pack(pady=20)


    
def cargar_ri_desde_wav():
    global respuesta_generada
    archivo_ri = filedialog.askopenfilename(
        title="Seleccione el archivo de Respuesta al Impulso",
        filetypes=[("WAV files", "*.wav")]
    )
    if not archivo_ri:
        return

    # Cargar solo la señal
    f = cargar_wav(archivo_ri)

    # Crear eje temporal
    fs = 44100  # Ajustá esto si tu archivo tiene otra frecuencia
    t = np.arange(len(f)) / fs

    # Normalizar señal
    respuesta_generada = normalizar_senial(f, t)[0]

    # Mensajes
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
