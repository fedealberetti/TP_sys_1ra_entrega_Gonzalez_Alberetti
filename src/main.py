import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from audio.generadores import *
from audio.io_audio import *
from audio.procesamiento import *
from utils.graficotemporal import graficar_dominio_temporal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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
    ventana = tk.Toplevel()
    ventana.title("Resultados de Análisis")
    ventana.geometry("800x600")
    
    # Crear Treeview con barra de desplazamiento
    frame = tk.Frame(ventana)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Barra de desplazamiento vertical
    scrollbar = ttk.Scrollbar(frame, orient="vertical")
    
    # Crear Treeview
    tree = ttk.Treeview(frame, columns=("Banda", "Parámetro", "Valor"), 
                        show="headings", yscrollcommand=scrollbar.set)
    
    # Configurar columnas
    tree.column("Banda", width=200, anchor="w")
    tree.column("Parámetro", width=150, anchor="w")
    tree.column("Valor", width=150, anchor="w")
    
    # Configurar encabezados
    tree.heading("Banda", text="Banda")
    tree.heading("Parámetro", text="Parámetro")
    tree.heading("Valor", text="Valor")
    
    # Configurar barra de desplazamiento
    scrollbar.config(command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.pack(side="left", fill="both", expand=True)
    
    # Llenar con datos
    for banda, datos in resultados:
        for param, valor in datos.items():
            # Solo mostrar parámetros numéricos (no datos crudos como Schroeder)
            if not isinstance(valor, (np.ndarray, list)):
                tree.insert("", "end", values=(banda, param, valor))
    
    # Botón para exportar
    btn_exportar = tk.Button(ventana, text="Exportar a CSV", 
                            command=lambda: exportar_csv(resultados))
    btn_exportar.pack(pady=10)

def exportar_csv(resultados):
    archivo = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )
    
    if not archivo:
        return
    
    try:
        with open(archivo, 'w', encoding='utf-8') as f:
            f.write("Banda,Parámetro,Valor\n")
            for banda, datos in resultados:
                for param, valor in datos.items():
                    # Solo exportar parámetros numéricos
                    if not isinstance(valor, (np.ndarray, list)):
                        f.write(f'"{banda}","{param}","{valor}"\n')
        messagebox.showinfo("Éxito", f"Datos exportados a:\n{archivo}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo exportar:\n{str(e)}")

# --- Función para calcular parámetros avanzados ---
def calcular_parametros_avanzados(opciones):
    # Aplicar suavizado
    if opciones["metodo"].get() == "hilbert":
        señal = aplicar_transformada_hilbert(respuesta_generada)
    else:
        señal = promedio_movil_convolucion(respuesta_generada, opciones["L"].get())
    
    # Aplicar Lundeby si está seleccionado
    if opciones["usar_lundeby"].get():
        señal = lundeby(señal, fs)
    
    # Preparar almacenamiento de resultados
    resultados = []
    nyquist = fs / 2  # Límite de Nyquist
    
    # Función interna para procesamiento
    def procesar_banda(senal_actual, nombre_banda):
        banda_resultados = {}
        
        # Análisis Schroeder si está seleccionado
        if opciones["hacer_schroeder"].get():
            sch = schroeder(senal_actual)
            banda_resultados["Schroeder"] = sch
        
        # Cálculo de parámetros
        if opciones["edt"].get():
            try:
                banda_resultados["EDT"] = param_edt(senal_actual, fs)
            except Exception as e:
                print(f"Error calculando EDT para {nombre_banda}: {str(e)}")
                banda_resultados["EDT"] = "Error"
        
        if opciones["c80"].get():
            try:
                banda_resultados["C80"] = param_c80(senal_actual, fs)
            except Exception as e:
                print(f"Error calculando C80 para {nombre_banda}: {str(e)}")
                banda_resultados["C80"] = "Error"
        
        if opciones["d50"].get():
            try:
                banda_resultados["D50"] = param_d50(senal_actual, fs)
            except Exception as e:
                print(f"Error calculando D50 para {nombre_banda}: {str(e)}")
                banda_resultados["D50"] = "Error"
        
        if opciones["t60"].get():
            try:
                # Asumiendo que tienes una función param_t60 implementada
                banda_resultados["T60"] = "Implementación pendiente"
            except Exception as e:
                print(f"Error calculando T60 para {nombre_banda}: {str(e)}")
                banda_resultados["T60"] = "Error"
            
        resultados.append((nombre_banda, banda_resultados))
    
    # Procesar bandas seleccionadas
    if opciones["general"].get():
        procesar_banda(señal, "Señal General")
    
    if opciones["octava"].get():
        # Filtrar frecuencias que no excedan Nyquist para octavas
        frecuencias_octava_filtradas = []
        for fc in banda8vas:
            # Calcular frecuencia superior para banda de octava: fc * √2
            if fc * np.sqrt(2) <= nyquist:
                frecuencias_octava_filtradas.append(fc)
            else:
                print(f"Advertencia: Se omite banda de octava {fc}Hz (supera Nyquist)")
        
        # Procesar cada frecuencia válida
        for fc in frecuencias_octava_filtradas:
            try:
                señal_filtrada = filtronorma(señal, fs, fc, 'octava')
                procesar_banda(señal_filtrada, f"Octava {fc}Hz")
            except ValueError as e:
                print(f"Error procesando octava {fc}Hz: {str(e)}")
            except Exception as e:
                print(f"Error inesperado procesando octava {fc}Hz: {str(e)}")
    
    if opciones["tercio"].get():
        # Filtrar frecuencias que no excedan Nyquist para tercios
        frecuencias_tercio_filtradas = []
        factor_tercio = 2 ** (1/6)  # Factor para tercio de octava
        
        for fc in frecuencias_tercios:
            # Calcular frecuencia superior para banda de tercio: fc * 2^(1/6)
            if fc * factor_tercio <= nyquist:
                frecuencias_tercio_filtradas.append(fc)
            else:
                print(f"Advertencia: Se omite banda de tercio {fc}Hz (supera Nyquist)")
        
        # Procesar cada frecuencia válida
        for fc in frecuencias_tercio_filtradas:
            try:
                señal_filtrada = filtronorma(señal, fs, fc, 'tercio')
                procesar_banda(señal_filtrada, f"Tercio {fc}Hz")
            except ValueError as e:
                print(f"Error procesando tercio {fc}Hz: {str(e)}")
            except Exception as e:
                print(f"Error inesperado procesando tercio {fc}Hz: {str(e)}")
    
    # Mostrar resultados en tabla
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
        "edt": tk.BooleanVar(value=True),
        "c80": tk.BooleanVar(value=True),
        "d50": tk.BooleanVar(value=False),
        "t60": tk.BooleanVar(value=True),
        "general": tk.BooleanVar(value=True),
        "octava": tk.BooleanVar(value=False),
        "tercio": tk.BooleanVar(value=False)
    }

    # Widgets de suavizado
    tk.Label(ventana, text="Método de suavizado").pack(anchor="w", padx=10, pady=5)
    # ... (opciones de suavizado existentes) ...

    # Checkboxes para análisis
    tk.Checkbutton(ventana, text="Usar método de Lundeby", 
                   variable=opciones["usar_lundeby"]).pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(ventana, text="Incluir análisis Schroeder", 
                   variable=opciones["hacer_schroeder"]).pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(ventana, text="Incluir regresión lineal", 
                   variable=opciones["hacer_regresion"]).pack(anchor="w", padx=10, pady=5)

    # Parámetros acústicos
    tk.Label(ventana, text="Parámetros a calcular:").pack(anchor="w", padx=10, pady=(10,0))
    tk.Checkbutton(ventana, text="EDT", variable=opciones["edt"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="C80", variable=opciones["c80"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="D50", variable=opciones["d50"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="T60", variable=opciones["t60"]).pack(anchor="w", padx=20)

    # Bandas de frecuencia
    tk.Label(ventana, text="Bandas de frecuencia:").pack(anchor="w", padx=10, pady=(10,0))
    tk.Checkbutton(ventana, text="Señal General", 
                   variable=opciones["general"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="Bandas de Octava", 
                   variable=opciones["octava"]).pack(anchor="w", padx=20)
    tk.Checkbutton(ventana, text="Bandas de Tercio", 
                   variable=opciones["tercio"]).pack(anchor="w", padx=20)

    # Botón único de cálculo
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

def hacer_regresion(opciones):
    if opciones["ver_regresion"].get():
        graficar_regresion(respuesta_generada)

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

def graficar_regresion(signal):
    energia = np.cumsum(signal[::-1] ** 2)[::-1]
    energia_db = 10 * np.log10(energia / np.max(energia))
    tiempo = np.linspace(0, len(signal) / fs, len(signal))
    idx_ini = int(0.1 * len(signal))
    idx_fin = int(0.3 * len(signal))
    x = tiempo[idx_ini:idx_fin]
    y = energia_db[idx_ini:idx_fin]
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_fit = m * x + b
    plt.figure()
    plt.plot(tiempo, energia_db, label="Curva de Schroeder")
    plt.plot(x, y_fit, label=f"Regresión (pendiente = {m:.2f})")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Nivel (dB)")
    plt.title("Regresión por Mínimos Cuadrados")
    plt.grid()
    plt.legend()
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
