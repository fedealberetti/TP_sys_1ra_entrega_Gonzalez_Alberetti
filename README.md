# TP_sys_1ra_entrega_Gonzalez_Alberetti
Primera entrega del Trabajo práctico de señales y sistemas Ing. en Sonido Untref


##  Instalación

1. **Clonar repositorio**:
```bash
git clone https://github.com/tu_usuario/ruido_rosa_project.git
cd ruido_rosa_project

pip install -r requirements.txt

TP_sys_1ra_entrega_Gonzalez_Alberetti/
├── src/                    # Código fuente principal
│   ├── audio/              # Funciones relacionadas con audio
│   │   ├── generadores.py  # Generación de ruido rosa
│   │   └── io_audio.py     # Grabación/reproducción de WAV
│   └── main.py             # Script principal de ejecución
├── tests/                  # Pruebas unitarias
├── requirements.txt        # Dependencias
└── README.md               # Documentación

1. Generación de Ruido Rosa (generadores.py)
Algoritmo:
Implementación del método de Voss-McCartney para crear ruido rosa mediante múltiples fuentes de ruido blanco combinadas.

Flujo:

Inicializa matriz con valores aleatorios

Aplica actualizaciones probabilísticas

Rellena valores faltantes (forward-fill)

Suma columnas y normaliza la señal







2. Manejo de Audio (io_audio.py)
Funcionalidades:

Guardar WAV:

python
def guardar_wav(nombre_archivo, señal, fs=44100)
Reproducir audio:

python
def reproducir_audio(señal, fs=44100)
Dependencias:

scipy.io.wavfile para guardar archivos

sounddevice para reproducción






Contribución con fork

Hacer fork del repositorio

Crear una rama:

bash
git checkout -b feature/nueva-funcion
Realizar tus cambios

Enviar un Pull Request