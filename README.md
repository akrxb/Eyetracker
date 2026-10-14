# Eye Tracker con MediaPipe y Control de Cursor 👁️🖱️

Un sistema de seguimiento ocular (*eye tracking*) en tiempo real mediante cámara web que permite controlar la posición del cursor e interactuar con elementos en pantalla (como botones virtuales) sin necesidad de hardware dedicado.

El proyecto utiliza **MediaPipe Face Landmarker** para la detección precisa de puntos clave faciales y del iris, combinando modelos de regresión adaptativos (**Ridge Regression**) y filtros de suavizado en tiempo real (**One Euro Filter**) para lograr una navegación estable y precisa.

---

## 🚀 Características principales

- **Puntos clave e invarianza:** Estimación de la posición del iris ($u, v$) en el sistema de coordenadas propio de cada ojo, haciéndolo resistente a inclinaciones leves de cabeza.
- **Compensación de pose y blendshapes:** Integración opcional de rasgos de pose de cabeza y *blendshapes* de mirada de MediaPipe.
- **Calibración guiada (15 puntos):** Proceso de calibración adaptativo con eliminación automática de datos atípicos y entrenamiento supervisado mediante validación cruzada.
- **Filtro de fijación y One Euro Filter:**
  - Congelamiento del cursor durante fijaciones para evitar vibraciones (*jitter*).
  - Suavizado adaptativo con filtro One Euro para reducir retardo (*lag*) durante sacádicas (movimientos rápidos del ojo).
- **Control por dwell (permanencia):** Selección de botones por tiempo de mirada fijada e histéresis espacial para evitar clics accidentales.
- **Detección de parpadeo:** Bloqueo temporal del cursor al cerrar los ojos para evitar saltos indeseados.

---

## 🛠️ Requisitos e Instalación

Si dispones de Anaconda o Miniconda instalado, puedes recrear exactamente el entorno virtual desde el archivo `environment.yml` incluido en el repositorio:

```bash
# 1. Clonar el repositorio
git clone https://github.com/akrxb/Eyetracker.git
cd Eyetracker

# 2. Crear el entorno virtual
conda env create -f environment.yml

# 3. Activar el entorno
conda activate eyetracking
