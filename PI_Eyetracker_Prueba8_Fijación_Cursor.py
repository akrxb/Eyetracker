"""
Eye tracker con webcam + MediaPipe -> TECLADO EN PANTALLA (QWERTY español)
Configurado para un Pixel 7 conectado por USB (modo webcam).

Novedades respecto a la Prueba8:
  1. Teclado QWERTY español (con Ñ), espacio, flechas izquierda/derecha para mover el cursor de
     texto, tecla BORRAR y tecla ENTER. Sin números por ahora.
  2. Varios modelos de mirada -> pantalla, y el programa elige el mejor por validación cruzada:
        lineal  : Ridge lineal sobre todos los rasgos
        poly2   : términos cuadráticos en los rasgos del ojo (lo que ya tenías)
        poly3   : términos cúbicos en los rasgos del ojo
        kernel  : modelo lineal + corrección no lineal suave (kernel RBF) sobre los residuos
     Al calibrar se imprime una tabla con el error de cada uno.
  3. Los puntos de calibración caen sobre las filas del teclado (más datos donde se escribe).
  4. Teclas grandes (ventana a pantalla completa), selección por permanencia (dwell) y
     "tecla más cercana" para que no haya zonas muertas entre teclas.
  5. Autocorrección de deriva: cada vez que se pulsa una tecla, el sistema compara dónde creía
     que mirabas con el centro de la tecla y desplaza suavemente el cursor para compensar.
  6. Apertura de cámara con DirectShow + MJPG en Windows (más fps y resolución real) y
     descarte de fotogramas iniciales (autoexposición).

Teclas del programa:  q / ESC = salir,  c = recalibrar,  r = reiniciar la corrección de deriva
"""
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

# ====================== Configuracion ======================
# --- Cámara (Pixel 7 en modo webcam USB) ---
# En el móvil: ajustes de USB -> "Webcam". Elige la cámara FRONTAL en la notificación del móvil.
# Si no ves imagen, prueba otro índice (0, 1, 2, 3...).
CAMARA_INDEX_O_URL = 2
CAMARA_ANCHO, CAMARA_ALTO = 1920, 720      # prueba 1920x1080: más píxeles en el iris = más precisión
                                            # (si va lento o la cámara no lo admite, vuelve a 1280x720)

PANTALLA_COMPLETA = True           # True = ventana a pantalla completa (teclas más grandes = más fácil)

SEGUNDOS_PREVIOS = 1.8             # tiempo para localizar y fijar el punto rojo
MUESTRAS_POR_PUNTO = 40            # muestras válidas por punto de calibración
MUESTRAS_DESCARTADAS = 10          # primeros fotogramas del punto (el ojo aún se asienta)
FOTOGRAMAS_CALENTAMIENTO = 30      # fotogramas que se tiran al abrir la cámara

UMBRAL_PARPADEO = 0.15
FRAMES_BLOQUEO = 3

VENTANA_MEDIANA = 5
ONE_EURO_MIN_CUTOFF = 0.4          # MENOR = cursor más estable pero con más retardo
ONE_EURO_BETA = 0.002              # MAYOR = menos retardo al mover la mirada

FIJACION_RADIO_PX = None           # None = automático (2x el ruido medido en la validación)
FIJACION_FRAMES_SALIDA = 3
FIJACION_MAX_PUNTOS = 25
IMAN_A_BOTON = True                # dibuja el cursor en el centro de la tecla activa
MOSTRAR_CURSOR_REAL = True         # punto pequeño azul con la posición real estimada (útil para depurar)

# --- Selección de teclas ---
DWELL_SEG = 3.0                    # tiempo mirando una tecla para pulsarla (bájalo cuando cojas práctica)
PAUSA_TRAS_SELECCION = 0.6         # pausa tras pulsar (para evitar repeticiones accidentales)
TOLERANCIA_PX_REL = 0.02           # una mirada a menos de (2% del ancho) de una tecla cuenta como esa tecla
HISTERESIS_PX_REL = 0.03           # la tecla activa se mantiene hasta salir (3% del ancho) de su borde

# --- Autocorrección de deriva ---
AUTOCORRECCION = True
AUTOCORR_RITMO = 0.25              # cuánto se corrige en cada pulsación (0 = nada, 1 = todo)
AUTOCORR_MAX_PX_REL = 0.12         # corrección total máxima (12% del ancho)

# --- Modelo ---
MODELO = "auto"                    # "auto" | "lineal" | "poly2" | "poly3" | "kernel"
USAR_BLENDSHAPES = True
USAR_POSE_CABEZA = True
VALIDAR = True
REFINAR_CON_VALIDACION = True
OJOS_POR_SEPARADO = False

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
VENTANA = "EyeTracker"


def _tam_pantalla():
    if PANTALLA_COMPLETA:
        try:
            import tkinter
            r = tkinter.Tk()
            r.withdraw()
            w, h = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
            return int(w), int(h)
        except Exception:
            pass
    return 1280, 720


ANCHO, ALTO = _tam_pantalla()
PANTALLA = np.array([ANCHO, ALTO], dtype=float)
ESC = ANCHO / 1280.0               # factor de escala para tamaños en píxeles

# Indices de landmarks: (iris, esquina_a, esquina_b, parpado_sup1, sup2, parpado_inf1, inf2)
OJO_A = (468, 33, 133, 159, 158, 145, 153)
OJO_B = (473, 362, 263, 386, 385, 374, 380)

# ---------- Rasgos que usa el modelo ----------
ESCALA_BASE = {"u": 0.1, "v": 0.1, "ua": 0.1, "va": 0.1, "ub": 0.1, "vb": 0.1, "bh": 0.5, "bv": 0.5,
               "ap": 0.1, "fx": 0.2, "fy": 0.2, "tx": 5.0, "ty": 5.0}
CLAVES_OJO = (["ua", "va", "ub", "vb"] if OJOS_POR_SEPARADO else ["u", "v"]) \
             + (["bh", "bv"] if USAR_BLENDSHAPES else [])
CLAVES_LIN = ["ap"] + (["fx", "fy", "tx", "ty"] if USAR_POSE_CABEZA else [])
CLAVES = CLAVES_OJO + CLAVES_LIN
N_OJO = len(CLAVES_OJO)
ESCALAS = np.array([ESCALA_BASE[k] for k in CLAVES])


class Salir(Exception):
    pass


# ====================== Teclado QWERTY español ======================
# Cada fila suma 10 "unidades" de ancho. Acciones especiales: IZQ, DER, BORRAR, ENTER, ESPACIO.
FILAS_TECLADO = [
    [(c, 1) for c in "qwertyuiop"],
    [(c, 1) for c in "asdfghjklñ"],
    [(c, 1) for c in "zxcvbnm,."] + [("BORRAR", 1)],
    [("IZQ", 2), ("ESPACIO", 4), ("DER", 2), ("ENTER", 2)],
]
MARGEN = int(12 * ESC)
HUECO = int(8 * ESC)
CAJA_TOP = int(28 * ESC)
ALTO_TEXTO = int(ALTO * 0.22)      # el cuadro de texto ocupa de CAJA_TOP a ALTO_TEXTO


def construir_teclado():
    top = ALTO_TEXTO + MARGEN
    n = len(FILAS_TECLADO)
    alto_fila = (ALTO - MARGEN - top - HUECO * (n - 1)) / n
    unidad = (ANCHO - 2 * MARGEN - 9 * HUECO) / 10
    res = []
    for r, fila in enumerate(FILAS_TECLADO):
        y1 = top + r * (alto_fila + HUECO)
        x = MARGEN
        for nombre, u in fila:
            w = u * unidad + (u - 1) * HUECO
            res.append((int(x), int(y1), int(x + w), int(y1 + alto_fila), nombre))
            x += w + HUECO
    return res


botones = construir_teclado()
CENTROS_BOTONES = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in botones]
INDICE = {b[4]: i for i, b in enumerate(botones)}
ALTO_TECLA = botones[0][3] - botones[0][1]


def puntos_calibracion():
    """5 columnas x 4 filas, justo sobre las filas del teclado, en orden serpiente."""
    ys = sorted({round((b[1] + b[3]) / 2) for b in botones})
    xs = [0.07, 0.285, 0.5, 0.715, 0.93]
    pts = []
    for i, y in enumerate(ys):
        fila = xs if i % 2 == 0 else xs[::-1]
        pts += [(x, y / ALTO) for x in fila]
    return pts


PUNTOS_CALIBRACION = puntos_calibracion()
# Teclas que se miran en la fase de validación (reparto por todo el teclado)
TECLAS_VALIDACION = ["q", "p", "g", "a", "ñ", "z", "m", "ESPACIO", "ENTER", "IZQ"]


def dist_a_tecla(p, b):
    dx = max(b[0] - p[0], 0, p[0] - b[2])
    dy = max(b[1] - p[1], 0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def tecla_cercana(p, margen):
    mejor, d_mejor = None, 1e9
    for i, b in enumerate(botones):
        d = dist_a_tecla(p, b)
        if d < d_mejor:
            mejor, d_mejor = i, d
    return mejor if d_mejor <= margen else None


# ====================== Editor de texto ======================
class Editor:
    def __init__(self):
        self.texto = ""
        self.c = 0              # posición del cursor de texto (antes del carácter c)
        self.scroll = 0

    def pulsar(self, accion):
        if accion == "IZQ":
            self.c = max(0, self.c - 1)
        elif accion == "DER":
            self.c = min(len(self.texto), self.c + 1)
        elif accion == "BORRAR":
            if self.c > 0:
                self.texto = self.texto[:self.c - 1] + self.texto[self.c:]
                self.c -= 1
        else:
            ch = {"ENTER": "\n", "ESPACIO": " "}.get(accion, accion)
            self.texto = self.texto[:self.c] + ch + self.texto[self.c:]
            self.c += 1


# ====================== Dibujo ======================
_cache_tam = {}


def tam_char(ch, escala, grosor):
    base = {"ñ": "n", "Ñ": "N"}.get(ch, ch)
    k = (base, escala, grosor)
    if k not in _cache_tam:
        (tw, th), _ = cv2.getTextSize(base, FUENTE, escala, grosor)
        _cache_tam[k] = (tw, th)
    return _cache_tam[k]


def poner_char(l, ch, x, y_base, escala, color, grosor):
    """putText de OpenCV no tiene Ñ: se dibuja la N y se le añade la tilde a mano."""
    base = {"ñ": "n", "Ñ": "N"}.get(ch, ch)
    if base != " ":
        cv2.putText(l, base, (int(x), int(y_base)), FUENTE, escala, color, grosor, cv2.LINE_AA)
    if ch in ("ñ", "Ñ"):
        tw, th = tam_char(ch, escala, grosor)
        t = np.linspace(-np.pi, np.pi, 12)
        xs = x + tw / 2 + t / np.pi * 0.32 * tw
        ys = y_base - th - 0.14 * th - np.sin(t) * 0.08 * th
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(l, [pts], False, color, max(1, grosor - 1), cv2.LINE_AA)


def dibujar_etiqueta(l, boton, color):
    x1, y1, x2, y2, nombre = boton
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if nombre in ("IZQ", "DER"):
        d = -1 if nombre == "IZQ" else 1
        largo = min(w * 0.30, h * 0.35)
        cv2.arrowedLine(l, (int(cx - d * largo), cy), (int(cx + d * largo), cy), color,
                        max(3, int(h * 0.04)), cv2.LINE_AA, tipLength=0.45)
        return
    texto = nombre.upper()
    (tw1, th1), _ = cv2.getTextSize(texto, FUENTE, 1.0, 1)
    factor_alto = 0.36 if len(texto) == 1 else 0.20
    escala = min(factor_alto * h / th1, 0.80 * w / tw1)
    grosor = max(2, int(escala * 1.8))
    if nombre == "ñ":
        tw, th = tam_char("Ñ", escala, grosor)
        poner_char(l, "Ñ", cx - tw / 2, cy + th / 2, escala, color, grosor)
    else:
        (tw, th), _ = cv2.getTextSize(texto, FUENTE, escala, grosor)
        cv2.putText(l, texto, (cx - tw // 2, cy + th // 2), FUENTE, escala, color, grosor, cv2.LINE_AA)


def construir_base():
    b = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    cv2.rectangle(b, (MARGEN, CAJA_TOP), (ANCHO - MARGEN, ALTO_TEXTO), (80, 80, 80), 2)
    for bt in botones:
        cv2.rectangle(b, (bt[0], bt[1]), (bt[2], bt[3]), (100, 100, 100), 2)
        dibujar_etiqueta(b, bt, (175, 175, 175))
    return b


def dibujar_texto(l, editor, ahora):
    escala = 1.3 * ESC
    grosor = max(2, int(2 * ESC))
    pad = int(14 * ESC)
    lh = int(escala * 34)
    x0 = MARGEN + pad
    ancho_max = ANCHO - 2 * (MARGEN + pad)
    max_lineas = max(1, (ALTO_TEXTO - CAJA_TOP - 2 * pad) // lh)

    # Disposición del texto en líneas (salto por Enter y por ancho)
    lineas, pos = [[]], []
    ln, x = 0, 0
    for ch in editor.texto:
        pos.append((ln, x))
        if ch == "\n":
            ln += 1
            x = 0
            lineas.append([])
            continue
        w = tam_char(ch, escala, grosor)[0]
        if x + w > ancho_max:
            ln += 1
            x = 0
            lineas.append([])
            pos[-1] = (ln, 0)
        lineas[ln].append((ch, x))
        x += w
    pos.append((ln, x))

    c_ln, c_x = pos[editor.c]
    if c_ln < editor.scroll:
        editor.scroll = c_ln
    elif c_ln >= editor.scroll + max_lineas:
        editor.scroll = c_ln - max_lineas + 1

    for i in range(max_lineas):
        idx = editor.scroll + i
        if idx >= len(lineas):
            break
        y_base = CAJA_TOP + pad + i * lh + int(lh * 0.78)
        for ch, cx in lineas[idx]:
            poner_char(l, ch, x0 + cx, y_base, escala, (240, 240, 240), grosor)

    if int(ahora * 2) % 2 == 0:       # cursor de texto parpadeante
        y_base = CAJA_TOP + pad + (c_ln - editor.scroll) * lh + int(lh * 0.78)
        xx = int(x0 + c_x)
        cv2.line(l, (xx, y_base - int(lh * 0.72)), (xx, y_base + int(lh * 0.12)), (0, 255, 255), max(2, grosor))


# ====================== Filtro One Euro ======================
class OneEuro:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class FiltroFijacion:
    """Congela el cursor en el centroide de la fijación mientras la mirada no salga del radio."""

    def __init__(self, radio, frames_salida=FIJACION_FRAMES_SALIDA, max_puntos=FIJACION_MAX_PUNTOS):
        self.radio, self.frames_salida, self.max_puntos = radio, frames_salida, max_puntos
        self.pts = []
        self.fuera = 0

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if not self.pts:
            self.pts = [p]
            return p
        centro = np.mean(self.pts, axis=0)
        if np.linalg.norm(p - centro) <= self.radio:
            self.fuera = 0
            self.pts.append(p)
            if len(self.pts) > self.max_puntos:
                self.pts.pop(0)
            return np.mean(self.pts, axis=0)
        self.fuera += 1
        if self.fuera >= self.frames_salida:
            self.pts, self.fuera = [p], 0
            return p
        return centro


# ====================== Extraccion de rasgos ======================
def rasgos_ojo(lm, ojo, w, h):
    """Posición del iris (u, v) y apertura, en el sistema de coordenadas del ojo."""
    iris, ca, cb, sup1, sup2, inf1, inf2 = ojo

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    a, b = P(ca), P(cb)
    if a[0] > b[0]:
        a, b = b, a
    eje = b - a
    ancho = np.linalg.norm(eje)
    ex = eje / ancho
    ey = np.array([-ex[1], ex[0]])
    centro = (a + b) / 2

    d = P(iris) - centro
    u = float(d @ ex) / ancho
    v = float(d @ ey) / ancho

    sup = (P(sup1) + P(sup2)) / 2
    inf = (P(inf1) + P(inf2)) / 2
    apertura = float((inf - sup) @ ey) / ancho
    return u, v, apertura


def mirada_blendshapes(res):
    if not res.face_blendshapes:
        return None
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
    try:
        bh = ((bs["eyeLookOutLeft"] + bs["eyeLookInRight"])
              - (bs["eyeLookInLeft"] + bs["eyeLookOutRight"])) / 2
        bv = ((bs["eyeLookUpLeft"] + bs["eyeLookUpRight"])
              - (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"])) / 2
    except KeyError:
        return None
    return bh, bv


def pose_cabeza(res):
    if not res.facial_transformation_matrixes:
        return None
    M = np.array(res.facial_transformation_matrixes[0])
    R = M[:3, :3] / np.linalg.norm(M[:3, 0])
    return R[0, 2], R[1, 2], M[0, 3], M[1, 3]


class Rastreador:
    def __init__(self, cap):
        self.cap = cap
        opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=USAR_BLENDSHAPES,
            output_facial_transformation_matrixes=USAR_POSE_CABEZA,
        )
        self.detector = vision.FaceLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.last_ts = -1
        self.bloqueo = 0

    def leer(self):
        """Devuelve (vector_de_rasgos | None, valido)."""
        ok, frame = self.cap.read()
        if not ok:
            return None, False
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.monotonic() - self.t0) * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts

        res = self.detector.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None, False

        lm = res.face_landmarks[0]
        ua, va, apa = rasgos_ojo(lm, OJO_A, w, h)
        ub, vb, apb = rasgos_ojo(lm, OJO_B, w, h)
        f = {"ap": (apa + apb) / 2}
        if OJOS_POR_SEPARADO:
            f.update(ua=ua, va=va, ub=ub, vb=vb)
        else:
            f.update(u=(ua + ub) / 2, v=(va + vb) / 2)

        if USAR_BLENDSHAPES:
            bs = mirada_blendshapes(res)
            if bs is None:
                return None, False
            f["bh"], f["bv"] = bs
        if USAR_POSE_CABEZA:
            pose = pose_cabeza(res)
            if pose is None:
                return None, False
            f["fx"], f["fy"], f["tx"], f["ty"] = pose

        if f["ap"] < UMBRAL_PARPADEO:
            self.bloqueo = FRAMES_BLOQUEO
            valido = False
        elif self.bloqueo > 0:
            self.bloqueo -= 1
            valido = False
        else:
            valido = True

        return np.array([f[k] for k in CLAVES]), valido


# ====================== Modelos ======================
TIPOS = ("lineal", "poly2", "poly3", "kernel")
ALPHAS = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def _rejilla(tipo):
    if tipo == "kernel":
        return [dict(alpha=1.0, gamma=g, alpha_k=a) for g in (0.1, 0.3, 1.0) for a in (0.03, 0.3, 3.0)]
    return [dict(alpha=a) for a in ALPHAS]


class Regresor:
    """Trabaja con los rasgos ya centrados y escalados (Fs). Las primeras N_OJO columnas son del ojo."""

    def __init__(self, tipo, alpha=1.0, gamma=0.3, alpha_k=0.3):
        self.tipo, self.alpha, self.gamma, self.alpha_k = tipo, alpha, gamma, alpha_k

    def _X(self, Fs):
        if self.tipo in ("poly2", "poly3"):
            return np.hstack([self.poly.transform(Fs[:, :N_OJO]), Fs[:, N_OJO:]])
        return Fs

    def fit(self, Fs, Yn):
        if self.tipo in ("poly2", "poly3"):
            self.poly = PolynomialFeatures(degree=int(self.tipo[-1]), include_bias=False) \
                .fit(np.zeros((1, N_OJO)))
        X = self._X(Fs)
        self.ridge = Ridge(alpha=self.alpha).fit(X, Yn)
        if self.tipo == "kernel":
            # Base lineal + corrección no lineal local sobre lo que la base no explica.
            # Lejos de los datos de calibración la corrección tiende a 0 (no extrapola a lo loco).
            res = Yn - self.ridge.predict(X)
            self.krr = KernelRidge(kernel="rbf", gamma=self.gamma, alpha=self.alpha_k) \
                .fit(Fs[:, :N_OJO], res)
        return self

    def predict(self, Fs):
        p = self.ridge.predict(self._X(Fs))
        if self.tipo == "kernel":
            p = p + self.krr.predict(Fs[:, :N_OJO])
        return p


class ModeloMirada:
    def fit(self, F, Y, grupos, tipo=MODELO):
        self.media = F.mean(axis=0)
        ojo = F[:, :N_OJO]
        lo, hi = np.percentile(ojo, 1, axis=0), np.percentile(ojo, 99, axis=0)
        margen = 0.15 * (hi - lo)
        self.lo, self.hi = lo - margen, hi + margen

        Fs = (F - self.media) / ESCALAS
        Yn = Y / PANTALLA
        cv = GroupKFold(n_splits=min(5, len(np.unique(grupos))))

        self.tabla = {}
        mejor = None
        for t in (TIPOS if tipo == "auto" else (tipo,)):
            mejor_t = None
            for params in _rejilla(t):
                errs = []
                for tr, te in cv.split(Fs, Yn, grupos):
                    m = Regresor(t, **params).fit(Fs[tr], Yn[tr])
                    errs.append(np.linalg.norm((m.predict(Fs[te]) - Yn[te]) * PANTALLA, axis=1))
                err = float(np.concatenate(errs).mean())
                if mejor_t is None or err < mejor_t[0]:
                    mejor_t = (err, params)
            self.tabla[t] = mejor_t
            if mejor is None or mejor_t[0] < mejor[0]:
                mejor = (mejor_t[0], t, mejor_t[1])
        self.cv_error, self.tipo, self.params = mejor
        self.reg = Regresor(self.tipo, **self.params).fit(Fs, Yn)
        return self

    def predecir(self, f):
        f = np.array(f, dtype=float)
        f[:N_OJO] = np.clip(f[:N_OJO], self.lo, self.hi)
        Fs = ((f - self.media) / ESCALAS).reshape(1, -1)
        p = self.reg.predict(Fs)[0] * PANTALLA
        return np.clip(p, 0, PANTALLA)


# ====================== Calibracion ======================
def lienzo_punto(px, py, color, radio, texto):
    l = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    cv2.circle(l, (px, py), int(radio * ESC), color, -1)
    cv2.putText(l, texto, (40, 50), FUENTE, 1.2 * ESC, (255, 255, 255), 2)
    return l


def quitar_atipicos(m, z_max=3.5):
    med = np.median(m, axis=0)
    mad = np.maximum(np.median(np.abs(m - med), axis=0) * 1.4826, 0.05 * ESCALAS)
    z = np.abs(m - med) / mad
    ok = (z[:, :N_OJO] < z_max).all(axis=1)
    return m[ok] if ok.sum() >= len(m) // 2 else m


def tecla_salir(k):
    return k == ord("q") or k == 27


def recoger_punto(rastreador, px, py, n_muestras=MUESTRAS_POR_PUNTO, previos=SEGUNDOS_PREVIOS, info=""):
    t_fin = time.monotonic() + previos
    while True:
        resto = t_fin - time.monotonic()
        if resto <= 0:
            break
        rastreador.cap.grab()
        cv2.imshow(VENTANA, lienzo_punto(px, py, (0, 0, 255), 25,
                                         f"{info} Mira el punto... {int(np.ceil(resto))}"))
        if tecla_salir(cv2.waitKey(1) & 0xFF):
            raise Salir

    muestras, leidos = [], 0
    while len(muestras) < n_muestras:
        feats, valido = rastreador.leer()
        leidos += 1
        progreso = len(muestras) / n_muestras
        msg = f"{info} Grabando..." if feats is not None else f"{info} No detecto tu cara"
        cv2.imshow(VENTANA, lienzo_punto(px, py, (0, 255, 0), 25 - 15 * progreso, msg))
        if tecla_salir(cv2.waitKey(1) & 0xFF):
            raise Salir
        if valido and leidos > MUESTRAS_DESCARTADAS:
            muestras.append(feats)
    return quitar_atipicos(np.array(muestras))


def calibrar(rastreador):
    F, Y, G = [], [], []
    for k, (rx, ry) in enumerate(PUNTOS_CALIBRACION):
        px, py = int(rx * ANCHO), int(ry * ALTO)
        m = recoger_punto(rastreador, px, py, info=f"[{k + 1}/{len(PUNTOS_CALIBRACION)}]")
        F.append(m)
        Y.append(np.tile([px, py], (len(m), 1)))
        G.append(np.full(len(m), k))
    return np.vstack(F), np.vstack(Y).astype(float), np.concatenate(G)


def validar(rastreador, modelo):
    """Mira el centro de varias teclas y mide cuánto se equivoca el modelo (sin suavizado)."""
    print("Validacion: mira cada punto cuando aparezca...")
    sesgos, ruidos = [], []
    Fv, Yv, Gv = [], [], []
    for k, nombre in enumerate(TECLAS_VALIDACION):
        cx, cy = CENTROS_BOTONES[INDICE[nombre]]
        m = recoger_punto(rastreador, int(cx), int(cy), n_muestras=30, previos=1.3,
                          info=f"[validacion {k + 1}/{len(TECLAS_VALIDACION)}]")
        Fv.append(m)
        Yv.append(np.tile([cx, cy], (len(m), 1)))
        Gv.append(np.full(len(m), len(PUNTOS_CALIBRACION) + k))
        preds = np.array([modelo.predecir(f) for f in m])
        sesgos.append(np.linalg.norm(np.median(preds, axis=0) - (cx, cy)))
        ruidos.append(np.linalg.norm(preds.std(axis=0)))
    ancho_tecla = botones[0][2] - botones[0][0]
    print(f"  Error sistematico (sesgo): medio {np.mean(sesgos):.0f} px, maximo {np.max(sesgos):.0f} px")
    print(f"  Ruido fotograma a fotograma: medio {np.mean(ruidos):.0f} px")
    print(f"  (una tecla de letra mide {ancho_tecla}x{ALTO_TECLA} px)")
    print("  Si el sesgo es mayor que media tecla, ver los consejos de precision (camara fija, luz, distancia).")
    return float(np.mean(ruidos)), np.vstack(Fv), np.vstack(Yv).astype(float), np.concatenate(Gv)


# ====================== Modo teclado ======================
def modo_teclado(rastreador, modelo, ruido_px=None):
    """Devuelve True si el usuario pide recalibrar, False si quiere salir."""
    if FIJACION_RADIO_PX is not None:
        radio = FIJACION_RADIO_PX
    elif ruido_px is not None:
        radio = float(np.clip(2.0 * ruido_px, 0.04 * ANCHO, 0.12 * ANCHO))
    else:
        radio = 0.06 * ANCHO
    print(f"Radio de fijacion: {radio:.0f} px")

    tol = TOLERANCIA_PX_REL * ANCHO
    histeresis = HISTERESIS_PX_REL * ANCHO
    max_offset = AUTOCORR_MAX_PX_REL * ANCHO

    base = construir_base()
    editor = Editor()
    fijacion = FiltroFijacion(radio)
    filtro_x = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
    filtro_y = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
    hist = deque(maxlen=VENTANA_MEDIANA)
    filt = PANTALLA / 2                  # posición filtrada SIN corrección de deriva
    offset = np.zeros(2)                 # corrección de deriva acumulada
    muestras_tecla = []
    activo, t_entrada = None, 0.0
    sel_idx, flash_hasta = None, 0.0

    while True:
        feats, valido = rastreador.leer()
        ahora = time.monotonic()

        if valido:
            hist.append(modelo.predecir(feats))
            med = np.median(np.array(hist), axis=0)
            fij = fijacion(med)
            filt = np.array([filtro_x(fij[0], ahora), filtro_y(fij[1], ahora)])
            if activo is not None:
                muestras_tecla.append(filt.copy())
        cursor = np.clip(filt + offset, 0, PANTALLA)

        # Tecla activa con histéresis
        candidato = tecla_cercana(cursor, tol)
        if activo is not None and dist_a_tecla(cursor, botones[activo]) <= histeresis:
            candidato = activo
        if candidato != activo:
            activo, t_entrada = candidato, ahora
            muestras_tecla = []

        progreso = 0.0 if activo is None else min(1.0, max(0.0, (ahora - t_entrada) / DWELL_SEG))
        if activo is not None and progreso >= 1.0:
            accion = botones[activo][4]
            editor.pulsar(accion)
            print(f"PULSADO: {accion:8s} | texto: {editor.texto!r}")
            sel_idx, flash_hasta = activo, ahora + 0.35
            t_entrada = ahora + PAUSA_TRAS_SELECCION

            # Autocorrección: ¿dónde creía el sistema que mirabas frente al centro de la tecla?
            if AUTOCORRECCION and len(muestras_tecla) >= 10:
                objetivo = np.array(CENTROS_BOTONES[activo]) - np.mean(muestras_tecla[-20:], axis=0)
                if np.linalg.norm(objetivo) < 0.8 * ALTO_TECLA:      # descarta pulsaciones dudosas
                    offset = (1 - AUTOCORR_RITMO) * offset + AUTOCORR_RITMO * objetivo
                    n = np.linalg.norm(offset)
                    if n > max_offset:
                        offset *= max_offset / n
            muestras_tecla = []

        # ---- dibujo ----
        l = base.copy()
        dibujar_texto(l, editor, ahora)
        if activo is not None:
            x1, y1, x2, y2, _ = botones[activo]
            if activo == sel_idx and ahora < flash_hasta:
                cv2.rectangle(l, (x1, y1), (x2, y2), (0, 255, 0), -1)
                dibujar_etiqueta(l, botones[activo], (0, 0, 0))
            else:
                cv2.rectangle(l, (x1, y1), (int(x1 + (x2 - x1) * progreso), y2), (0, 110, 0), -1)
                cv2.rectangle(l, (x1, y1), (x2, y2), (0, 255, 0), 4)
                dibujar_etiqueta(l, botones[activo], (255, 255, 255))

        pos = cursor
        if IMAN_A_BOTON and activo is not None:
            pos = np.array(CENTROS_BOTONES[activo])
        cv2.circle(l, (int(pos[0]), int(pos[1])), int(14 * ESC), (0, 255, 255), -1)
        if MOSTRAR_CURSOR_REAL:
            cv2.circle(l, (int(cursor[0]), int(cursor[1])), int(6 * ESC), (255, 150, 0), -1)
        cv2.putText(l, f"q/ESC salir | c recalibrar | r reiniciar deriva | modelo: {modelo.tipo} | "
                       f"deriva: {offset[0]:+.0f},{offset[1]:+.0f} px",
                    (MARGEN, int(18 * ESC)), FUENTE, 0.5 * ESC, (120, 120, 120), 1)
        cv2.imshow(VENTANA, l)

        k = cv2.waitKey(1) & 0xFF
        if tecla_salir(k):
            return False
        if k == ord("c"):
            return True
        if k == ord("r"):
            offset = np.zeros(2)


# ====================== Programa principal ======================
def abrir_camara():
    if isinstance(CAMARA_INDEX_O_URL, int):
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(CAMARA_INDEX_O_URL, cv2.CAP_DSHOW)   # más rápido y respeta MJPG
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(CAMARA_INDEX_O_URL)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMARA_ANCHO)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMARA_ALTO)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(CAMARA_INDEX_O_URL)
    return cap


def main():
    cap = abrir_camara()
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {CAMARA_INDEX_O_URL}. Prueba otro indice (0, 1, 2, 3...).")
        return
    print(f"Camara ({CAMARA_INDEX_O_URL}): {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    print(f"Pantalla de trabajo: {ANCHO}x{ALTO} | tecla de letra: {botones[0][2] - botones[0][0]}x{ALTO_TECLA} px")
    for _ in range(FOTOGRAMAS_CALENTAMIENTO):    # deja que se asienten autoexposicion y enfoque
        cap.read()

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    if PANTALLA_COMPLETA:
        cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(VENTANA, ANCHO, ALTO)

    rastreador = Rastreador(cap)
    try:
        while True:
            F, Y, G = calibrar(rastreador)
            print(f"Calibracion terminada: {len(F)} muestras.")
            modelo = ModeloMirada().fit(F, Y, G)
            print("Comparativa de modelos (error de validacion cruzada, pesimista en los bordes):")
            for t, (e, _) in modelo.tabla.items():
                print(f"  {t:7s}: {e:6.0f} px" + ("   <-- elegido" if t == modelo.tipo else ""))
            ruido = None
            if VALIDAR:
                ruido, Fv, Yv, Gv = validar(rastreador, modelo)
                if REFINAR_CON_VALIDACION:
                    modelo = ModeloMirada().fit(np.vstack([F, Fv]), np.vstack([Y, Yv]),
                                                np.concatenate([G, Gv]), tipo=modelo.tipo)
                    print(f"Modelo refinado con las teclas ({modelo.tipo}): error CV {modelo.cv_error:.0f} px")
            if not modo_teclado(rastreador, modelo, ruido):
                break
    except Salir:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()