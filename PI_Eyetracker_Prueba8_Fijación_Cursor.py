"""
Eye tracker con webcam + MediaPipe (versión mejorada)
 
Cambios principales respecto a la versión anterior:
  1. Rasgos del ojo mejores: posición del iris horizontal Y vertical, en el sistema
     de coordenadas del propio ojo (invariante a la inclinación de la cabeza).
  2. Rasgos opcionales de pose de cabeza y de "blendshapes" de MediaPipe.
  3. Calibración más limpia: se vacía el buffer de la cámara, se descartan los primeros
     fotogramas de cada punto, se eliminan atípicos y se entrena con TODAS las muestras.
  4. Un solo modelo Ridge con rasgos escalados y alpha elegido por validación cruzada
     por puntos (ya no hay alpha forzado a mano).
  5. Fase de validación que mide el error real en píxeles (sesgo y ruido).
  6. Los parpadeos ya no mueven el cursor en modo en vivo.
  7. Suavizado: mediana + filtro One Euro (en vez de media exponencial fija).
  8. Histéresis en los botones + selección por permanencia (dwell).
 
Teclas en modo cursor:  q = salir,  c = recalibrar
"""
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
 
# ====================== Configuracion ======================
ANCHO, ALTO = 1280, 720            # tamaño de la ventana de trabajo
CAMARA_ANCHO, CAMARA_ALTO = 1280, 720   # más resolución = iris más preciso
 
SEGUNDOS_PREVIOS = 2.0             # tiempo para localizar y fijar el punto rojo
MUESTRAS_POR_PUNTO = 40            # muestras válidas por punto de calibración
MUESTRAS_DESCARTADAS = 10          # primeros fotogramas del punto (el ojo aún se asienta)
 
UMBRAL_PARPADEO = 0.15             # apertura normalizada por debajo de la cual hay parpadeo
FRAMES_BLOQUEO = 3                 # fotogramas que se ignoran tras abrir el ojo
 
VENTANA_MEDIANA = 5                # mediana de las últimas N predicciones (quita picos)
ONE_EURO_MIN_CUTOFF = 0.4          # MENOR = cursor más estable pero con más retardo
ONE_EURO_BETA = 0.002              # MAYOR = menos retardo al mover la mirada (¡en px/s, no subir mucho!)
 
# Filtro de fijación: mientras la mirada se queda dentro de un radio, el cursor se congela
# en el centro de esa fijación. Un movimiento real (sácada) lo rompe en pocos fotogramas.
FIJACION_RADIO_PX = None           # None = automático (2x el ruido medido en la validación)
FIJACION_FRAMES_SALIDA = 3         # fotogramas seguidos fuera del radio para considerar que te has movido
FIJACION_MAX_PUNTOS = 25           # puntos que se promedian en una fijación (~0,8 s)
IMAN_A_BOTON = True                # dibuja el cursor en el centro del botón activo
 
DWELL_SEG = 1.5                    # tiempo mirando un botón para "pulsarlo"
PAUSA_TRAS_SELECCION = 1.0         # pausa para evitar repeticiones inmediatas
HISTERESIS_PX = 50                 # el botón activo se mantiene hasta salir X px de su borde
 
USAR_BLENDSHAPES = True            # rasgos de mirada que calcula MediaPipe (pruébalo con True/False)
USAR_POSE_CABEZA = True            # compensar el movimiento de la cabeza
VALIDAR = True                     # medir el error tras calibrar
REFINAR_CON_VALIDACION = True      # reentrenar incluyendo las muestras de los botones (más precisión justo donde importa)
OJOS_POR_SEPARADO = False          # True = el modelo ve cada ojo por separado (u,v de cada uno) en vez de su media
 
PUNTOS_CALIBRACION = [
    (0.1, 0.2), (0.3, 0.2), (0.5, 0.2), (0.7, 0.2), (0.9, 0.2),
    (0.1, 0.5), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5),
    (0.1, 0.8), (0.3, 0.8), (0.5, 0.8), (0.7, 0.8), (0.9, 0.8),
]  # 15 puntos: 5 columnas x 3 filas
 
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
VENTANA = "EyeTracker"
PANTALLA = np.array([ANCHO, ALTO], dtype=float)
 
# Indices de landmarks: (iris, esquina_a, esquina_b, parpado_sup1, sup2, parpado_inf1, inf2)
OJO_A = (468, 33, 133, 159, 158, 145, 153)
OJO_B = (473, 362, 263, 386, 385, 374, 380)
 
# ---------- Rasgos que usa el modelo ----------
# "Ojo": entran con términos cuadráticos. "Lineales": solo lineales (más seguros).
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
 
 
# ====================== Rejilla 3x3 de botones ======================
FILAS, COLUMNAS = 3, 3
MARGEN_REJILLA = 20
ancho_boton = (ANCHO - MARGEN_REJILLA * (COLUMNAS + 1)) // COLUMNAS
alto_boton = (ALTO - MARGEN_REJILLA * (FILAS + 1)) // FILAS
NOMBRES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
 
botones = []
for fila in range(FILAS):
    for col in range(COLUMNAS):
        x1 = MARGEN_REJILLA + col * (ancho_boton + MARGEN_REJILLA)
        y1 = MARGEN_REJILLA + fila * (alto_boton + MARGEN_REJILLA)
        botones.append((x1, y1, x1 + ancho_boton, y1 + alto_boton, NOMBRES[fila * COLUMNAS + col]))
 
CENTROS_BOTONES = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in botones]
 
 
def dentro(p, boton, margen=0):
    x1, y1, x2, y2, _ = boton
    return (x1 - margen <= p[0] <= x2 + margen) and (y1 - margen <= p[1] <= y2 + margen)
 
 
# ====================== Filtro One Euro ======================
class OneEuro:
    """Filtro adaptativo: suaviza mucho cuando miras fijo y poco cuando te mueves rápido."""
 
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
        if self.fuera >= self.frames_salida:      # movimiento real: empieza otra fijación
            self.pts, self.fuera = [p], 0
            return p
        return centro                              # posible pico aislado: se ignora
 
 
# ====================== Extraccion de rasgos ======================
def rasgos_ojo(lm, ojo, w, h):
    """Posición del iris (u, v) y apertura, en el sistema de coordenadas del ojo.
 
    u: desplazamiento del iris a lo largo del eje esquina-esquina (horizontal del ojo)
    v: desplazamiento perpendicular a ese eje (vertical del ojo)
    Todo normalizado por el ancho del ojo => invariante a escala e inclinación de cabeza.
    """
    iris, ca, cb, sup1, sup2, inf1, inf2 = ojo
 
    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])
 
    a, b = P(ca), P(cb)
    if a[0] > b[0]:
        a, b = b, a
    eje = b - a
    ancho = np.linalg.norm(eje)
    ex = eje / ancho
    ey = np.array([-ex[1], ex[0]])          # perpendicular, apunta hacia abajo
    centro = (a + b) / 2
 
    d = P(iris) - centro
    u = float(d @ ex) / ancho
    v = float(d @ ey) / ancho
 
    sup = (P(sup1) + P(sup2)) / 2
    inf = (P(inf1) + P(inf2)) / 2
    apertura = float((inf - sup) @ ey) / ancho
    return u, v, apertura
 
 
def mirada_blendshapes(res):
    """Mirada horizontal/vertical según los blendshapes de MediaPipe (None si no hay)."""
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
    """(fx, fy, tx, ty): hacia dónde apunta la cara y dónde está, sin ángulos de Euler."""
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
        """Devuelve (vector_de_rasgos | None, valido).
        valido=False si hay parpadeo (o justo después) o no hay cara."""
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
 
        # Parpadeo: se invalida también unos fotogramas después (el párpado reabre y el iris es ruidoso)
        if f["ap"] < UMBRAL_PARPADEO:
            self.bloqueo = FRAMES_BLOQUEO
            valido = False
        elif self.bloqueo > 0:
            self.bloqueo -= 1
            valido = False
        else:
            valido = True
 
        return np.array([f[k] for k in CLAVES]), valido
 
 
# ====================== Modelo ======================
class ModeloMirada:
    """Ridge sobre [términos cuadráticos de los rasgos del ojo + rasgos lineales]."""
 
    ALPHAS = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
 
    def _expandir(self, F):
        Fs = (F - self.media) / ESCALAS
        return np.hstack([self.poly.transform(Fs[:, :N_OJO]), Fs[:, N_OJO:]])
 
    def fit(self, F, Y, grupos):
        self.media = F.mean(axis=0)
        self.poly = PolynomialFeatures(degree=2, include_bias=False).fit(np.zeros((1, N_OJO)))
 
        # Rango visto en calibración (para no extrapolar de forma salvaje)
        ojo = F[:, :N_OJO]
        lo, hi = np.percentile(ojo, 1, axis=0), np.percentile(ojo, 99, axis=0)
        margen = 0.15 * (hi - lo)
        self.lo, self.hi = lo - margen, hi + margen
 
        X = self._expandir(F)
        Yn = Y / PANTALLA
 
        # alpha por validación cruzada AGRUPADA por punto de calibración
        cv = GroupKFold(n_splits=min(5, len(np.unique(grupos))))
        mejor = None
        for alpha in self.ALPHAS:
            errs = []
            for tr, te in cv.split(X, Yn, grupos):
                m = Ridge(alpha=alpha).fit(X[tr], Yn[tr])
                errs.append(np.linalg.norm((m.predict(X[te]) - Yn[te]) * PANTALLA, axis=1))
            err = float(np.concatenate(errs).mean())
            if mejor is None or err < mejor[0]:
                mejor = (err, alpha)
        self.cv_error, self.alpha = mejor
        self.ridge = Ridge(alpha=self.alpha).fit(X, Yn)
        return self
 
    def predecir(self, f):
        f = np.array(f, dtype=float)
        f[:N_OJO] = np.clip(f[:N_OJO], self.lo, self.hi)
        p = self.ridge.predict(self._expandir(f.reshape(1, -1)))[0] * PANTALLA
        return np.clip(p, 0, PANTALLA)
 
 
# ====================== Calibracion ======================
def lienzo_punto(px, py, color, radio, texto):
    l = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    cv2.circle(l, (px, py), radio, color, -1)
    cv2.putText(l, texto, (40, 40), FUENTE, 1, (255, 255, 255), 2)
    return l
 
 
def quitar_atipicos(m, z_max=3.5):
    med = np.median(m, axis=0)
    mad = np.maximum(np.median(np.abs(m - med), axis=0) * 1.4826, 0.05 * ESCALAS)
    z = np.abs(m - med) / mad
    ok = (z[:, :N_OJO] < z_max).all(axis=1)
    return m[ok] if ok.sum() >= len(m) // 2 else m
 
 
def recoger_punto(rastreador, px, py, n_muestras=MUESTRAS_POR_PUNTO, previos=SEGUNDOS_PREVIOS):
    # 1) Punto rojo: el usuario lo localiza y lo fija. Se sigue leyendo (grab) la cámara
    #    para que no se acumulen fotogramas viejos en el buffer.
    t_fin = time.monotonic() + previos
    while True:
        resto = t_fin - time.monotonic()
        if resto <= 0:
            break
        rastreador.cap.grab()
        cv2.imshow(VENTANA, lienzo_punto(px, py, (0, 0, 255), 25,
                                         f"Mira el punto... {int(np.ceil(resto))}"))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise Salir
 
    # 2) Punto verde que se encoge (ayuda a mantener la mirada en el centro)
    muestras, leidos = [], 0
    while len(muestras) < n_muestras:
        feats, valido = rastreador.leer()
        leidos += 1
        progreso = len(muestras) / n_muestras
        msg = "Grabando..." if feats is not None else "No detecto tu cara"
        cv2.imshow(VENTANA, lienzo_punto(px, py, (0, 255, 0), int(25 - 15 * progreso), msg))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise Salir
        if valido and leidos > MUESTRAS_DESCARTADAS:
            muestras.append(feats)
    return quitar_atipicos(np.array(muestras))
 
 
def calibrar(rastreador):
    F, Y, G = [], [], []
    for k, (rx, ry) in enumerate(PUNTOS_CALIBRACION):
        px, py = int(rx * ANCHO), int(ry * ALTO)
        m = recoger_punto(rastreador, px, py)
        F.append(m)
        Y.append(np.tile([px, py], (len(m), 1)))
        G.append(np.full(len(m), k))
    return np.vstack(F), np.vstack(Y).astype(float), np.concatenate(G)
 
 
def validar(rastreador, modelo):
    """Mira el centro de cada botón y mide cuánto se equivoca el modelo (sin suavizado)."""
    print("Validacion: mira cada punto cuando aparezca...")
    sesgos, ruidos = [], []
    Fv, Yv, Gv = [], [], []
    for k, (cx, cy) in enumerate(CENTROS_BOTONES):
        m = recoger_punto(rastreador, int(cx), int(cy), n_muestras=30, previos=1.5)
        Fv.append(m)
        Yv.append(np.tile([cx, cy], (len(m), 1)))
        Gv.append(np.full(len(m), len(PUNTOS_CALIBRACION) + k))
        preds = np.array([modelo.predecir(f) for f in m])
        sesgos.append(np.linalg.norm(np.median(preds, axis=0) - (cx, cy)))
        ruidos.append(np.linalg.norm(preds.std(axis=0)))
    print(f"  Error sistematico (sesgo): medio {np.mean(sesgos):.0f} px, maximo {np.max(sesgos):.0f} px")
    print(f"  Ruido fotograma a fotograma: medio {np.mean(ruidos):.0f} px")
    print(f"  (un boton mide {ancho_boton}x{alto_boton} px)")
    print("  Sesgo alto => modelo/rasgos/calibracion.  Ruido alto => mas suavizado o mejor luz/camara.")
    return float(np.mean(ruidos)), np.vstack(Fv), np.vstack(Yv).astype(float), np.concatenate(Gv)
 
 
# ====================== Modo cursor ======================
def texto_centrado(l, texto, caja, escala, color, grosor):
    x1, y1, x2, y2 = caja
    (tw, th), _ = cv2.getTextSize(texto, FUENTE, escala, grosor)
    cv2.putText(l, texto, ((x1 + x2 - tw) // 2, (y1 + y2 + th) // 2), FUENTE, escala, color, grosor)
 
 
def modo_cursor(rastreador, modelo, ruido_px=None):
    """Devuelve True si el usuario pide recalibrar, False si quiere salir."""
    if FIJACION_RADIO_PX is not None:
        radio = FIJACION_RADIO_PX
    elif ruido_px is not None:
        radio = float(np.clip(2.0 * ruido_px, 50, 150))
    else:
        radio = 80.0
    print(f"Radio de fijacion: {radio:.0f} px")
    fijacion = FiltroFijacion(radio)
    filtro_x = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
    filtro_y = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
    hist = deque(maxlen=VENTANA_MEDIANA)
    cursor = PANTALLA / 2
    activo, t_entrada = None, 0.0
    sel_idx, flash_hasta, ultimo = None, 0.0, "-"
 
    while True:
        feats, valido = rastreador.leer()
        ahora = time.monotonic()
 
        # En parpadeo (o sin cara) el cursor se queda quieto
        if valido:
            hist.append(modelo.predecir(feats))
            med = np.median(np.array(hist), axis=0)
            fij = fijacion(med)
            cursor = np.array([filtro_x(fij[0], ahora), filtro_y(fij[1], ahora)])
 
        # Boton activo con histeresis: una vez dentro, hay que salir "con margen" para soltarlo
        candidato = next((i for i, b in enumerate(botones) if dentro(cursor, b)), None)
        if activo is not None and dentro(cursor, botones[activo], HISTERESIS_PX):
            candidato = activo
        if candidato != activo:
            activo, t_entrada = candidato, ahora
 
        progreso = 0.0 if activo is None else min(1.0, max(0.0, (ahora - t_entrada) / DWELL_SEG))
        if activo is not None and progreso >= 1.0:
            ultimo = botones[activo][4]
            print(f"SELECCIONADO: {ultimo}")
            sel_idx, flash_hasta = activo, ahora + 0.4
            t_entrada = ahora + PAUSA_TRAS_SELECCION
 
        # ---- dibujo ----
        l = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
        for i, (x1, y1, x2, y2, nombre) in enumerate(botones):
            if i == sel_idx and ahora < flash_hasta:
                cv2.rectangle(l, (x1, y1), (x2, y2), (0, 255, 0), -1)
                texto_centrado(l, nombre, (x1, y1, x2, y2), 3, (0, 0, 0), 5)
            elif i == activo:
                cv2.rectangle(l, (x1, y1), (int(x1 + (x2 - x1) * progreso), y2), (0, 110, 0), -1)
                cv2.rectangle(l, (x1, y1), (x2, y2), (0, 255, 0), 4)
                texto_centrado(l, nombre, (x1, y1, x2, y2), 3, (255, 255, 255), 5)
            else:
                cv2.rectangle(l, (x1, y1), (x2, y2), (100, 100, 100), 3)
                texto_centrado(l, nombre, (x1, y1, x2, y2), 3, (150, 150, 150), 3)
 
        pos = cursor
        if IMAN_A_BOTON and activo is not None:
            pos = np.array(CENTROS_BOTONES[activo])
        cv2.circle(l, (int(pos[0]), int(pos[1])), 15, (0, 255, 255), -1)
        cv2.putText(l, f"q salir | c recalibrar | ultimo: {ultimo}", (30, ALTO - 30),
                    FUENTE, 0.6, (120, 120, 120), 1)
        cv2.imshow(VENTANA, l)
 
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            return False
        if k == ord("c"):
            return True
 
 
# ====================== Programa principal ======================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # permite 30 fps a alta resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMARA_ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMARA_ALTO)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Camara: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
 
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, ANCHO, ALTO)
 
    rastreador = Rastreador(cap)
    try:
        while True:
            F, Y, G = calibrar(rastreador)
            print(f"Calibracion terminada: {len(F)} muestras.")
            modelo = ModeloMirada().fit(F, Y, G)
            print(f"Alpha elegido: {modelo.alpha} | error de validacion cruzada: "
                  f"{modelo.cv_error:.0f} px (estimacion pesimista: aqui se extrapolan puntos de borde)")
            ruido = None
            if VALIDAR:
                ruido, Fv, Yv, Gv = validar(rastreador, modelo)   # el error impreso es el HONESTO (antes de refinar)
                if REFINAR_CON_VALIDACION:
                    modelo = ModeloMirada().fit(np.vstack([F, Fv]), np.vstack([Y, Yv]),
                                                np.concatenate([G, Gv]))
                    print(f"Modelo refinado con los botones: alpha {modelo.alpha}, "
                          f"error CV {modelo.cv_error:.0f} px")
            if not modo_cursor(rastreador, modelo, ruido):
                break
    except Salir:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()