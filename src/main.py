import csv
import os
import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = "assets"
WINDOW_NAME = "Provador Virtual - TCC UNIP"

FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.02
FILTER_D_CUTOFF = 1.0

PITCH_CALIB_FRAMES = 30
MEASURE_DIR = "medicoes"
CHECK_DIR = "conferencia"

HOLD_SECONDS = 0.3
ALPHA_THRESHOLD = 10
YAW_HIDE = 0.3
MIN_PERSP = 0.35

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

CATEGORIES = ["chapeus", "oculos", "brincos", "piercings"]
CATEGORY_LABELS = {
    "chapeus": "Chapéus",
    "oculos": "Óculos",
    "brincos": "Brincos",
    "piercings": "Piercings",
}

ACCENT = (255, 181, 71)
ACCENT_TEXT = (26, 20, 8)
PANEL = (14, 14, 16, 210)
TEXT = (245, 245, 247, 255)
TEXT_MUTED = (199, 199, 204, 255)
OK_COLOR = (74, 222, 128)
WARN_COLOR = (251, 191, 36)

FONT_REGULAR = [
    os.path.join(ASSETS_DIR, "fonts", "Manrope-Medium.ttf"),
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_BOLD = [
    os.path.join(ASSETS_DIR, "fonts", "Manrope-Bold.ttf"),
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

FIT_RULES = {
    "chapeu": {
        "anchor": "head_top",
        "pivot": (0.5, 0.90),
        "width_ref": "face_width",
        "width_factor": 1.9,
        "persp_x": True,
        "persp_y": False,
    },
    "oculos": {
        "anchor": "eyes",
        "pivot": (0.5, 0.45),
        "width_ref": "temple_width",
        "width_factor": 1.05,
        "persp_x": True,
        "persp_y": True,
    },
    "brinco": {
        "anchor": "earlobes",
        "pivot": (0.5, 0.0),
        "width_ref": "face_width",
        "width_factor": 0.09,
        "persp_x": False,
        "persp_y": False,
    },
    "piercing_septo": {
        "anchor": "septum",
        "pivot": (0.5, 0.10),
        "width_ref": "nose_width",
        "width_factor": 0.30,
        "persp_x": False,
        "persp_y": False,
    },
}

ACCESSORIES = {

    "bone":           {"file": "bone.png",           "nome": "Boné",           "categoria": "chapeus",   "tipo": "chapeu", "pivot": (0.5, 1.0), "width_factor": 1.15, "anchor_shift": 0.25},
    "boina":          {"file": "chapeu.png",         "nome": "Boina",          "categoria": "chapeus",   "tipo": "chapeu", "pivot": (0.5, 0.88), "width_factor": 1.15, "anchor_shift": 0.12},

    "oculos_aviador": {"file": "oculo2.png",         "nome": "Aviador",        "categoria": "oculos",    "tipo": "oculos", "pivot": (0.5, 0.5),  "width_factor": 1.15, "remove_holes": True},
    "oculos_sol":     {"file": "oculos.png",         "nome": "Óculos de sol",  "categoria": "oculos",    "tipo": "oculos", "pivot": (0.5, 0.48), "width_factor": 1.1},
    "oculos_gatinho": {"file": "oculos3.png",        "nome": "Gatinho",        "categoria": "oculos",    "tipo": "oculos", "pivot": (0.5, 0.5),  "width_factor": 1.1},

    "brinco_delicado":{"file": "brinco.png",         "nome": "Delicado",       "categoria": "brincos",   "tipo": "brinco", "width_factor": 0.07, "remove_holes": True},
    "brinco_argola":  {"file": "brinco_perola.png",  "nome": "Argola dourada", "categoria": "brincos",   "tipo": "brinco", "width_factor": 0.10, "remove_holes": True},
    "brinco_perola":  {"file": "perola.png",         "nome": "Pérola",         "categoria": "brincos",   "tipo": "brinco", "width_factor": 0.08},

    "piercing_septo": {"file": "piercing_septo.png", "nome": "Septo rosé",     "categoria": "piercings", "tipo": "piercing_septo", "pivot": (0.5, 0.0), "width_factor": 0.22, "crop_top": 0.3},
}

OVERRIDABLE = ("anchor", "pivot", "width_ref", "width_factor", "persp_x", "persp_y", "anchor_shift")

L_EYE_OUT, L_EYE_IN = 33, 133
R_EYE_IN, R_EYE_OUT = 362, 263
FOREHEAD = 10
CHIN = 152
FACE_L, FACE_R = 234, 454
TEMPLE_L, TEMPLE_R = 127, 356
LOBE_L, LOBE_R = 132, 361
NOSE_TIP = 1
COLUMELLA = 94
ALA_L, ALA_R = 129, 358

def ensure_bgra(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def trim_transparent(img):
    ys, xs = np.where(img[:, :, 3] > ALPHA_THRESHOLD)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")

def find_asset(file_name):
    path = os.path.join(ASSETS_DIR, file_name)
    if os.path.exists(path):
        return path
    base = os.path.splitext(path)[0]
    for ext in IMAGE_EXTS:
        for candidate in (base + ext, base + ext.upper()):
            if os.path.exists(candidate):
                return candidate
    return None

def remove_background(img, tol=18, remove_holes=False):
    bgr = np.ascontiguousarray(img[:, :, :3])
    h, w = bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if mask[y + 1, x + 1] == 0:
            cv2.floodFill(bgr.copy(), mask, (x, y), 0, (tol,) * 3, (tol,) * 3, flags)
    bg = mask[1:-1, 1:-1] > 0

    if remove_holes and bg.any():
        bg_color = np.median(bgr[bg], axis=0)
        same = (np.abs(bgr.astype(np.int16) - bg_color).max(axis=2) <= tol).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(same, connectivity=4)
        min_area = 0.005 * h * w
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                bg |= labels == i

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    out = img.copy()
    out[:, :, 3] = alpha
    return out

def keep_single_earring(img):
    cols = (img[:, :, 3] > ALPHA_THRESHOLD).any(axis=0)
    w = img.shape[1]
    lo, hi = int(w * 0.25), int(w * 0.75)
    empty = np.where(~cols[lo:hi])[0]
    if len(empty) == 0:
        return img
    split = lo + int(empty[len(empty) // 2])
    return trim_transparent(img[:, :split])

def load_accessories():
    loaded = {}
    for key, cfg in ACCESSORIES.items():
        path = find_asset(cfg["file"])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
        if img is None:
            print(f"[AVISO] Nao encontrei '{cfg['file']}' em '{ASSETS_DIR}/'. '{key}' sera ignorado.")
            continue

        img = ensure_bgra(img)
        if img[:, :, 3].min() >= 250:
            img = remove_background(img, remove_holes=cfg.get("remove_holes", False))
            print(f"[AVISO] '{os.path.basename(path)}' nao tem fundo transparente: removi o fundo "
                  f"automaticamente. Confira se ficou bom; se nao, use um PNG transparente.")

        img = trim_transparent(img)
        if cfg.get("crop_top"):

            img = trim_transparent(img[int(img.shape[0] * cfg["crop_top"]):])
        if cfg["tipo"] == "brinco":
            before = img.shape[1]
            img = keep_single_earring(img)
            if img.shape[1] < before:
                print(f"[OK] '{os.path.basename(path)}' tinha um par de brincos: usando so um.")

        loaded[key] = img
        print(f"[OK] {key}: {img.shape[1]}x{img.shape[0]} px")
    return loaded

def build_category_map(loaded):
    mapping = {cat: [] for cat in CATEGORIES}
    for key, cfg in ACCESSORIES.items():
        if key in loaded and cfg["categoria"] in mapping:
            mapping[cfg["categoria"]].append(key)
    return mapping

def get_fit(key):
    cfg = ACCESSORIES[key]
    fit = dict(FIT_RULES[cfg["tipo"]])
    fit.update({k: cfg[k] for k in OVERRIDABLE if k in cfg})
    return fit

class OneEuroFilter:

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x

        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

def landmarks_to_array(face_landmarks, frame_w, frame_h):
    return np.array(
        [(p.x * frame_w, p.y * frame_h, p.z * frame_w) for p in face_landmarks.landmark],
        dtype=np.float32,
    )

def compute_face(pts):

    def p2(i):
        return pts[i, :2]

    def p3(i):
        return pts[i]

    left_eye = (p2(L_EYE_OUT) + p2(L_EYE_IN)) / 2
    right_eye = (p2(R_EYE_IN) + p2(R_EYE_OUT)) / 2

    swap = right_eye[0] < left_eye[0]
    if swap:
        left_eye, right_eye = right_eye, left_eye

    def pair(a, b, dims=2):
        pa, pb = pts[a, :dims], pts[b, :dims]
        return (pb, pa) if swap else (pa, pb)

    face_l, face_r = pair(FACE_L, FACE_R)
    lobe_l, lobe_r = pair(LOBE_L, LOBE_R)
    nose_tip = p2(NOSE_TIP)

    eye_vec = right_eye - left_eye
    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    rad = np.radians(roll)
    right_dir = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)

    e3 = p3(R_EYE_OUT) - p3(L_EYE_OUT)
    yaw = float(np.degrees(np.arctan2(abs(e3[2]), np.hypot(e3[0], e3[1]))))

    v3 = p3(CHIN) - p3(FOREHEAD)
    pitch_raw = float(np.degrees(np.arctan2(v3[2], np.hypot(v3[0], v3[1]))))

    measures = {
        "face_width": float(np.linalg.norm(p3(FACE_R) - p3(FACE_L))),
        "temple_width": float(np.linalg.norm(p3(TEMPLE_R) - p3(TEMPLE_L))),
        "nose_width": float(np.linalg.norm(p3(ALA_R) - p3(ALA_L))),
    }

    down_dir = np.array([-np.sin(rad), np.cos(rad)], dtype=np.float32)
    face_h = float(np.linalg.norm(p2(CHIN) - p2(FOREHEAD)))

    lobe_push = right_dir * measures["face_width"] * 0.02
    lobe_lift = down_dir * face_h * 0.02
    anchors = {
        "eyes": (left_eye + right_eye) / 2,
        "head_top": p2(FOREHEAD),
        "septum": p2(COLUMELLA),
        "earlobe_left": lobe_l - lobe_push - lobe_lift,
        "earlobe_right": lobe_r + lobe_push - lobe_lift,
    }

    span = float(np.dot(face_r - face_l, right_dir))
    yaw_ratio = float(np.dot(nose_tip - face_l, right_dir) / span) if span > 1 else 0.5

    return {
        "roll": roll,
        "yaw": yaw,
        "pitch_raw": pitch_raw,
        "pitch": pitch_raw,
        "measures": measures,
        "anchors": anchors,
        "yaw_ratio": yaw_ratio,
        "down_dir": down_dir,
        "face_h": face_h,
    }

def get_placements(fit, face):
    a = face["anchors"]
    if fit["anchor"] == "earlobes":
        placements = []
        if face["yaw_ratio"] > YAW_HIDE:
            placements.append((a["earlobe_left"], False))
        if face["yaw_ratio"] < 1 - YAW_HIDE:
            placements.append((a["earlobe_right"], True))
        return placements
    return [(a[fit["anchor"]], False)]

def overlay_at_pivot(frame, img, pivot, anchor, width, angle_deg,
                     squash_x=1.0, squash_y=1.0, flip=False):
    if width < 2:
        return frame

    if flip:
        img = cv2.flip(img, 1)
        pivot = (1.0 - pivot[0], pivot[1])

    h, w = img.shape[:2]
    target_w = width * squash_x
    target_h = h * (width / w) * squash_y

    if target_w < w or target_h < h:
        img = cv2.resize(
            img,
            (max(2, int(round(target_w))), max(2, int(round(target_h)))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = img.shape[:2]

    sx, sy = target_w / w, target_h / h
    rad = np.radians(angle_deg)
    c, sn = np.cos(rad), np.sin(rad)

    A = np.array([[c * sx, -sn * sy], [sn * sx, c * sy]], dtype=np.float32)
    pivot_px = np.array([pivot[0] * w, pivot[1] * h], dtype=np.float32)
    t = np.asarray(anchor[:2], dtype=np.float32) - A @ pivot_px
    M = np.hstack([A, t.reshape(2, 1)])

    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    pts = corners @ A.T + t

    fh, fw = frame.shape[:2]
    x0 = max(int(np.floor(pts[:, 0].min())), 0)
    x1 = min(int(np.ceil(pts[:, 0].max())), fw)
    y0 = max(int(np.floor(pts[:, 1].min())), 0)
    y1 = min(int(np.ceil(pts[:, 1].max())), fh)
    if x1 <= x0 or y1 <= y0:
        return frame

    M[0, 2] -= x0
    M[1, 2] -= y0
    warped = cv2.warpAffine(
        img, M, (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    blended = warped[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)
    return frame

def draw_accessory(frame, img, fit, face):
    width = face["measures"][fit["width_ref"]] * fit["width_factor"]
    squash_x = max(np.cos(np.radians(face["yaw"])), MIN_PERSP) if fit["persp_x"] else 1.0
    squash_y = max(np.cos(np.radians(face["pitch"])), MIN_PERSP) if fit["persp_y"] else 1.0

    shift = face["down_dir"] * face["face_h"] * fit.get("anchor_shift", 0.0)
    points = []
    for point, flip in get_placements(fit, face):
        point = point + shift
        points.append(point)
        frame = overlay_at_pivot(frame, img, fit["pivot"], point, width, face["roll"],
                                 squash_x=squash_x, squash_y=squash_y, flip=flip)
    return frame, points, width

def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)

class Interface:

    CIRCLE = 60
    PILL_H = 40

    def __init__(self, frame_w, frame_h, accessories, category_map):
        self.w, self.h = frame_w, frame_h
        self.category_map = category_map
        self.f_title = load_font(FONT_BOLD, 18)
        self.f_pill = load_font(FONT_BOLD, 15)
        self.f_small = load_font(FONT_REGULAR, 14)
        self.f_label = load_font(FONT_REGULAR, 14)
        self.f_label_b = load_font(FONT_BOLD, 14)
        self.thumbs = {key: self._make_thumb(img) for key, img in accessories.items()}
        self.cache_key = None
        self.regions = []
        self.hitboxes = []

    @staticmethod
    def _make_thumb(img_bgra, size=40):
        thumb = Image.fromarray(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA))
        thumb.thumbnail((size, size), Image.LANCZOS)
        return thumb

    def _text_w(self, text, font):
        left, _, right, _ = font.getbbox(text)
        return right - left

    def _build(self, current_cat, selection, status, fps_shown, extra=""):
        layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, "RGBA")
        rects, hits = [], []

        title = "Provador Virtual"
        tw = self._text_w(title, self.f_title)
        box = (24, 24, 24 + tw + 36, 24 + self.PILL_H)
        d.rounded_rectangle(box, radius=self.PILL_H // 2, fill=PANEL)
        d.text((box[0] + 18, (box[1] + box[3]) / 2), title, font=self.f_title, fill=TEXT, anchor="lm")
        rects.append(box)

        if status == "lost":
            status_txt, dot = "Procurando rosto", WARN_COLOR
        else:
            status_txt, dot = "Rosto detectado", OK_COLOR
        status_txt += f"  ·  {fps_shown} FPS{extra}"
        sw = self._text_w(status_txt, self.f_small)
        box = (self.w - 24 - (sw + 56), 24, self.w - 24, 24 + self.PILL_H)
        d.rounded_rectangle(box, radius=self.PILL_H // 2, fill=PANEL)
        cy = (box[1] + box[3]) / 2
        d.ellipse((box[0] + 18, cy - 4, box[0] + 26, cy + 4), fill=dot)
        d.text((box[0] + 36, cy), status_txt, font=self.f_small, fill=TEXT, anchor="lm")
        rects.append(box)

        cat_widths = [self._text_w(CATEGORY_LABELS[c], self.f_pill) + 40 for c in CATEGORIES]
        cats_w = sum(cat_widths) + 8 * (len(CATEGORIES) - 1)

        items = [None] + self.category_map[current_cat]
        slot = self.CIRCLE + 36
        items_w = len(items) * slot - 36

        dock_w = max(cats_w, items_w) + 56
        dock_h = 16 + self.PILL_H + 14 + self.CIRCLE + 8 + 18 + 16
        dx0 = (self.w - dock_w) / 2
        dy0 = self.h - 24 - dock_h
        dock = (dx0, dy0, dx0 + dock_w, dy0 + dock_h)

        if status == "lost":

            top, bottom = 24 + self.PILL_H + 70, dy0 - 20
            ry = max((bottom - top) / 2, 60)
            rx = ry * 0.78
            cx, cy = self.w / 2, (top + bottom) / 2
            oval = (cx - rx, cy - ry, cx + rx, cy + ry)
            for start in range(0, 360, 12):
                d.arc(oval, start, start + 7, fill=(255, 255, 255, 190), width=3)
            msg = "Encaixe seu rosto no contorno"
            mw = self._text_w(msg, self.f_pill)
            mbox = (cx - mw / 2 - 20, oval[1] - 56, cx + mw / 2 + 20, oval[1] - 56 + self.PILL_H)
            d.rounded_rectangle(mbox, radius=self.PILL_H // 2, fill=(255, 255, 255, 240))
            d.text((cx, (mbox[1] + mbox[3]) / 2), msg, font=self.f_pill, fill=(20, 20, 24, 255), anchor="mm")
            rects.append((oval[0] - 4, mbox[1], oval[2] + 4, oval[3] + 4))

        d.rounded_rectangle(dock, radius=28, fill=(14, 14, 16, 220))
        rects.append(dock)

        x = (self.w - cats_w) / 2
        y0 = dy0 + 16
        for cat, cw in zip(CATEGORIES, cat_widths):
            pill = (x, y0, x + cw, y0 + self.PILL_H)
            empty = not self.category_map[cat]
            if cat == current_cat:
                d.rounded_rectangle(pill, radius=self.PILL_H // 2, fill=ACCENT)
                color = ACCENT_TEXT
            else:
                color = (110, 110, 116, 255) if empty else TEXT_MUTED
            d.text(((pill[0] + pill[2]) / 2, (pill[1] + pill[3]) / 2), CATEGORY_LABELS[cat],
                   font=self.f_pill, fill=color, anchor="mm")
            if not empty:
                hits.append((pill, ("cat", cat)))
            x += cw + 8

        selected = selection[current_cat]
        x = (self.w - items_w) / 2
        cy0 = y0 + self.PILL_H + 14
        for i, key in enumerate(items):
            idx = i - 1
            circ = (x, cy0, x + self.CIRCLE, cy0 + self.CIRCLE)
            is_sel = idx == selected
            d.ellipse(circ, fill=(236, 236, 240, 255))
            if is_sel:
                d.ellipse((circ[0] - 4, circ[1] - 4, circ[2] + 4, circ[3] + 4), outline=ACCENT, width=4)

            ccx, ccy = x + self.CIRCLE / 2, cy0 + self.CIRCLE / 2
            if key is None:
                r = 16
                d.ellipse((ccx - r, ccy - r, ccx + r, ccy + r), outline=(90, 90, 96, 255), width=3)
                d.line((ccx - 11, ccy + 11, ccx + 11, ccy - 11), fill=(90, 90, 96, 255), width=3)
                label = "Nenhum"
            else:
                thumb = self.thumbs[key]
                layer.alpha_composite(thumb, dest=(int(ccx - thumb.width / 2), int(ccy - thumb.height / 2)))
                label = ACCESSORIES[key]["nome"]

            font = self.f_label_b if is_sel else self.f_label
            d.text((ccx, cy0 + self.CIRCLE + 17), label, font=font,
                   fill=TEXT if is_sel else TEXT_MUTED, anchor="mm")
            hits.append(((circ[0] - 10, circ[1] - 6, circ[2] + 10, circ[3] + 30), ("item", current_cat, idx)))
            x += slot

        rgba = np.asarray(layer)
        self.regions = []
        for (x0, y0_, x1, y1) in rects:
            x0, y0_ = max(int(x0) - 6, 0), max(int(y0_) - 6, 0)
            x1, y1 = min(int(np.ceil(x1)) + 6, self.w), min(int(np.ceil(y1)) + 6, self.h)
            if x1 <= x0 or y1 <= y0_:
                continue
            patch = rgba[y0_:y1, x0:x1].astype(np.float32)
            alpha = patch[:, :, 3:4] / 255.0
            fg = patch[:, :, 2::-1] * alpha
            self.regions.append((x0, y0_, x1, y1, fg, 1.0 - alpha))
        self.hitboxes = hits

    def render(self, frame, current_cat, selection, status, fps_shown, extra=""):
        key = (current_cat, tuple(selection.values()), status, fps_shown, extra)
        if key != self.cache_key:
            self._build(current_cat, selection, status, fps_shown, extra)
            self.cache_key = key
        for x0, y0, x1, y1, fg, inv_a in self.regions:
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            frame[y0:y1, x0:x1] = (fg + roi * inv_a).astype(np.uint8)
        return frame

    def hit(self, x, y):
        for (x0, y0, x1, y1), action in self.hitboxes:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return action
        return None

def get_screen_size():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(16), user32.GetSystemMetrics(17)
    except Exception:
        pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight() - 80
        root.destroy()
        return size
    except Exception:
        return 1280, 720

def export_check_images(accessories):
    os.makedirs(CHECK_DIR, exist_ok=True)
    for key, img in accessories.items():
        bg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        bg[:] = (255, 0, 255)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        out = img[:, :, :3] * alpha + bg * (1.0 - alpha)
        path = os.path.join(CHECK_DIR, f"{key}.png")
        cv2.imwrite(path, out.astype(np.uint8))
        print(f"[OK] {path}")
    print("Abra as imagens: o que estiver magenta ficou transparente.")

def save_measurements(rows):
    if not rows:
        print("[AVISO] Nenhuma medicao para salvar.")
        return
    os.makedirs(MEASURE_DIR, exist_ok=True)
    path = os.path.join(MEASURE_DIR, datetime.now().strftime("medicao_%Y%m%d_%H%M%S.csv"))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {len(rows)} quadros salvos em {path}")

def main():
    measure_mode = "--medir" in sys.argv
    accessories = load_accessories()
    category_map = build_category_map(accessories)

    if "--conferir" in sys.argv:
        export_check_images(accessories)
        return

    if not any(category_map.values()):
        print(f"\n[ERRO] Nenhum acessorio carregado. Coloque PNGs com fundo transparente em '{ASSETS_DIR}/'.")
        sys.exit(1)

    selection = {cat: -1 for cat in CATEGORIES}
    current_cat = next(c for c in CATEGORIES if category_map[c])

    def active_key():
        idx = selection[current_cat]
        return category_map[current_cat][idx] if idx >= 0 else None

    def select_category(cat):
        nonlocal current_cat
        if not category_map[cat]:
            print(f"[AVISO] Nenhum item carregado em {CATEGORY_LABELS[cat]}.")
            return
        current_cat = cat
        if selection[cat] == -1:
            selection[cat] = 0

    landmark_filter = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
    filter_on = True

    pitch_samples = []
    pitch_offset = 0.0

    rows = []
    t_start = time.perf_counter()
    last_pitch_print = 0.0
    last_face = None
    last_seen = 0.0
    fps = 0.0
    fps_shown = 0
    fps_updated = 0.0
    prev_t = time.perf_counter()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Nao foi possivel abrir a camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ok, frame = cap.read()
    if not ok:
        print("[ERRO] Nao foi possivel ler o frame da camera.")
        sys.exit(1)

    cam_h, cam_w = frame.shape[:2]
    screen_w, screen_h = get_screen_size()
    scale = min(1.0, (screen_w - 16) / cam_w, (screen_h - 8) / cam_h)
    frame_w, frame_h = int(cam_w * scale), int(cam_h * scale)
    print(f"[OK] Camera {cam_w}x{cam_h} -> janela {frame_w}x{frame_h}")
    ui = Interface(frame_w, frame_h, accessories, category_map)

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("\n=== Provador Virtual - TCC UNIP ===")
    print("  Clique nas categorias e nos acessorios para escolher.")
    print("  Teclado: [1-4] categoria | [A]/[D] trocar item | [C] recalibrar pitch | [Q] sair")
    if measure_mode:
        print("  MODO DE MEDICAO: [F] liga/desliga o filtro. O CSV e salvo ao sair.")
        print(f"  Olhe reto para a camera nos primeiros {PITCH_CALIB_FRAMES} quadros (calibracao do pitch).")
    print("  Nenhuma imagem e salva ou enviada: o processamento e local e em tempo real.\n")

    try:
        while True:

            ok, frame = cap.read()
            if not ok:
                print("[ERRO] Nao foi possivel ler o frame da camera.")
                break

            t_proc = time.perf_counter()
            now = t_proc
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            if now - fps_updated > 1.0:
                fps_shown = int(round(fps))
                fps_updated = now

            frame = cv2.flip(frame, 1)
            if scale < 1.0:
                frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detected = bool(results.multi_face_landmarks)
            if detected:
                raw = landmarks_to_array(results.multi_face_landmarks[0], frame_w, frame_h)
                pts = landmark_filter(raw, now) if filter_on else raw
                last_face = compute_face(pts)

                if len(pitch_samples) < PITCH_CALIB_FRAMES:
                    pitch_samples.append(last_face["pitch_raw"])
                    pitch_offset = float(np.median(pitch_samples))
                    if len(pitch_samples) == PITCH_CALIB_FRAMES:
                        print(f"[OK] Pitch calibrado: rosto reto = {pitch_offset:.1f} graus")
                last_face["pitch"] = last_face["pitch_raw"] - pitch_offset

                last_seen = now
                status = "ok"
            elif last_face is not None and now - last_seen < HOLD_SECONDS:
                status = "ok"
            else:
                last_face = None
                landmark_filter.reset()
                status = "lost"

            key_active = active_key()
            anchor_points, acc_width = [], 0.0
            if key_active and last_face is not None:
                frame, anchor_points, acc_width = draw_accessory(
                    frame, accessories[key_active], get_fit(key_active), last_face)

            extra = ""
            if measure_mode:
                extra = "  ·  MEDINDO" + ("" if filter_on else "  ·  filtro OFF")
            frame = ui.render(frame, current_cat, selection, status, fps_shown, extra)

            proc_ms = (time.perf_counter() - t_proc) * 1000.0

            if measure_mode:
                anchor = anchor_points[0] if anchor_points else (None, None)
                rows.append({
                    "t_s": round(now - t_start, 4),
                    "rosto_detectado": int(detected),
                    "status": status,
                    "filtro": int(filter_on),
                    "acessorio": key_active or "",
                    "ancora_x": None if anchor[0] is None else round(float(anchor[0]), 2),
                    "ancora_y": None if anchor[1] is None else round(float(anchor[1]), 2),
                    "largura_px": round(float(acc_width), 2),
                    "roll": None if last_face is None else round(last_face["roll"], 2),
                    "yaw": None if last_face is None else round(last_face["yaw"], 2),
                    "pitch_bruto": None if last_face is None else round(last_face["pitch_raw"], 2),
                    "pitch": None if last_face is None else round(last_face["pitch"], 2),
                    "proc_ms": round(proc_ms, 3),
                    "fps": round(fps, 2),
                })
                if last_face is not None and now - last_pitch_print > 1.0:
                    print(f"  pitch bruto {last_face['pitch_raw']:6.1f}  |  corrigido {last_face['pitch']:6.1f}"
                          f"  |  proc {proc_ms:5.1f} ms")
                    last_pitch_print = now

            cv2.imshow(WINDOW_NAME, frame)

            while clicks:
                action = ui.hit(*clicks.pop(0))
                if action and action[0] == "cat":
                    select_category(action[1])
                elif action and action[0] == "item":
                    selection[action[1]] = action[2]

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                select_category(CATEGORIES[key - ord("1")])

            elif key in (ord("d"), ord("a")):
                items = category_map[current_cat]
                step = 1 if key == ord("d") else -1

                idx = selection[current_cat] + 1 + step
                selection[current_cat] = idx % (len(items) + 1) - 1

            elif key == ord("c"):
                pitch_samples.clear()
                print("[*] Recalibrando pitch: olhe reto para a camera.")

            elif key == ord("f") and measure_mode:
                filter_on = not filter_on
                landmark_filter.reset()
                print(f"[*] Filtro {'LIGADO' if filter_on else 'DESLIGADO'}")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:

        if measure_mode:
            save_measurements(rows)

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()