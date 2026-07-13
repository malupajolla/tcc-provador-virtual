import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from datetime import datetime


ASSETS_DIR = "assets"


CATEGORIES = ["chapeus", "oculos", "brincos", "piercings"]

CATEGORY_LABELS = {
    "chapeus": "Chapeus",
    "oculos": "Oculos",
    "brincos": "Brincos",
    "piercings": "Piercings",
}

ACCESSORIES = {

    "chapeu": {
        "file": "chapeu.png",
        "scale": 3.0,
        "offset_x": 0,
        "offset_y": -1.0,
        "anchor": "forehead",
        "categoria": "chapeus",
    },


    "oculos": {
        "file": "oculos.png",
        "scale": 2.2,
        "offset_x": 0,
        "offset_y": 0,
        "anchor": "eyes",
        "categoria": "oculos",
    },
    "oculos2": {
        "file": "oculo2.png",
        "scale": 2.2,
        "offset_x": 0,
        "offset_y": 0,
        "anchor": "eyes",
        "categoria": "oculos",
    },


    "brinco_argola": {
        "file": "brinco_argola.png",
        "scale": 0.45,
        "offset_x": 0,
        "offset_y": 0.2,
        "anchor": "ears",
        "categoria": "brincos",
    },
    "brinco_perola": {
        "file": "brinco_perola.png",
        "scale": 0.35,
        "offset_x": 0,
        "offset_y": 0.2,
        "anchor": "ears",
        "categoria": "brincos",
    },


    "piercing_nariz": {
        "file": "piercing_nariz.png",
        "scale": 0.1,
        "offset_x": 0.15,
        "offset_y": 0.15,
        "anchor": "nose",
        "categoria": "piercings",
    },
    "piercing_septo": {
        "file": "piercing_septo.png",
        "scale": 0.15,
        "offset_x": 0,
        "offset_y": -0.04,
        "anchor": "nose",
        "categoria": "piercings",
    },
}

SMOOTHING = 0.75


LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FOREHEAD_TOP = 10
NOSE_BASE = 164
LEFT_EAR = 234
RIGHT_EAR = 454


def build_category_map():
    """Agrupa as chaves de ACCESSORIES por categoria, mantendo a ordem."""
    mapping = {cat: [] for cat in CATEGORIES}
    for key, cfg in ACCESSORIES.items():
        cat = cfg.get("categoria")
        if cat in mapping:
            mapping[cat].append(key)
    return mapping


def load_accessories():

    loaded = {}

    for name, cfg in ACCESSORIES.items():
        path = os.path.join(ASSETS_DIR, cfg["file"])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[AVISO] Não encontrei '{path}'. O acessório '{name}' será ignorado.")
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

        elif img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            img = cv2.merge((b, g, r, alpha))

        elif img.shape[2] == 4:
            pass

        else:
            print(f"[AVISO] Formato inesperado em '{path}'. Ignorando.")
            continue

        loaded[name] = img
        print(f"[OK] Acessório carregado: {name} ({path})")

    return loaded


def overlay_transparent(background, overlay, cx, cy, width, angle, flip_h=False):

    if overlay is None:
        return background

    if flip_h:
        overlay = cv2.flip(overlay, 1)

    aspect = overlay.shape[0] / overlay.shape[1]

    new_w = max(int(width), 10)
    new_h = max(int(new_w * aspect), 10)

    resized = cv2.resize(
        overlay,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    M = cv2.getRotationMatrix2D(
        (new_w / 2, new_h / 2),
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        resized,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    h, w = rotated.shape[:2]

    y1 = int(cy - h // 2)
    y2 = int(y1 + h)
    x1 = int(cx - w // 2)
    x2 = int(x1 + w)

    bg_h, bg_w = background.shape[:2]

    oy1 = max(0, -y1)
    oy2 = h - max(0, y2 - bg_h)
    ox1 = max(0, -x1)
    ox2 = w - max(0, x2 - bg_w)

    y1 = max(0, y1)
    y2 = min(bg_h, y2)
    x1 = max(0, x1)
    x2 = min(bg_w, x2)

    if y2 <= y1 or x2 <= x1:
        return background

    roi = rotated[oy1:oy2, ox1:ox2]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return background

    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
    fg = roi[:, :, :3].astype(np.float32)
    bg = background[y1:y2, x1:x2].astype(np.float32)

    blended = fg * alpha + bg * (1.0 - alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background


def get_landmark_px(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def compute_geometry(face_landmarks, frame_w, frame_h):

    lm = face_landmarks.landmark

    left_pt = np.array(get_landmark_px(lm[LEFT_EYE_OUTER], frame_w, frame_h))
    right_pt = np.array(get_landmark_px(lm[RIGHT_EYE_OUTER], frame_w, frame_h))
    top_pt = np.array(get_landmark_px(lm[FOREHEAD_TOP], frame_w, frame_h))
    nose_pt = np.array(get_landmark_px(lm[NOSE_BASE], frame_w, frame_h))
    left_ear_pt = np.array(get_landmark_px(lm[LEFT_EAR], frame_w, frame_h))
    right_ear_pt = np.array(get_landmark_px(lm[RIGHT_EAR], frame_w, frame_h))

    center = ((left_pt + right_pt) // 2).astype(int)
    eye_dist = float(np.linalg.norm(right_pt - left_pt))

    d_y = int(right_pt[1]) - int(left_pt[1])
    d_x = int(right_pt[0]) - int(left_pt[0])

    angle = np.degrees(np.arctan2(d_y, d_x))

    return center, eye_dist, angle, top_pt, nose_pt, left_ear_pt, right_ear_pt


def get_anchor_position(cfg, center, eye_dist, top_pt, nose_pt, side="left", left_ear_pt=None, right_ear_pt=None):

    anchor_type = cfg.get("anchor", "eyes")

    if anchor_type == "forehead":
        anchor_x = int(center[0])
        anchor_y = int(top_pt[1])

    elif anchor_type == "nose":
        anchor_x = int(nose_pt[0])
        anchor_y = int(nose_pt[1])

    elif anchor_type == "ears":
        ear_pt = left_ear_pt if side == "left" else right_ear_pt
        anchor_x = int(ear_pt[0])
        anchor_y = int(ear_pt[1])

    else:  # eyes
        anchor_x = int(center[0])
        anchor_y = int(center[1])

    offset_x = cfg.get("offset_x", 0)

    if anchor_type == "ears" and side == "right":
        offset_x = -offset_x

    anchor_x += int(eye_dist * offset_x)
    anchor_y += int(eye_dist * cfg.get("offset_y", 0))

    return anchor_x, anchor_y


def draw_interface(frame, active, current_category, frame_w, face_detected):

    bar_h = 60 + 22 * len(CATEGORIES)
    cv2.rectangle(frame, (0, 0), (frame_w, bar_h), (0, 0, 0), -1)

    header = "1-4 categoria | A/D item | +/- tam | I/K/J/L offset | S salvar | Q sair"
    cv2.putText(
        frame, header, (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    y = 45
    for i, cat in enumerate(CATEGORIES):
        item = active[cat]["key"]
        item_label = item if item else "nenhum"
        is_active = (cat == current_category)

        color = (0, 255, 255) if is_active else (200, 200, 200)
        prefix = "> " if is_active else "  "

        line = f"{prefix}[{i + 1}] {CATEGORY_LABELS[cat]}: {item_label}"
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
        )
        y += 22

    active_key = active[current_category]["key"]
    if active_key:
        cfg = ACCESSORIES[active_key]
        info = (f"scale={cfg['scale']:.2f}  offset_x={cfg['offset_x']:.2f}  "
                f"offset_y={cfg['offset_y']:.2f}  (copie esses valores pro config quando estiver bom)")
        cv2.putText(
            frame, info, (10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    if not face_detected:
        cv2.putText(
            frame,
            "Rosto nao detectado. Posicione seu rosto no centro da camera.",
            (30, bar_h + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )

    return frame


def save_snapshot(frame):
    output_dir = "prints"
    os.makedirs(output_dir, exist_ok=True)

    filename = datetime.now().strftime("provador_%Y%m%d_%H%M%S.png")
    path = os.path.join(output_dir, filename)

    cv2.imwrite(path, frame)
    print(f"[OK] Foto salva em: {path}")


def main():
    accessories = load_accessories()
    category_map = build_category_map()

    for cat in CATEGORIES:
        category_map[cat] = [k for k in category_map[cat] if k in accessories]

    if not any(category_map[cat] for cat in CATEGORIES):
        print("\n[ERRO] Nenhum acessório carregado em nenhuma categoria.")
        print(f"Coloque imagens PNG com fundo transparente na pasta '{ASSETS_DIR}/'")
        expected_files = [cfg["file"] for cfg in ACCESSORIES.values()]
        print("Arquivos esperados:")
        for file_name in expected_files:
            print(f" - {file_name}")
        sys.exit(1)


    active = {}
    for cat in CATEGORIES:
        has_items = len(category_map[cat]) > 0
        active[cat] = {
            "idx": 0 if has_items else -1,
            "key": category_map[cat][0] if has_items else None,
        }

    current_category = next((c for c in CATEGORIES if category_map[c]), CATEGORIES[0])

    smooth_state = {cat: {} for cat in CATEGORIES}

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a câmera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n=== Provador Virtual – TCC UNIP ===")
    print("  [1][2][3][4] → seleciona categoria (Chapeus/Oculos/Brincos/Piercings)")
    print("  [A] / [D]    → trocar item dentro da categoria (inclui 'nenhum')")
    print("  [+] / [-]    → aumentar/diminuir tamanho do item da categoria ativa")
    print("  [S]          → salvar foto")
    print("  [Q]          → sair")
    print("=====================================\n")

    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print("[ERRO] Não foi possível ler o frame da câmera.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_detected = bool(results.multi_face_landmarks)

        if face_detected:
            face_landmarks = results.multi_face_landmarks[0]

            (center, eye_dist, angle, top_pt, nose_pt,
             left_ear_pt, right_ear_pt) = compute_geometry(face_landmarks, frame_w, frame_h)

            for cat in CATEGORIES:
                key = active[cat]["key"]
                if key is None:
                    continue

                cfg = ACCESSORIES[key]
                img = accessories[key]

                if cfg["anchor"] == "ears":
                    sides = ["left", "right"]
                else:
                    sides = ["single"]

                for side in sides:
                    anchor_x, anchor_y = get_anchor_position(
                        cfg, center, eye_dist, top_pt, nose_pt,
                        side=side, left_ear_pt=left_ear_pt, right_ear_pt=right_ear_pt
                    )
                    size = int(eye_dist * cfg["scale"])
                    cur_angle = angle

                    state = smooth_state[cat].setdefault(side, {})
                    if state.get("x") is not None:
                        anchor_x = int(state["x"] * SMOOTHING + anchor_x * (1 - SMOOTHING))
                        anchor_y = int(state["y"] * SMOOTHING + anchor_y * (1 - SMOOTHING))
                        size = int(state["size"] * SMOOTHING + size * (1 - SMOOTHING))
                        cur_angle = state["angle"] * SMOOTHING + cur_angle * (1 - SMOOTHING)

                    state["x"] = anchor_x
                    state["y"] = anchor_y
                    state["size"] = size
                    state["angle"] = cur_angle

                    flip_h = (cfg["anchor"] == "ears" and side == "right")

                    frame = overlay_transparent(
                        background=frame,
                        overlay=img,
                        cx=anchor_x,
                        cy=anchor_y,
                        width=size,
                        angle=-cur_angle,
                        flip_h=flip_h
                    )
        else:
            for cat in CATEGORIES:
                smooth_state[cat] = {}

        frame = draw_interface(frame, active, current_category, frame_w, face_detected)

        cv2.imshow("Provador Virtual – TCC UNIP", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            idx = key - ord("1")
            if idx < len(CATEGORIES):
                current_category = CATEGORIES[idx]
                print(f"[*] Categoria ativa: {CATEGORY_LABELS[current_category]}")

        elif key == ord("d"):
            items = category_map[current_category]
            if items:
                cur_idx = active[current_category]["idx"]
                new_idx = cur_idx + 1
                if new_idx >= len(items):
                    new_idx = -1
                active[current_category]["idx"] = new_idx
                active[current_category]["key"] = items[new_idx] if new_idx >= 0 else None
                smooth_state[current_category] = {}
                print(f"[>] {CATEGORY_LABELS[current_category]}: {active[current_category]['key'] or 'nenhum'}")

        elif key == ord("a"):
            items = category_map[current_category]
            if items:
                cur_idx = active[current_category]["idx"]
                new_idx = cur_idx - 1
                if new_idx < -1:
                    new_idx = len(items) - 1
                active[current_category]["idx"] = new_idx
                active[current_category]["key"] = items[new_idx] if new_idx >= 0 else None
                smooth_state[current_category] = {}
                print(f"[<] {CATEGORY_LABELS[current_category]}: {active[current_category]['key'] or 'nenhum'}")

        elif key == ord("+") or key == ord("="):
            active_key = active[current_category]["key"]
            if active_key:
                ACCESSORIES[active_key]["scale"] += 0.1
                print(f"[+] Escala de '{active_key}': {ACCESSORIES[active_key]['scale']:.2f}")

        elif key == ord("-") or key == ord("_"):
            active_key = active[current_category]["key"]
            if active_key:
                ACCESSORIES[active_key]["scale"] = max(0.1, ACCESSORIES[active_key]["scale"] - 0.1)
                print(f"[-] Escala de '{active_key}': {ACCESSORIES[active_key]['scale']:.2f}")

        elif key == ord("i"):  
            active_key = active[current_category]["key"]
            if active_key:
                ACCESSORIES[active_key]["offset_y"] -= 0.05
                print(f"[offset_y] {active_key}: {ACCESSORIES[active_key]['offset_y']:.2f}")

        elif key == ord("k"):  
            active_key = active[current_category]["key"]
            if active_key:
                ACCESSORIES[active_key]["offset_y"] += 0.05
                print(f"[offset_y] {active_key}: {ACCESSORIES[active_key]['offset_y']:.2f}")

        elif key == ord("j"):  
            active_key = active[current_category]["key"]
            if active_key:
                ACCESSORIES[active_key]["offset_x"] -= 0.05
                print(f"[offset_x] {active_key}: {ACCESSORIES[active_key]['offset_x']:.2f}")

        elif key == ord("l"):  
            active_key = active[current_category]["key"]
            if active_key:
                ACCESSORIES[active_key]["offset_x"] += 0.05
                print(f"[offset_x] {active_key}: {ACCESSORIES[active_key]['offset_x']:.2f}")

        elif key == ord("s"):
            save_snapshot(frame)

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
