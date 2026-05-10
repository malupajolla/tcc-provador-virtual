import cv2
import numpy as np
import mediapipe as mp
import os
import sys


ASSETS_DIR = "assets"

ACCESSORIES = {
    "oculos":  {"file": "oculos.png",  "scale": 2.2, "offset_y": 0},
    "chapeu":  {"file": "chapeu.png",  "scale": 3.0, "offset_y": -1.8},
    "bigode":  {"file": "bigode.png",  "scale": 1.4, "offset_y": 0.6},
}
ACCESSORY_KEYS = list(ACCESSORIES.keys())


LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263

FOREHEAD_TOP    = 10
NOSE_BASE       = 164


def load_accessories():
    loaded = {}
    for name, cfg in ACCESSORIES.items():
        path = os.path.join(ASSETS_DIR, cfg["file"])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[AVISO] Não encontrei '{path}'. O acessório '{name}' será ignorado.")
        else:

            if img.shape[2] == 3:
                b, g, r = cv2.split(img)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                img = cv2.merge((b, g, r, alpha))
            loaded[name] = img
            print(f"[OK] Acessório carregado: {name} ({path})")
    return loaded


def overlay_transparent(background, overlay, cx, cy, width, angle):
   
   
    if overlay is None:
        return background


    aspect = overlay.shape[0] / overlay.shape[1]
    new_w = max(width, 10)
    new_h = max(int(new_w * aspect), 10)
    resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        resized, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    h, w = rotated.shape[:2]
    y1, y2 = cy - h // 2, cy - h // 2 + h
    x1, x2 = cx - w // 2, cx - w // 2 + w

    bg_h, bg_w = background.shape[:2]
    oy1, oy2 = max(0, -y1), h - max(0, y2 - bg_h)
    ox1, ox2 = max(0, -x1), w - max(0, x2 - bg_w)
    y1, y2 = max(0, y1), min(bg_h, y2)
    x1, x2 = max(0, x1), min(bg_w, x2)

    if y2 <= y1 or x2 <= x1:
        return background

    roi = rotated[oy1:oy2, ox1:ox2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return background

    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
    fg    = roi[:, :, :3].astype(np.float32)
    bg    = background[y1:y2, x1:x2].astype(np.float32)

    blended = fg * alpha + bg * (1.0 - alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def get_landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def compute_geometry(face_landmarks, frame_w, frame_h):

    lm = face_landmarks.landmark

    left_pt  = np.array(get_landmark_px(lm[LEFT_EYE_OUTER],  frame_w, frame_h))
    right_pt = np.array(get_landmark_px(lm[RIGHT_EYE_OUTER], frame_w, frame_h))
    top_pt   = np.array(get_landmark_px(lm[FOREHEAD_TOP],     frame_w, frame_h))
    nose_pt  = np.array(get_landmark_px(lm[NOSE_BASE],        frame_w, frame_h))

    center = ((left_pt + right_pt) // 2).astype(int)
    eye_dist = float(np.linalg.norm(right_pt - left_pt))

    dY = int(right_pt[1]) - int(left_pt[1])
    dX = int(right_pt[0]) - int(left_pt[0])
    angle = np.degrees(np.arctan2(dY, dX))

    return center, eye_dist, angle, top_pt, nose_pt


def main():
    accessories = load_accessories()

    if not accessories:
        print("\n[ERRO] Nenhum acessório carregado.")
        print(f"Coloque imagens PNG (com fundo transparente) na pasta '{ASSETS_DIR}/'")
        print("Nomes esperados: oculos.png, chapeu.png, bigode.png")
        sys.exit(1)

    available = [k for k in ACCESSORY_KEYS if k in accessories]
    current_idx = 0

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

    print("\n=== Provador Virtual – TCC UNIP ===")
    print("  [A] / [D]  → trocar acessório")
    print("  [Q]        → sair")
    print("=====================================\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_name = available[current_idx]
        cfg  = ACCESSORIES[current_name]
        img  = accessories[current_name]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                center, eye_dist, angle, top_pt, nose_pt = compute_geometry(
                    face_landmarks, frame_w, frame_h
                )

                if current_name == "chapeu":
                    anchor_x = int(center[0])
                    anchor_y = int(top_pt[1] + eye_dist * cfg["offset_y"])
                elif current_name == "bigode":
                    anchor_x = int(nose_pt[0])
                    anchor_y = int(nose_pt[1] + eye_dist * cfg["offset_y"])
                else:  
                    anchor_x = int(center[0])
                    anchor_y = int(center[1] + eye_dist * cfg["offset_y"])

                size = int(eye_dist * cfg["scale"])
                frame = overlay_transparent(frame, img, anchor_x, anchor_y, size, -angle)

        
        label = f"Acessorio: {current_name}  |  [A]/[D] trocar  [Q] sair"
        cv2.rectangle(frame, (0, 0), (frame_w, 30), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Provador Virtual – TCC UNIP", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            current_idx = (current_idx + 1) % len(available)
            print(f"[>] Acessório: {available[current_idx]}")
        elif key == ord('a'):
            current_idx = (current_idx - 1) % len(available)
            print(f"[<] Acessório: {available[current_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
