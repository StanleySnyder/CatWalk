import sys
import os
import re
import random
import time
import math
from typing import List, Tuple

# -------- Ресурсы (PyInstaller совместимо) --------
def resource_path(rel_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base, rel_path)

# --- Настройки ---
IMAGES_DIR = resource_path("images")

BASE_SPEED = 300                 # скорость (px/s)
MAX_LOGO_FRACTION = 0.25         # доля экрана для логотипа

# Пауза/возврат
INACTIVITY_TO_RESUME_SEC = 10 * 60  # 10 минут «тишины» -> продолжить
MOUSE_MOVE_THRESHOLD = 2            # пикселей
GRACE_SECONDS = 0.2                 # защита от случайного дрожания сразу после старта
PAUSE_BEHAVIOR = "freeze"           # "freeze" (замереть на месте). В Linux iconify часто неудобен.

# Веса по папкам
WEIGHT_MODE = "per_image"           # "per_image" | "per_folder_total"
DEFAULT_ROOT_WEIGHT = 1.0
WEIGHT_NAME_REGEX = re.compile(r"(-?\d+(?:\.\d+)?)")

# --- pygame ---
import pygame


# ---------- Вспомогательные (файлы/веса) ----------
def _is_image_file(path: str) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    return os.path.isfile(path) and os.path.splitext(path.lower())[1] in exts

def _parse_weight_from_folder_name(name: str):
    m = WEIGHT_NAME_REGEX.search(name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def discover_images_with_weights(images_dir: str):
    items = []
    if not os.path.isdir(images_dir):
        return items

    # файлы в корне
    root_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    root_files = [p for p in root_files if _is_image_file(p)]
    for p in root_files:
        items.append({"path": p, "weight": float(DEFAULT_ROOT_WEIGHT), "group": "__ROOT__"})

    # подпапки 1 уровня
    for name in os.listdir(images_dir):
        sub = os.path.join(images_dir, name)
        if not os.path.isdir(sub):
            continue
        w = _parse_weight_from_folder_name(name)
        if w is None or w <= 0:
            continue
        files = [os.path.join(sub, f) for f in os.listdir(sub)]
        files = [p for p in files if _is_image_file(p)]
        if not files:
            continue
        if WEIGHT_MODE == "per_image":
            for p in files:
                items.append({"path": p, "weight": float(w), "group": name})
        else:
            split = float(w) / len(files)
            for p in files:
                items.append({"path": p, "weight": split, "group": name})
    return items

def pick_weighted_excluding(weights: List[float], exclude_idx: int) -> int:
    if not weights:
        return 0
    total = 0.0
    adjusted = []
    for i, w in enumerate(weights):
        ww = 0.0 if i == exclude_idx else max(0.0, float(w))
        adjusted.append(ww)
        total += ww

    if total <= 0.0:
        if len(weights) <= 1:
            return 0
        idx = exclude_idx
        while idx == exclude_idx:
            idx = random.randrange(len(weights))
        return idx

    r = random.random() * total
    acc = 0.0
    for i, ww in enumerate(adjusted):
        acc += ww
        if r <= acc:
            return i
    return len(weights) - 1

# ---------- Графика ----------
def load_and_scale_image(path: str, max_w: int, max_h: int) -> pygame.Surface:
    img = pygame.image.load(path).convert_alpha()
    iw, ih = img.get_size()
    scale = min(max_w / iw, max_h / ih, 1.0)
    if scale < 1.0:
        img = pygame.transform.smoothscale(img, (max(1, int(iw*scale)), max(1, int(ih*scale))))
    return img

def random_heading_away_from_walls(hit_left: bool, hit_right: bool, hit_top: bool, hit_bottom: bool, away_deg=10) -> float:
    def deg2rad(d: float) -> float: return d * math.pi / 180.0
    def clamp_angle(a: float) -> float:
        while a <= -math.pi: a += 2*math.pi
        while a >   math.pi: a -= 2*math.pi
        return a

    min_cos_x = math.cos(deg2rad(90 - away_deg))
    while True:
        ang = random.uniform(-math.pi, math.pi)
        vx = math.cos(ang)
        vy = math.sin(ang)
        if hit_left and vx <=  +min_cos_x:   continue
        if hit_right and vx >=  -min_cos_x:  continue
        if hit_top and vy <=   +min_cos_x:   continue
        if hit_bottom and vy >= -min_cos_x:  continue
        return clamp_angle(ang)

def vel_from_heading(h: float, s: float) -> Tuple[float, float]:
    return math.cos(h) * s, math.sin(h) * s

# ---------- Основная программа ----------
def main():
    pygame.init()
    pygame.display.set_caption("Linux Screensaver — weighted images, straight segments")

    # определяем размер рабочего стола
    try:
        # pygame 2.1+: список мониторов; берём первый
        SW, SH = pygame.display.get_desktop_sizes()[0]
    except Exception:
        info = pygame.display.Info()
        SW, SH = info.current_w, info.current_h

    # полноэкранное безрамочное окно
    flags = pygame.NOFRAME | pygame.FULLSCREEN
    screen = pygame.display.set_mode((SW, SH), flags)

    clock = pygame.time.Clock()

    # --- загрузка файлов и весов ---
    items = discover_images_with_weights(IMAGES_DIR)
    if not items:
        print(f"⚠️ В '{IMAGES_DIR}' нет картинок (.png/.jpg/.jpeg/.bmp/.gif/.webp).")
        pygame.quit()
        return

    max_logo_w = int(SW * MAX_LOGO_FRACTION)
    max_logo_h = int(SH * MAX_LOGO_FRACTION)

    surfaces: List[pygame.Surface] = []
    weights: List[float] = []
    for it in items:
        surfaces.append(load_and_scale_image(it["path"], max_logo_w, max_logo_h))
        weights.append(float(it["weight"]))

    # старт
    current_idx = pick_weighted_excluding(weights, exclude_idx=-1)
    logo = surfaces[current_idx]
    rect = logo.get_rect()

    x = float(random.randint(0, max(0, SW - rect.width)))
    y = float(random.randint(0, max(0, SH - rect.height)))

    heading = random.uniform(-math.pi, math.pi)
    speed = float(BASE_SPEED)
    vx, vy = vel_from_heading(heading, speed)

    last_time = time.perf_counter()
    start_time = last_time
    last_activity_time = last_time
    last_mouse_pos = pygame.mouse.get_pos()

    paused = False
    running = True

    while running:
        activity_detected = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Esc — выход
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    activity_detected = True
            elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                if event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    lx, ly = last_mouse_pos
                    if abs(mx - lx) + abs(my - ly) >= MOUSE_MOVE_THRESHOLD:
                        activity_detected = True
                    last_mouse_pos = (mx, my)
                else:
                    activity_detected = True

        now = time.perf_counter()

        # защита от случайной активности при старте
        if now - start_time < GRACE_SECONDS:
            activity_detected = False

        # Пауза/возврат
        if activity_detected:
            last_activity_time = now
            paused = True
        else:
            if paused and (now - last_activity_time >= INACTIVITY_TO_RESUME_SEC):
                paused = False
                # сброс dt, чтобы не было скачка
                last_time = now

        if paused:
            # можно рисовать «затенение» или просто экономить CPU
            clock.tick(30)
            continue

        # ===== движение по прямой =====
        dt = now - last_time
        last_time = now

        x += vx * dt
        y += vy * dt
        rect.x = int(round(x))
        rect.y = int(round(y))

        hit_left = hit_right = hit_top = hit_bottom = False

        if rect.left <= 0:
            rect.left = 0; x = float(rect.left); hit_left = True
        elif rect.right >= SW:
            rect.right = SW; x = float(rect.left); hit_right = True

        if rect.top <= 0:
            rect.top = 0; y = float(rect.top); hit_top = True
        elif rect.bottom >= SH:
            rect.bottom = SH; y = float(rect.top); hit_bottom = True

        if hit_left or hit_right or hit_top or hit_bottom:
            # смена картинки по весам, исключая текущую
            prev_idx = current_idx
            current_idx = pick_weighted_excluding(weights, exclude_idx=prev_idx)
            logo = surfaces[current_idx]
            center = rect.center
            rect = logo.get_rect(center=center)
            rect.clamp_ip(pygame.Rect(0, 0, SW, SH))
            x, y = float(rect.x), float(rect.y)

            heading = random_heading_away_from_walls(hit_left, hit_right, hit_top, hit_bottom)
            vx, vy = vel_from_heading(heading, speed)

        # рендер
        screen.fill((0, 0, 0))     # фон (на Linux «кликается насквозь» без X11-хаков сложно, поэтому обычный fullscreen)
        screen.blit(logo, rect)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
