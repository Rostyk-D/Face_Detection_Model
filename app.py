import io
import zipfile
import tempfile
import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# -----------------------------
# НАЛАШТУВАННЯ СТОРІНКИ
# -----------------------------
st.set_page_config(
    page_title="Face Detection Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Face Detection Demo")
st.caption("Завантаж один файл, кілька зображень або ZIP-папку з фото — і подивись результат детекції.")


# -----------------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    """
    Завантаження моделі один раз.
    Якщо у моделі є custom layers / loss / metrics, додай їх у custom_objects.
    """
    custom_objects = {}
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        return model
    except Exception as e:
        raise RuntimeError(f"Не вдалося завантажити модель: {e}")


def resize_with_padding_rgb(img_rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = target_size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if new_w < w or new_h < h else cv2.INTER_LINEAR
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top

    out = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return out


def np_canny_uint8_impl(img_uint8, k_low=0.66, k_high=1.33, d=9, sigmaColor=75, sigmaSpace=75):
    if img_uint8 is None:
        return np.zeros((224, 224), dtype=np.uint8)

    try:
        img_f = cv2.bilateralFilter(img_uint8, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    except Exception:
        img_f = img_uint8

    if img_f.ndim == 3 and img_f.shape[2] == 3:
        gray = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_f if img_f.ndim == 2 else img_f[:, :, 0]

    v = np.median(gray)
    low = int(max(0, k_low * v))
    high = int(min(255, k_high * v))
    return cv2.Canny(gray, low, high)


def stable_mod_type(stem: str) -> int:
    """
    Стабільно повертає 0..3 для конкретного stem.
    """
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 4


def apply_train_style_augmentation(rgb: np.ndarray, stem: str):
    """
    Імітація notebook:
      0 -> чиста + засвітлена
      1 -> чиста + затемнена
      2 -> шумна + засвітлена
      3 -> шумна + затемнена
    """
    BRIGHT_FACTOR = 1.30
    DARK_FACTOR = 0.70
    TRAIN_NOISE_SIGMA_MIN = 0.10
    TRAIN_NOISE_SIGMA_MAX = 0.15
    SEED = 42

    mod_type = stable_mod_type(stem)

    img_f = rgb.astype(np.float32) / 255.0

    seed_int = int(hashlib.md5(f"{stem}_{SEED}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)

    if mod_type == 0:
        img_f = img_f * BRIGHT_FACTOR
    elif mod_type == 1:
        img_f = img_f * DARK_FACTOR
    elif mod_type == 2:
        img_f = img_f * BRIGHT_FACTOR
        sigma = rng.uniform(TRAIN_NOISE_SIGMA_MIN, TRAIN_NOISE_SIGMA_MAX)
        noise = rng.normal(0.0, sigma, img_f.shape).astype(np.float32)
        img_f = img_f + noise
    elif mod_type == 3:
        img_f = img_f * DARK_FACTOR
        sigma = rng.uniform(TRAIN_NOISE_SIGMA_MIN, TRAIN_NOISE_SIGMA_MAX)
        noise = rng.normal(0.0, sigma, img_f.shape).astype(np.float32)
        img_f = img_f + noise

    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8), mod_type


def pil_to_rgb_array(file_obj) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file_obj)).convert("RGB")
    return np.array(img)


def normalize_model_output(pred):
    """
    Приводить вихід моделі до numpy array.
    """
    if isinstance(pred, dict):
        pred = next(iter(pred.values()))
    elif isinstance(pred, (list, tuple)):
        pred = pred[0]
    return pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred)


def decode_boxes(pred_array: np.ndarray):
    """
    Очікуваний формат:
      [xc, yc, w, h, score]
    або масив shape (N, 5)
    або shape (1, N, 5)
    """
    arr = np.asarray(pred_array).astype(np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 1:
        arr = arr[None, :]

    boxes = []
    for row in arr:
        if row.shape[0] < 5:
            continue

        xc, yc, bw, bh, score = row[:5]
        boxes.append({
            "xc": float(xc),
            "yc": float(yc),
            "w": float(bw),
            "h": float(bh),
            "score": float(score),
        })
    return boxes


def draw_boxes(img_rgb: np.ndarray, boxes, conf_threshold: float = 0.5, color=(0, 255, 0), thickness: int = 2):
    """
    Малює прямокутники на RGB-зображенні.
    Координати вважаються нормалізованими 0..1.
    """
    out = img_rgb.copy()
    h, w = out.shape[:2]

    for b in boxes:
        score = b["score"]
        if score < conf_threshold:
            continue

        xc, yc, bw, bh = b["xc"], b["yc"], b["w"], b["h"]

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )
    return out


def make_prediction(model, img_rgb: np.ndarray, target_size: int, conf_threshold: float, thickness: int, stem: str):
    """
    Підготовка input для моделі:
      - resize + padding
      - train-style augmentation як у notebook
      - edge map через Canny
    """
    img_ready = resize_with_padding_rgb(img_rgb, target_size)

    img_used, mod_type = apply_train_style_augmentation(img_ready, stem)
    edge = np_canny_uint8_impl(img_used)

    img_in = img_used.astype(np.float32) / 255.0
    edge_in = (edge.astype(np.float32) / 255.0)[..., None]

    pred = model([img_in[None, ...], edge_in[None, ...]], training=False)
    pred_np = normalize_model_output(pred)
    boxes = decode_boxes(pred_np)

    vis = draw_boxes(img_used, boxes, conf_threshold=conf_threshold, color=(0, 255, 0), thickness=thickness)
    return img_ready, img_used, edge, vis, boxes, mod_type, pred_np


def image_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    im = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def is_image_file(name: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_zip_images(zip_file) -> list[tuple[str, np.ndarray]]:
    """
    Повертає список (name, rgb_array) з ZIP.
    """
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = Path(tmpdir) / "upload.zip"
        with open(zpath, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zpath, "r") as z:
            for member in z.namelist():
                if member.endswith("/") or not is_image_file(member):
                    continue
                try:
                    data = z.read(member)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    results.append((member, np.array(img)))
                except Exception:
                    continue
    return results


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Налаштування")

    model_path = st.text_input(
        "Шлях до моделі",
        value="face_detector_rgb_edge_best_1.keras"
    )

    conf_threshold = st.slider("Поріг впевненості", 0.0, 1.0, 0.30, 0.01)
    thickness = st.slider("Товщина рамки", 1, 6, 2, 1)

    st.markdown("---")
    st.info("Підтримка: одиночне фото, кілька фото, ZIP-архів із папкою.")


# -----------------------------
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# -----------------------------
try:
    model = load_model(model_path)
    st.success(f"Модель завантажено: {model_path}")
except Exception as e:
    st.error(str(e))
    st.stop()


# -----------------------------
# UI
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Один файл", "Кілька файлів", "ZIP-папка"])

with tab1:
    up = st.file_uploader("Завантаж зображення", type=["jpg", "jpeg", "png", "bmp", "webp"], key="single")

    if up is not None:
        img_rgb = pil_to_rgb_array(up)
        orig, used, edge, vis, boxes, mod_type, pred_np = make_prediction(
            model,
            img_rgb,
            int(target_size),
            conf_threshold,
            thickness,
            stem=Path(up.name).stem
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Оригінал")
            st.image(orig, use_column_width=True)
        with c2:
            st.subheader(f"Input after preprocess (mod={mod_type})")
            st.image(used, use_column_width=True)
        with c3:
            st.subheader("Результат детекції")
            st.image(vis, use_column_width=True)

        scores = [b["score"] for b in boxes]
        max_score = max(scores) if scores else 0.0
        count = len([b for b in boxes if b["score"] >= conf_threshold])

        st.write(f"Знайдено боксів: **{count}**")
        st.write(f"Max score: **{max_score:.4f}**")
        st.write(f"Raw prediction shape: `{np.asarray(pred_np).shape}`")

        if boxes:
            st.dataframe(
                [
                    {
                        "xc": round(b["xc"], 4),
                        "yc": round(b["yc"], 4),
                        "w": round(b["w"], 4),
                        "h": round(b["h"], 4),
                        "score": round(b["score"], 4),
                    }
                    for b in boxes
                ],
                use_container_width=True
            )

        st.download_button(
            "Завантажити результат PNG",
            data=image_to_png_bytes(vis),
            file_name="face_detection_result.png",
            mime="image/png"
        )

with tab2:
    ups = st.file_uploader(
        "Завантаж кілька зображень",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="multi"
    )

    if ups:
        st.write(f"Завантажено файлів: **{len(ups)}**")

        for idx, up in enumerate(ups, start=1):
            with st.expander(f"{idx}. {up.name}", expanded=(idx == 1)):
                img_rgb = pil_to_rgb_array(up)
                orig, used, edge, vis, boxes, mod_type, pred_np = make_prediction(
                    model,
                    img_rgb,
                    int(target_size),
                    conf_threshold,
                    thickness,
                    stem=Path(up.name).stem
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(orig, caption="Оригінал", use_column_width=True)
                with c2:
                    st.image(used, caption=f"Input after preprocess (mod={mod_type})", use_column_width=True)
                with c3:
                    st.image(vis, caption="Результат", use_column_width=True)

                scores = [b["score"] for b in boxes]
                max_score = max(scores) if scores else 0.0
                count = len([b for b in boxes if b["score"] >= conf_threshold])

                st.write(f"Боксів вище порога: **{count}**")
                st.write(f"Max score: **{max_score:.4f}**")
                st.write(f"Raw prediction shape: `{np.asarray(pred_np).shape}`")

                st.download_button(
                    f"Завантажити {up.name} результат",
                    data=image_to_png_bytes(vis),
                    file_name=f"detected_{Path(up.name).stem}.png",
                    mime="image/png",
                    key=f"dl_{idx}"
                )

with tab3:
    zip_up = st.file_uploader("Завантаж ZIP-архів із зображеннями", type=["zip"], key="zip")

    if zip_up is not None:
        images = extract_zip_images(zip_up)
        if not images:
            st.warning("У ZIP не знайдено зображень.")
        else:
            st.write(f"Знайдено зображень у ZIP: **{len(images)}**")

            for idx, (name, img_rgb) in enumerate(images, start=1):
                with st.expander(f"{idx}. {name}", expanded=(idx == 1)):
                    orig, used, edge, vis, boxes, mod_type, pred_np = make_prediction(
                        model,
                        img_rgb,
                        int(target_size),
                        conf_threshold,
                        thickness,
                        stem=Path(name).stem
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(orig, caption="Оригінал", use_column_width=True)
                    with c2:
                        st.image(used, caption=f"Input after preprocess (mod={mod_type})", use_column_width=True)
                    with c3:
                        st.image(vis, caption="Результат", use_column_width=True)

                    scores = [b["score"] for b in boxes]
                    max_score = max(scores) if scores else 0.0
                    count = len([b for b in boxes if b["score"] >= conf_threshold])

                    st.write(f"Боксів вище порога: **{count}**")
                    st.write(f"Max score: **{max_score:.4f}**")
                    st.write(f"Raw prediction shape: `{np.asarray(pred_np).shape}`")

                    st.download_button(
                        f"Завантажити {Path(name).stem} результат",
                        data=image_to_png_bytes(vis),
                        file_name=f"detected_{Path(name).stem}.png",
                        mime="image/png",
                        key=f"zipdl_{idx}"
                    )