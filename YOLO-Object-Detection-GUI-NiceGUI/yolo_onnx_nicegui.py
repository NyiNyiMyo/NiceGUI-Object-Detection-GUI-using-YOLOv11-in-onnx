# -*- coding: utf-8 -*-
"""
📟 Makers - YOLO Object Detection  (NiceGUI native desktop edition)

Run (dev):      python yolo_onnx_nicegui.py
Requirements:   pip install nicegui pywebview onnxruntime opencv-python numpy
                (GPU: pip install onnxruntime-gpu)
"""
# Needed for PyInstaller + NiceGUI native mode (must run before anything else)
from multiprocessing import freeze_support  # noqa
freeze_support()  # noqa

import ast
import asyncio
import base64
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed!")
    print("Install with: pip install onnxruntime")
    print("(For GPU acceleration: pip install onnxruntime-gpu)")
    sys.exit(1)

from fastapi.responses import Response, StreamingResponse
from nicegui import app, run, ui


# ============================================================
# Paths (PyInstaller --onefile compatible)
# ============================================================
def app_dir() -> Path:
    """Folder that holds the .py file, or the .exe when frozen.
    Used for user-writable / user-replaceable things (models override, captures)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(relative: str) -> Path:
    """Path to a bundled read-only resource.
    In a --onefile build PyInstaller unpacks data files into sys._MEIPASS."""
    base = getattr(sys, '_MEIPASS', None)
    return (Path(base) if base else app_dir()) / relative


def find_default_model():
    """Look next to the executable first (lets users swap the model without
    rebuilding), then in the bundled resources."""
    for candidate in (app_dir() / 'models' / 'yolo11n.onnx',
                      resource_path('models/yolo11n.onnx')):
        if candidate.exists():
            return candidate
    return None


def disable_power_throttling():
    """Windows only: stop the OS from putting this process into Efficiency mode
    (EcoQoS) when it has no foreground window, e.g. terminal minimized or a
    --windowed build. That throttling can slow ONNX inference 3-4x."""
    if not sys.platform.startswith('win'):
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG),
                        ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = k32.GetCurrentProcess()
        # version 1, control EXECUTION_SPEED (0x1), state 0 = throttling OFF
        state = PROCESS_POWER_THROTTLING_STATE(1, 0x1, 0)
        k32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
        k32.SetPriorityClass(handle, 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


# ============================================================
# Colorful light theme
# ============================================================
COLORS = {
    'fg': '#1e293b',
    'fg_dim': '#64748b',
    'accent': '#4f46e5',
    'success': '#16a34a',
    'warning': '#ea580c',
    'error': '#dc2626',
    'border': '#e2e8f0',
    'card': '#ffffff',
    'soft': '#f8fafc',
    'danger': '#dc2626',
    'green': '#16a34a',
    'sky': '#0284c7',
    'orange': '#ea580c',
}

# One accent color per mode
MODE_COLORS = {
    'image': '#4f46e5',   # indigo
    'video': '#db2777',   # pink
    'webcam': '#059669',  # emerald
    'folder': '#d97706',  # amber
}

HEADER_GRADIENT = 'background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);'

# A fixed color palette (BGR) used to draw boxes per class id
BOX_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 128), (128, 128, 0), (128, 0, 128),
    (192, 192, 192), (64, 64, 255), (64, 255, 64), (255, 64, 64), (200, 200, 0),
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
PLACEHOLDER = "📟 Makers - YOLO Object Detection\n\nClick 'Image', 'Video', or 'Webcam' to start"

# Grid settings for folder inference (each cell is rendered at this size)
GRID_COLS, GRID_ROWS = 2, 2
GRID_COUNT = GRID_COLS * GRID_ROWS
CELL_W, CELL_H = 640, 480

# Camera backend: DirectShow on Windows (fast open, reliable enumeration, faster reads)
CAM_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY

# mode -> Page, used by the MJPEG streaming endpoint
STREAMS = {}


# ============================================================
# ONNX inference (unchanged logic)
# ============================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad image to new_shape while keeping aspect ratio.
    Returns the padded image, the resize ratio, and (dw, dh) half-padding."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class SimpleBox:
    """Mimics the small subset of ultralytics' Boxes API the UI code relies on."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class SimpleResult:
    """Mimics the small subset of ultralytics' Results API the UI code relies on."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class ONNXYOLO:
    """Thin ONNX Runtime wrapper that reproduces the predict() -> result interface
    the rest of the app expects, so the GUI code barely has to change."""

    def __init__(self, model_path):
        providers = []
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        so = ort.SessionOptions()
        n_threads = os.environ.get('YOLO_ORT_THREADS')   # optional tuning; unset = ORT default
        if n_threads and n_threads.isdigit() and int(n_threads) > 0:
            so.intra_op_num_threads = int(n_threads)
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        self.names = self._load_names()

    def _load_names(self):
        names = {}
        try:
            meta = self.session.get_modelmeta()
            custom = meta.custom_metadata_map
            if custom and 'names' in custom:
                parsed = ast.literal_eval(custom['names'])
                if isinstance(parsed, dict):
                    names = {int(k): v for k, v in parsed.items()}
                elif isinstance(parsed, (list, tuple)):
                    names = {i: v for i, v in enumerate(parsed)}
        except Exception:
            names = {}

        if not names:
            names = {i: f"class{i}" for i in range(1000)}
        return names

    def _postprocess(self, pred, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
        orig_h, orig_w = orig_shape[:2]

        pred = pred[0]
        # Normalize to shape (num_anchors, 4 + num_classes)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.shape[1] == 0:
            return [], [], []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = scores > conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return [], [], []

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Undo letterbox padding/scale to map back to original image coordinates
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thres, iou_thres)
        if indices is None or len(indices) == 0:
            return [], [], []
        indices = np.array(indices).flatten()

        final_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]
        final_scores = [float(scores[i]) for i in indices]
        final_class_ids = [int(class_ids[i]) for i in indices]

        return final_boxes, final_scores, final_class_ids

    def _draw(self, img_bgr, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = box.cls[0]
            conf = box.conf[0]
            name = self.names.get(cls_id, f"class{cls_id}")
            color = BOX_PALETTE[cls_id % len(BOX_PALETTE)]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(img_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, ty1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_bgr

    def infer(self, image_bgr, conf=0.25, iou=0.45):
        """Run detection on a BGR numpy image. Returns (SimpleResult, annotated_bgr)."""
        letter_img, ratio, (dw, dh) = letterbox(image_bgr, (self.input_h, self.input_w))
        img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, 0).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]

        boxes, scores, class_ids = self._postprocess(pred, conf, iou, ratio, dw, dh, image_bgr.shape)
        simple_boxes = [SimpleBox(class_ids[i], scores[i], boxes[i]) for i in range(len(boxes))]
        result = SimpleResult(simple_boxes, self.names)

        annotated = self._draw(image_bgr.copy(), simple_boxes)
        return result, annotated


def fit_into_cell(img_bgr, w, h, bg=(45, 45, 45)):
    """Fit an image into a w x h cell (keep aspect ratio, centered)."""
    ih, iw = img_bgr.shape[:2]
    r = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((h, w, 3), bg, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = resized
    return cell


def build_grid(annotated_list, names_list):
    """Compose annotated BGR images into a GRID_COLS x GRID_ROWS grid (BGR)."""
    cells = []
    for img, name in zip(annotated_list, names_list):
        cell = fit_into_cell(img, CELL_W, CELL_H)
        label = name if len(name) <= 40 else name[:37] + "..."
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(cell, (0, CELL_H - th - bl - 10), (tw + 12, CELL_H), (30, 30, 30), -1)
        cv2.putText(cell, label, (6, CELL_H - bl - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), (64, 64, 64), 2)
        cells.append(cell)
    while len(cells) < GRID_COUNT:
        cells.append(np.full((CELL_H, CELL_W, 3), (45, 45, 45), dtype=np.uint8))
    rows = [np.hstack(cells[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


# ============================================================
# Display helpers
# ============================================================
def to_jpeg_bytes(img_bgr, max_w=1600, quality=88):
    """Encode a BGR image as JPEG bytes (downscaled if wider than max_w)."""
    h, w = img_bgr.shape[:2]
    if w > max_w:
        img_bgr = cv2.resize(img_bgr, (max_w, max(1, int(h * max_w / w))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode frame for display")
    return buf.tobytes()


def fit_width(img_bgr, max_w):
    """Always returns a NEW image no wider than max_w (fast linear resize)."""
    h, w = img_bgr.shape[:2]
    if w > max_w:
        return cv2.resize(img_bgr, (max_w, max(1, int(h * max_w / w))),
                          interpolation=cv2.INTER_LINEAR)
    return img_bgr.copy()


def to_data_uri(img_bgr, max_w=1600, quality=88):
    """JPEG data-URI for one-shot images (Image / Folder pages)."""
    return 'data:image/jpeg;base64,' + base64.b64encode(
        to_jpeg_bytes(img_bgr, max_w, quality)).decode('ascii')


def open_capture(source):
    """Open a video file or a camera index. Cameras use the platform's fast
    backend, MJPG and a 1-frame buffer for higher FPS and lower latency."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, CAM_BACKEND)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap
    return cv2.VideoCapture(source)


def draw_fps(img_bgr, fps):
    """FPS overlay near the top-left corner: big yellow text on a dark badge.
    Size scales with the frame so it stays readable after downscaling."""
    h, w = img_bgr.shape[:2]
    scale = max(0.9, min(w, h) / 500.0)
    thick = max(2, int(round(scale * 2)))
    text = f"FPS: {fps:.1f}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(round(8 * scale))
    x, y = 12, 12
    cv2.rectangle(img_bgr, (x, y), (x + tw + 2 * pad, y + th + bl + 2 * pad),
                  (60, 30, 20), -1)
    cv2.rectangle(img_bgr, (x, y), (x + tw + 2 * pad, y + th + bl + 2 * pad),
                  (0, 255, 255), 2)
    cv2.putText(img_bgr, text, (x + pad, y + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thick, cv2.LINE_AA)
    return img_bgr


def dialog_const(kind):
    """pywebview file-dialog constant that works on old and new versions."""
    import webview
    enum = getattr(webview, 'FileDialog', None)
    if enum is not None:
        return getattr(enum, kind)
    return getattr(webview, f'{kind}_DIALOG')


class Page:
    """Holds the widgets/state that belong to one mode page."""

    def __init__(self, mode):
        self.mode = mode
        self.image = None
        self.placeholder = None
        self.file_label = None
        self.select_btn = None
        self.stop_btn = None
        self.clear_btn = None
        self.save_btn = None
        self.swap_btn = None
        self.shuffle_btn = None
        self.pause_btn = None
        self.progress = None
        self.result_col = None
        self.last_result = None   # full-resolution annotated BGR image
        # live stream hand-off: worker thread -> FastAPI MJPEG endpoint
        self.stream = None
        self.stream_active = False
        self.jpeg = None
        self.ver = 0


# ============================================================
# FastAPI: MJPEG live stream for Video / Webcam
# ============================================================
@app.get('/stream/{mode}')
async def stream_mode(mode: str):
    """Serves the latest annotated frames as multipart JPEG. The browser decodes
    them natively, so no base64 / websocket / Vue re-render per frame."""
    page = STREAMS.get(mode)
    if page is None:
        return Response(status_code=404)

    async def frames():
        last = -1
        while True:
            jpeg, ver = page.jpeg, page.ver
            if jpeg is not None and ver != last:
                last = ver
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                       + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
            elif not page.stream_active:
                break
            else:
                await asyncio.sleep(0.004)

    return StreamingResponse(frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control': 'no-store'})


# ============================================================
# Application
# ============================================================
class DetectorApp:
    """
    YOLO Object Detection - NiceGUI native desktop app.
    Header tabs for modes, left drawer for model + settings, status footer.
    """

    MODES = [
        ('image', "🖼 Image"),
        ('video', "🎬 Video"),
        ('webcam', "📹 Webcam"),
        ('folder', "📂 Folder"),
    ]

    def __init__(self):
        # Variables
        self.model = None
        self.model_path = None
        self.current_file = None
        self.folder_path = None
        self.folder_images = []
        self.video_thread = None
        self.stop_video = False
        self.is_processing = False
        self.active_page = None
        self.paused = False
        self.conf_value = 0.30
        self.iou_value = 0.30

        # Camera state
        self.available_cameras = [0]
        self.current_cam_index = 0
        self._swapping = False
        self.webcam_source = None

        # worker -> UI hand-off
        self._status = None
        self._video_payload = None

        self.pages = {}

    # ============================================================
    # UI construction
    # ============================================================
    def build(self):
        ui.colors(primary='#4f46e5', secondary='#0ea5e9', accent='#db2777',
                  positive='#16a34a', negative='#dc2626', warning='#ea580c', info='#0284c7')
        ui.add_css('''
            body { background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 50%, #fdf2f8 100%) fixed; }
            .q-page-container, .q-layout { background: transparent; }
            .q-tab { min-height: 44px; }
            .q-tab__label { font-weight: 700; font-size: 14px; }
            .nicegui-content { padding: 14px; }
        ''')

        # ---- Header: brand + mode tabs ----
        with ui.header(elevated=True).classes('items-center gap-4 px-4 py-1 no-wrap').style(HEADER_GRADIENT):
            with ui.column().classes('gap-0'):
                ui.label('📟 Makers').classes('text-2xl font-bold text-white leading-tight')
                ui.label('YOLO Object Detection').classes('text-xs text-white').style('opacity:.85')
            ui.space()
            ui.label('📁 Select Option').classes('text-xs font-bold text-white gt-sm').style('opacity:.85')
            self.tabs = ui.tabs(value='image', on_change=lambda e: self.show_page(e.value)) \
                .props('dense inline-label no-caps indicator-color=yellow-5 active-color=white') \
                .classes('text-white')
            with self.tabs:
                for mode, text in self.MODES:
                    ui.tab(mode, label=text)

        # ---- Left drawer: model + detection settings ----
        with ui.left_drawer(fixed=True, bordered=True).props('width=300') \
                .style('background: linear-gradient(180deg, #ffffff 0%, #eef2ff 100%)'):
            self._build_drawer()

        # ---- Footer: status bar ----
        with ui.footer().classes('items-center px-4 py-1').style(
                f"background:#ffffff; border-top:3px solid {COLORS['accent']}"):
            self.status_label = ui.label('Ready').classes('text-sm font-bold').style(
                f"color:{COLORS['fg']}")

        # ---- Pages ----
        with ui.tab_panels(self.tabs, value='image', animated=False) \
                .props('keep-alive').classes('w-full bg-transparent'):
            for mode, _ in self.MODES:
                self.pages[mode] = self._build_page(mode)

        self.show_page('image')

        # Background jobs after the page is up
        ui.timer(0.1, self._tick)
        ui.timer(0.2, self.auto_load_model, once=True)
        ui.timer(0.6, self._init_cameras, once=True)
        ui.timer(1.0, self._maximize, once=True)

    def _section(self, text, color):
        ui.label(text).classes('text-sm font-bold').style(f'color:{color}')

    def _build_drawer(self):
        with ui.column().classes('w-full gap-3'):
            # --- Model ---
            self._section('🤖 ONNX Model', COLORS['accent'])
            with ui.card().classes('w-full no-shadow rounded-xl gap-1').style(
                    f"background:{COLORS['accent']}12; border:1px solid {COLORS['accent']}55"):
                ui.label('Current Model:').classes('text-xs').style(f"color:{COLORS['fg_dim']}")
                self.model_label = ui.label('Loading...').classes('text-sm font-bold').style(
                    f"color:{COLORS['error']}; word-break:break-all")
                ui.button('📂 Change Model', on_click=self.load_model, color=COLORS['accent']) \
                    .props('unelevated no-caps rounded').classes('w-full mt-2')
                ui.button('🔄 Reload', on_click=self.reload_model, color=COLORS['sky']) \
                    .props('unelevated no-caps rounded').classes('w-full')

            # --- Settings ---
            self._section('⚙️ Detection Settings', '#db2777')
            with ui.card().classes('w-full no-shadow rounded-xl gap-1').style(
                    'background:#db277712; border:1px solid #db277755'):
                with ui.row().classes('w-full items-center justify-between no-wrap'):
                    ui.label('Confidence Threshold:').classes('text-xs').style(f"color:{COLORS['fg']}")
                    self.conf_label = ui.label('0.30').classes('text-sm font-bold').style(
                        f"color:{COLORS['accent']}")
                self.conf_scale = ui.slider(min=0.1, max=0.9, step=0.01, value=0.30,
                                            on_change=self.update_conf_label) \
                    .props('color=indigo-6 label')

                with ui.row().classes('w-full items-center justify-between no-wrap mt-2'):
                    ui.label('IoU Threshold:').classes('text-xs').style(f"color:{COLORS['fg']}")
                    self.iou_label = ui.label('0.30').classes('text-sm font-bold').style(
                        'color:#db2777')
                self.iou_scale = ui.slider(min=0.1, max=0.9, step=0.01, value=0.30,
                                           on_change=self.update_iou_label) \
                    .props('color=pink-6 label')

    def _btn(self, text, color, on_click, disabled=False):
        b = ui.button(text, on_click=on_click, color=color) \
            .props('unelevated no-caps rounded').classes('px-4 font-bold')
        if disabled:
            b.set_enabled(False)
        return b

    def _build_page(self, mode):
        page = Page(mode)
        STREAMS[mode] = page
        accent = MODE_COLORS[mode]
        title = dict(self.MODES)[mode]
        has_results = mode in ('image', 'folder')
        view_h = 'calc(100vh - 330px)'

        with ui.tab_panel(mode).classes('p-0 gap-3'):
            # Header: title + selected file
            with ui.card().classes('w-full no-shadow rounded-xl').style(
                    f"background:{COLORS['card']}; border-left:8px solid {accent}"):
                with ui.row().classes('items-center gap-4 w-full'):
                    ui.label(title).classes('text-xl font-bold').style(f'color:{accent}')
                    ui.label('📄 Selected File:').classes('text-sm').style(f"color:{COLORS['fg_dim']}")
                    page.file_label = ui.label('No file selected').classes('font-bold').style(
                        f"color:{COLORS['error']}")

            # Toolbar: mode-specific action + common actions
            select_cmd = {
                'image': self.upload_image,
                'video': self.upload_video,
                'webcam': self.use_webcam,
                'folder': self.upload_folder,
            }[mode]
            with ui.row().classes('items-center gap-2 w-full'):
                page.select_btn = self._btn(title, accent, select_cmd)

                if mode == 'folder':
                    page.shuffle_btn = self._btn("🎲 Shuffle 4 Images", accent,
                                                 self.shuffle_folder, disabled=True)
                if mode == 'webcam':
                    page.swap_btn = self._btn("🔀 Swap Cam", accent, self.swap_camera)
                if mode == 'video':
                    page.pause_btn = self._btn("⏸ Pause", COLORS['sky'],
                                               self.toggle_pause, disabled=True)
                if mode in ('video', 'webcam'):
                    page.stop_btn = self._btn("⏹ Stop", COLORS['danger'],
                                              self.stop_detection, disabled=True)

                page.clear_btn = self._btn("🗑 Clear", COLORS['orange'], self.clear_display)

                if mode == 'webcam':
                    page.save_btn = self._btn("📸 Capture", COLORS['green'],
                                              self.capture_frame, disabled=True)
                elif mode != 'video':
                    page.save_btn = self._btn("💾 Save Result", COLORS['green'],
                                              self.save_result, disabled=True)

            # Content: display (+ results for image/folder only)
            with ui.row().classes('w-full no-wrap gap-3 items-stretch'):
                with ui.card().classes('col no-shadow rounded-xl p-0 gap-0').style(
                        f"background:{COLORS['card']}; min-width:0; overflow:hidden; "
                        f"border:2px solid {accent}55"):
                    ui.label('📺 Display').classes('w-full text-sm font-bold px-4 py-2').style(
                        f'color:{accent}; background:{accent}18')
                    with ui.element('div').classes('relative w-full').style(
                            f'height:{view_h}; min-height:340px; background:{accent}0d'):
                        page.placeholder = ui.label(PLACEHOLDER).classes(
                            'absolute-center text-center text-lg').style(
                            f"white-space:pre-line; color:{COLORS['fg_dim']}; width:80%")
                        if mode in ('video', 'webcam'):
                            # live MJPEG <img> fed by the FastAPI /stream endpoint
                            page.stream = ui.element('img').classes('w-full').style(
                                'height:100%; object-fit:contain; display:block')
                            page.stream.set_visibility(False)
                        else:
                            page.image = ui.image().props(
                                'fit=contain no-spinner no-transition').classes('w-full').style(
                                'height:100%')
                            page.image.set_visibility(False)
                    # Progress bar - only shown while actively processing
                    page.progress = ui.linear_progress(show_value=False).props(
                        'indeterminate rounded').classes('w-full')
                    page.progress.set_visibility(False)

                if has_results:
                    with ui.card().classes('no-shadow rounded-xl p-0 gap-0').style(
                            f"width:380px; flex:none; background:{COLORS['card']}; "
                            f"overflow:hidden; border:2px solid {accent}55"):
                        ui.label('📊 Detection Results').classes(
                            'w-full text-sm font-bold px-4 py-2').style(
                            f'color:{accent}; background:{accent}18')
                        with ui.scroll_area().classes('w-full').style(
                                'height:calc(100vh - 372px); min-height:300px'):
                            page.result_col = ui.column().classes('w-full gap-2 p-3')
                    self.show_no_results(page)
        return page

    # ============================================================
    # Page switching
    # ============================================================
    def show_page(self, mode):
        # Switching pages while a stream is running: stop it first
        if self.is_processing and self.active_page and self.active_page.mode != mode:
            self.stop_video = True
        self.active_page = self.pages[mode]

    async def _maximize(self):
        """Maximize the native window on launch (best effort, cross platform)."""
        try:
            win = app.native.main_window
            if win is not None:
                r = win.maximize()
                if asyncio.iscoroutine(r):
                    await r
        except Exception:
            pass

    # ============================================================
    # Small helpers
    # ============================================================
    def update_conf_label(self, e=None):
        self.conf_value = float(self.conf_scale.value)
        self.conf_label.set_text(f"{self.conf_value:.2f}")

    def update_iou_label(self, e=None):
        self.iou_value = float(self.iou_scale.value)
        self.iou_label.set_text(f"{self.iou_value:.2f}")

    def update_status(self, message, color=None):
        self.status_label.set_text(message)
        self.status_label.style(replace=f"color:{COLORS.get(color, color) if color else COLORS['fg']}")

    def alert(self, kind, title, message):
        """Modal message box (replacement for tkinter messagebox)."""
        icon, color = {
            'info': ('info', COLORS['sky']),
            'warning': ('warning', COLORS['warning']),
            'error': ('error', COLORS['error']),
            'success': ('check_circle', COLORS['success']),
        }[kind]
        with ui.dialog() as d, ui.card().classes('rounded-xl').style(
                f'min-width:360px; border-top:6px solid {color}'):
            with ui.row().classes('items-center gap-2'):
                ui.icon(icon, size='28px').style(f'color:{color}')
                ui.label(title).classes('text-lg font-bold').style(f'color:{color}')
            ui.label(message).style('white-space:pre-line')
            with ui.row().classes('w-full justify-end'):
                ui.button('OK', on_click=d.close, color=color).props('unelevated no-caps rounded')
        d.on('hide', lambda: d.delete())
        d.open()

    async def ask_yes_no(self, title, message):
        with ui.dialog() as d, ui.card().classes('rounded-xl').style(
                f"min-width:380px; border-top:6px solid {COLORS['accent']}"):
            ui.label(title).classes('text-lg font-bold').style(f"color:{COLORS['accent']}")
            ui.label(message).style('white-space:pre-line')
            with ui.row().classes('w-full justify-end'):
                ui.button('Yes', on_click=lambda: d.submit(True), color=COLORS['green']) \
                    .props('unelevated no-caps rounded')
                ui.button('No', on_click=lambda: d.submit(False), color=COLORS['danger']) \
                    .props('unelevated no-caps rounded')
        result = await d
        d.delete()
        return bool(result)

    async def native_dialog(self, kind, file_types=(), directory='', save_filename=''):
        """Native OS file/folder dialog through pywebview. Returns a path string or None."""
        win = getattr(app.native, 'main_window', None)
        if win is None:
            self.alert('warning', 'Warning',
                       'Native file dialogs need desktop mode (ui.run(native=True)).')
            return None
        res = await win.create_file_dialog(
            dialog_const(kind), directory=directory,
            save_filename=save_filename, file_types=tuple(file_types))
        if not res:
            return None
        if isinstance(res, (list, tuple)):
            res = res[0]
        return str(res)

    def _show_uri(self, page, uri):
        page.placeholder.set_visibility(False)
        page.image.set_visibility(True)
        page.image.set_source(uri)

    def _require_model(self):
        if not self.model:
            self.alert('warning', 'Warning', "Please load a model first!")
            return False
        return True

    def _begin_job(self, page, show_progress):
        self.is_processing = True
        self.stop_video = False
        page.select_btn.set_enabled(False)
        if page.shuffle_btn:
            page.shuffle_btn.set_enabled(False)
        if page.stop_btn:
            page.stop_btn.set_enabled(True)
        if page.pause_btn:
            self.paused = False
            page.pause_btn.set_enabled(True)
            page.pause_btn.set_text("⏸ Pause")
        if page.mode == 'webcam':
            page.save_btn.set_enabled(True)
        if show_progress:
            page.progress.set_visibility(True)
        if page.stream is not None:
            self._start_stream(page)

    def _start_stream(self, page):
        page.jpeg = None
        page.stream_active = True
        page.placeholder.set_visibility(False)
        page.stream.props(f'src="/stream/{page.mode}?t={int(time.time() * 1000)}"')
        page.stream.set_visibility(True)

    def _end_job(self, page):
        page.progress.set_visibility(False)
        page.select_btn.set_enabled(True)
        if page.shuffle_btn and self.folder_path:
            page.shuffle_btn.set_enabled(True)
        if page.stop_btn:
            page.stop_btn.set_enabled(False)
        if page.pause_btn:
            self.paused = False
            page.pause_btn.set_enabled(False)
            page.pause_btn.set_text("⏸ Pause")
        self.is_processing = False

    def _tick(self):
        """UI-thread timer: pulls status/completion from worker threads."""
        if self._status is not None:
            text, color = self._status
            self._status = None
            self.update_status(text, color)
        self._poll_video_done()

    # ============================================================
    # Camera detection / swapping
    # ============================================================
    def _probe_cameras(self):
        """Probe camera indices (runs in a worker thread). A camera only counts if
        it opens AND delivers a frame, so phantom/metadata devices are skipped."""
        backends = [CAM_BACKEND] if CAM_BACKEND == cv2.CAP_ANY else [CAM_BACKEND, cv2.CAP_ANY]
        found = []
        for backend in backends:
            for i in range(6):
                try:
                    cap = cv2.VideoCapture(i, backend)
                    ok = cap is not None and cap.isOpened()
                    if ok:
                        ok = False
                        for _attempt in range(3):
                            got, _frame = cap.read()
                            if got:
                                ok = True
                                break
                            time.sleep(0.05)
                    cap.release()
                    if ok:
                        found.append(i)
                except Exception:
                    pass
            if found:
                break
        return found or [0]

    async def _init_cameras(self):
        cams = await run.io_bound(self._probe_cameras)
        self.available_cameras = cams
        self.current_cam_index = cams[0]

    async def swap_camera(self):
        """Cycle to the next detected camera index.

        Rescans when idle (so cameras plugged in later are found), then:
        clean stop -> wait for the capture to fully release -> restart, so the
        old and new camera handles never overlap."""
        if self._swapping:
            return

        page = self.pages['webcam']
        if not self.is_processing:
            self.update_status("Scanning cameras...", 'warning')
            page.swap_btn.set_enabled(False)
            self.available_cameras = await run.io_bound(self._probe_cameras)
            page.swap_btn.set_enabled(True)

        if len(self.available_cameras) < 2:
            self.update_status("Only one camera detected", 'warning')
            ui.notify("Only one camera detected", type='warning')
            return

        try:
            idx = self.available_cameras.index(self.current_cam_index)
        except ValueError:
            idx = -1
        idx = (idx + 1) % len(self.available_cameras)
        new_index = self.available_cameras[idx]
        self.current_cam_index = new_index

        if self.webcam_source is None:
            self.update_status(
                f"Camera set to index {new_index} (used next time you select Webcam)",
                'accent')
            return

        self._swapping = True
        page.swap_btn.set_enabled(False)

        if self.is_processing:
            self.update_status(f"Switching to camera {new_index}...", 'warning')
            self.stop_video = True
            old_thread = self.video_thread
            if old_thread is not None:
                await run.io_bound(old_thread.join, 5)
            self._poll_video_done()  # make sure the finished job is cleaned up
            self._finish_camera_swap(new_index)
        else:
            self.current_file = self.webcam_source = new_index
            page.file_label.set_text(f"📹 Webcam {new_index}")
            page.file_label.style(replace=f"color:{COLORS['success']}")
            self.update_status(f"Switched to camera index {new_index}", 'accent')
            self._swapping = False
            page.swap_btn.set_enabled(True)

    def _finish_camera_swap(self, new_index):
        page = self.pages['webcam']
        self.current_file = self.webcam_source = new_index
        page.file_label.set_text(f"📹 Webcam {new_index}")
        page.file_label.style(replace=f"color:{COLORS['success']}")
        self._swapping = False
        page.swap_btn.set_enabled(True)
        self.update_status(f"Switched to camera index {new_index}", 'accent')
        # Seamlessly resume detection on the new camera
        self.run_video(page, self.current_file)

    # ============================================================
    # Model loading (ONNX)
    # ============================================================
    async def auto_load_model(self):
        """Auto-load model from models folder (external first, then bundled)"""
        model_path = find_default_model()

        if model_path is not None:
            self.update_status("Loading default model from models/yolo11n.onnx...", 'warning')
            await self.load_model_file(str(model_path))
        else:
            self.update_status("No model found in 'models' folder. Please select a model.", 'error')
            self.model_label.set_text("No model loaded")
            self.model_label.style(replace=f"color:{COLORS['error']}; word-break:break-all")

            if await self.ask_yes_no(
                    "Model Not Found",
                    "Model file 'models/yolo11n.onnx' not found.\n\n"
                    "Would you like to select an ONNX model file now?"):
                await self.load_model()

    async def load_model(self):
        """Load YOLO ONNX model from file dialog"""
        models_dir = app_dir() / 'models'
        file_path = await self.native_dialog(
            'OPEN',
            file_types=("ONNX Models (*.onnx)", "All files (*.*)"),
            directory=str(models_dir if models_dir.exists() else app_dir()))
        if not file_path:
            return
        await self.load_model_file(file_path)

    async def load_model_file(self, file_path):
        """Load ONNX model from given path"""
        try:
            self.update_status("Loading model...", 'warning')
            self.model = await run.io_bound(ONNXYOLO, file_path)
            self.model_path = file_path

            model_name = Path(file_path).name
            self.model_label.set_text(f"✓ {model_name}")
            self.model_label.style(replace=f"color:{COLORS['success']}; word-break:break-all")
            self.update_status(
                f"Model loaded successfully: {model_name}  ({self.model.providers[0]})", 'success')

        except Exception as e:
            self.alert('error', 'Error', f"Failed to load model:\n{str(e)}")
            self.model_label.set_text("❌ Failed to load")
            self.model_label.style(replace=f"color:{COLORS['error']}; word-break:break-all")
            self.update_status("Error loading model", 'error')

    async def reload_model(self):
        """Reload the current model"""
        if self.model_path:
            await self.load_model_file(self.model_path)
        else:
            self.alert('info', 'Info', "No model to reload. Please load a model first.")

    # ============================================================
    # Selection actions - each one starts detection immediately
    # ============================================================
    async def upload_image(self):
        """Select an image and run detection immediately"""
        if not self._require_model():
            return
        file_path = await self.native_dialog(
            'OPEN', file_types=("Image files (*.jpg;*.jpeg;*.png;*.bmp)", "All files (*.*)"))
        if not file_path:
            return

        page = self.pages['image']
        self.current_file = file_path
        page.file_label.set_text(Path(file_path).name)
        page.file_label.style(replace=f"color:{COLORS['success']}")
        self.update_status(f"Image loaded: {Path(file_path).name}", 'success')
        await self.run_image(page, file_path)

    async def upload_video(self):
        """Select a video and run detection immediately"""
        if not self._require_model():
            return
        file_path = await self.native_dialog(
            'OPEN', file_types=("Video files (*.mp4;*.avi;*.mov;*.mkv)", "All files (*.*)"))
        if not file_path:
            return

        page = self.pages['video']
        self.current_file = file_path
        page.file_label.set_text(Path(file_path).name)
        page.file_label.style(replace=f"color:{COLORS['success']}")
        self.update_status(f"Video loaded: {Path(file_path).name}", 'success')
        self.run_video(page, file_path)

    def use_webcam(self):
        """Use webcam for detection - starts immediately"""
        if not self._require_model():
            return
        page = self.pages['webcam']
        self.current_file = self.webcam_source = self.current_cam_index
        page.file_label.set_text(f"📹 Webcam {self.current_cam_index}")
        page.file_label.style(replace=f"color:{COLORS['success']}")
        self.update_status(f"Webcam {self.current_cam_index} selected", 'success')
        self.run_video(page, self.current_file)

    async def upload_folder(self):
        """Select a folder, randomly pick 4 images and run batch detection"""
        if not self._require_model():
            return
        folder = await self.native_dialog('FOLDER')
        if not folder:
            return

        images = [p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            self.alert('warning', 'Warning', "No images found in the selected folder!")
            return

        page = self.pages['folder']
        self.folder_path = folder
        self.folder_images = images
        page.file_label.set_text(f"{Path(folder).name}  ({len(images)} images)")
        page.file_label.style(replace=f"color:{COLORS['success']}")
        self.update_status(f"Folder loaded: {Path(folder).name} ({len(images)} images)", 'success')
        await self.shuffle_folder()

    async def shuffle_folder(self):
        """Pick a new random set of up to 4 images and run detection"""
        if not self._require_model() or not self.folder_images:
            return
        page = self.pages['folder']
        picks = random.sample(self.folder_images, min(GRID_COUNT, len(self.folder_images)))
        await self.run_folder(page, picks)

    # ============================================================
    # Image inference
    # ============================================================
    async def run_image(self, page, path):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status("Processing image...", 'warning')
        conf, iou = self.conf_value, self.iou_value
        try:
            result, annotated_bgr, uri = await run.io_bound(self._image_work, path, conf, iou)
        except Exception as e:
            traceback.print_exc()
            self._job_failed(page, "Detection failed", str(e))
            return
        self._image_done(page, result, annotated_bgr, uri)

    def _image_work(self, path, conf, iou):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError("Could not read the selected image file")
        result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
        return result, annotated_bgr, to_data_uri(annotated_bgr)

    def _image_done(self, page, result, annotated_bgr, uri):
        self._show_uri(page, uri)
        self.display_results(page, result)
        page.last_result = annotated_bgr
        page.save_btn.set_enabled(True)
        self.update_status(f"✓ Detection complete: {len(result.boxes)} objects found", 'success')
        self._end_job(page)

    def _job_failed(self, page, status, detail):
        self.alert('error', 'Error', f"{status}:\n{detail}")
        self.update_status(status, 'error')
        self._end_job(page)

    # ============================================================
    # Folder (batch) inference -> 2x2 grid
    # ============================================================
    async def run_folder(self, page, paths):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status(f"Processing {len(paths)} images...", 'warning')
        conf, iou = self.conf_value, self.iou_value
        try:
            entries, grid_bgr, uri = await run.io_bound(self._folder_work, paths, conf, iou)
        except Exception as e:
            traceback.print_exc()
            self._job_failed(page, "Batch detection failed", str(e))
            return
        self._folder_done(page, entries, grid_bgr, uri)

    def _folder_work(self, paths, conf, iou):
        entries, annotated_list, names_list = [], [], []
        for p in paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
            entries.append((p.name, result))
            annotated_list.append(annotated_bgr)
            names_list.append(p.name)
        if not entries:
            raise ValueError("None of the selected images could be read")

        grid_bgr = build_grid(annotated_list, names_list)
        return entries, grid_bgr, to_data_uri(grid_bgr)

    def _folder_done(self, page, entries, grid_bgr, uri):
        self._show_uri(page, uri)
        self.display_folder_results(page, entries)
        page.last_result = grid_bgr
        page.save_btn.set_enabled(True)
        total = sum(len(r.boxes) for _, r in entries)
        self.update_status(
            f"✓ Detection complete: {total} objects found in {len(entries)} images",
            'success')
        self._end_job(page)

    # ============================================================
    # Video / webcam inference (worker thread + UI timer hand-off)
    # ============================================================
    def run_video(self, page, source):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=False)
        self.video_thread = threading.Thread(target=self._video_worker,
                                             args=(page, source), daemon=True)
        self.video_thread.start()

    def _video_worker(self, page, source):
        frame_count = 0
        total_detections = 0
        error = None
        open_failed = False
        fps = 0.0
        last_status = 0.0
        try:
            cap = open_capture(source)
            if not cap.isOpened():
                cap.release()
                open_failed = True
            else:
                t_prev = time.perf_counter()
                while cap.isOpened() and not self.stop_video:
                    if self.paused:
                        time.sleep(0.05)
                        t_prev = time.perf_counter()
                        continue
                    t0 = time.perf_counter()
                    ret, frame = cap.read()
                    t1 = time.perf_counter()
                    if not ret:
                        break

                    frame_count += 1
                    result, annotated_bgr = self.model.infer(
                        frame, conf=self.conf_value, iou=self.iou_value)
                    t2 = time.perf_counter()
                    total_detections += len(result.boxes)
                    if self.stop_video:
                        break

                    # smoothed FPS of the whole read + infer + encode loop
                    now = time.perf_counter()
                    dt = now - t_prev
                    t_prev = now
                    inst = (1.0 / dt) if dt > 0 else 0.0
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

                    page.last_result = annotated_bgr          # clean copy for Capture
                    shown = draw_fps(fit_width(annotated_bgr, 1280), fps)
                    page.jpeg = to_jpeg_bytes(shown, max_w=1280, quality=80)
                    page.ver += 1
                    t3 = time.perf_counter()
                    if now - last_status > 0.25:              # throttle status updates
                        last_status = now
                        self._status = (
                            f"Frame {frame_count} - {len(result.boxes)} objects detected"
                            f"  |  read {(t1 - t0) * 1000:.0f} ms · "
                            f"infer {(t2 - t1) * 1000:.0f} ms · "
                            f"draw+encode {(t3 - t2) * 1000:.0f} ms",
                            'warning')

                    time.sleep(0.001)

                cap.release()
        except Exception as e:
            traceback.print_exc()
            error = str(e)

        page.stream_active = False   # lets the MJPEG response finish after the last frame
        stopped = self.stop_video or open_failed
        self._video_payload = (page, frame_count, total_detections, stopped, error, open_failed)

    def _poll_video_done(self):
        payload = self._video_payload
        if payload is None:
            return
        self._video_payload = None
        page, frame_count, total_detections, stopped, error, open_failed = payload

        self._end_job(page)
        if page.mode == 'webcam' and page.save_btn:
            page.save_btn.set_enabled(False)

        if open_failed:
            self.alert('error', 'Error', "Failed to open video source!")

        if error:
            self.alert('error', 'Error', f"Video processing failed:\n{error}")
            self.update_status("Video processing failed", 'error')
        elif self._swapping:
            pass  # camera swap in progress; status handled there
        elif not stopped:
            self.update_status(
                f"✓ Video complete: {frame_count} frames, {total_detections} total detections",
                'success')
            self.alert('info', 'Complete',
                       f"Video processing complete!\n"
                       f"Frames: {frame_count}\n"
                       f"Total detections: {total_detections}")
        else:
            self.update_status("Video processing stopped", 'warning')

    def toggle_pause(self):
        """Play/pause toggle for video playback"""
        page = self.pages['video']
        if not self.is_processing:
            return
        self.paused = not self.paused
        if self.paused:
            page.pause_btn.set_text("▶ Play")
            self.update_status("Video paused", 'warning')
        else:
            page.pause_btn.set_text("⏸ Pause")
            self.update_status("Video playing", 'accent')

    def capture_frame(self):
        """Save the current annotated webcam frame instantly (no dialog)"""
        page = self.pages['webcam']
        if page.last_result is None:
            self.alert('warning', 'Warning', "No frame to capture yet!")
            return
        try:
            out_dir = app_dir() / 'captures'
            out_dir.mkdir(exist_ok=True)
            file_path = out_dir / f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            frame = page.last_result.copy()
            cv2.imwrite(str(file_path), frame)
            self.update_status(f"📸 Captured: {file_path}", 'success')
        except Exception as e:
            self.alert('error', 'Error', f"Failed to capture frame:\n{str(e)}")

    def stop_detection(self):
        """Stop video processing"""
        self.paused = False
        self.stop_video = True
        self.update_status("Stopping...", 'warning')

    # ============================================================
    # Results panel (image + folder pages only)
    # ============================================================
    def _clear_results(self, page):
        page.result_col.clear()

    def show_no_results(self, page):
        self._clear_results(page)
        hint = {
            'image': "No detections yet\n\nSelect an image\nto run detection",
            'folder': "No detections yet\n\nSelect a folder\nto run batch detection",
        }[page.mode]
        with page.result_col:
            ui.label(hint).classes('w-full text-center text-sm mt-12').style(
                f"white-space:pre-line; color:{COLORS['fg_dim']}")

    def _conf_color(self, v):
        return (COLORS['success'] if v > 0.7
                else COLORS['warning'] if v > 0.4 else COLORS['error'])

    def _mini_card(self):
        return ui.card().classes('w-full no-shadow rounded-lg gap-1 py-2 px-3').style(
            f"background:{COLORS['soft']}; border:1px solid {COLORS['border']}")

    def _total_card(self, page, count, subtitle="Objects Detected"):
        accent = MODE_COLORS[page.mode]
        with ui.card().classes('w-full items-center no-shadow rounded-xl gap-0').style(
                f"background:{accent}14; border:2px solid {accent}"):
            ui.label(f"{count}").classes('text-4xl font-bold').style(f'color:{accent}')
            ui.label(subtitle).classes('text-xs').style(f"color:{COLORS['fg_dim']}")

    def _section_header(self, page, text):
        ui.label(text).classes('text-sm font-bold mt-2').style(f"color:{MODE_COLORS[page.mode]}")

    def display_results(self, page, result):
        """Display detection results in structured format"""
        self._clear_results(page)
        boxes = result.boxes

        with page.result_col:
            self._total_card(page, len(boxes))

            if len(boxes) == 0:
                ui.label("No objects detected").classes('w-full text-center text-xs mt-4').style(
                    f"color:{COLORS['fg_dim']}")
                return

            self._section_header(page, "📈 Summary by Class")

            class_counts = {}
            for box in boxes:
                class_name = result.names[int(box.cls[0])]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                self.create_summary_card(page, class_name, count)

            ui.separator().classes('my-2')

            self._section_header(page, "🔍 Detailed Detections")
            for i, box in enumerate(boxes, 1):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0]
                self.create_detection_card(page, i, class_name, conf, x1, y1, x2, y2)

    def display_folder_results(self, page, entries):
        """Analysis for the 2x2 grid: overall totals, class summary across all
        images, then one card per image (count, avg/max confidence, classes)."""
        self._clear_results(page)

        total = sum(len(r.boxes) for _, r in entries)
        all_confs = [float(b.conf[0]) for _, r in entries for b in r.boxes]
        avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0

        with page.result_col:
            self._total_card(page, total, f"Objects Detected in {len(entries)} Images")

            if total == 0:
                ui.label("No objects detected").classes('w-full text-center text-xs mt-4').style(
                    f"color:{COLORS['fg_dim']}")
            else:
                # Overall stats row
                with ui.row().classes('w-full no-wrap justify-around rounded-lg py-2').style(
                        f"background:{COLORS['success']}14; border:1px solid {COLORS['success']}55"):
                    for label, value in (("Avg Confidence", f"{avg_conf * 100:.1f}%"),
                                         ("Max Confidence", f"{max(all_confs) * 100:.1f}%"),
                                         ("Avg / Image", f"{total / len(entries):.1f}")):
                        with ui.column().classes('items-center gap-0'):
                            ui.label(value).classes('text-sm font-bold').style(
                                f"color:{COLORS['success']}")
                            ui.label(label).style(f"font-size:10px; color:{COLORS['fg_dim']}")

                self._section_header(page, "📈 Summary by Class")
                class_counts = {}
                for _, r in entries:
                    for b in r.boxes:
                        n = r.names[int(b.cls[0])]
                        class_counts[n] = class_counts.get(n, 0) + 1
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    self.create_summary_card(page, class_name, count)

            ui.separator().classes('my-2')

            self._section_header(page, "🔍 Detailed Detections")
            for i, (name, r) in enumerate(entries, 1):
                self.create_image_card(page, i, name, r)

    def create_summary_card(self, page, class_name, count):
        """Create a summary card for each class"""
        accent = MODE_COLORS[page.mode]
        with self._mini_card():
            with ui.row().classes('w-full items-center justify-between no-wrap'):
                ui.label(class_name.capitalize()).classes('text-sm font-bold')
                ui.label(str(count)).classes('px-3 py-1 rounded-lg text-white text-sm font-bold').style(
                    f'background:{accent}')

    def create_image_card(self, page, index, name, result):
        """Per-image analysis card used by folder mode"""
        accent = MODE_COLORS[page.mode]
        boxes = result.boxes
        with self._mini_card():
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                ui.label(f"#{index}").classes('text-xs font-bold').style(f'color:{accent}')
                short = name if len(name) <= 26 else name[:23] + "..."
                ui.label(short).classes('text-sm font-bold col')
                ui.label(str(len(boxes))).classes('px-2 py-0 rounded-lg text-white text-sm font-bold').style(
                    f'background:{accent}')

            if not boxes:
                ui.label("No objects detected").classes('text-xs').style(f"color:{COLORS['fg_dim']}")
                return

            confs = [float(b.conf[0]) for b in boxes]
            avg = sum(confs) / len(confs)
            counts = {}
            for b in boxes:
                n = result.names[int(b.cls[0])]
                counts[n] = counts.get(n, 0) + 1
            cls_text = ", ".join(f"{n.capitalize()} ×{c}" for n, c in
                                 sorted(counts.items(), key=lambda x: x[1], reverse=True))

            conf_color = self._conf_color(avg)
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                ui.label("Confidence:").classes('text-xs').style(f"color:{COLORS['fg_dim']}")
                ui.linear_progress(value=max(0.0, min(1.0, avg)), show_value=False,
                                   color=conf_color).props('rounded size=8px').style('width:90px')
                ui.label(f"avg {avg * 100:.1f}% · max {max(confs) * 100:.1f}%").classes(
                    'text-xs font-bold').style(f'color:{conf_color}')

            ui.label(cls_text).classes('text-xs').style(f"color:{COLORS['fg_dim']}")

    def create_detection_card(self, page, index, class_name, conf, x1, y1, x2, y2):
        """Create a detection card for individual detection"""
        accent = MODE_COLORS[page.mode]
        conf_pct = max(0.0, min(1.0, conf))
        conf_color = self._conf_color(conf_pct)
        with self._mini_card():
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                ui.label(f"#{index}").classes('text-xs font-bold').style(f'color:{accent}')
                ui.label(class_name.capitalize()).classes('text-sm font-bold col')

            with ui.row().classes('w-full items-center no-wrap gap-2'):
                ui.label("Confidence:").classes('text-xs').style(f"color:{COLORS['fg_dim']}")
                ui.linear_progress(value=conf_pct, show_value=False,
                                   color=conf_color).props('rounded size=8px').style('width:110px')
                ui.label(f"{conf_pct * 100:.1f}%").classes('text-xs font-bold').style(
                    f'color:{conf_color}')

            ui.label(f"Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})").classes('text-xs').style(
                f"font-family:'Courier New',monospace; color:{COLORS['fg_dim']}")

    # ============================================================
    # Save / Clear
    # ============================================================
    async def save_result(self):
        """Save detection result (image or the 2x2 grid)"""
        page = self.active_page
        if page is None or page.last_result is None:
            self.alert('warning', 'Warning', "No result to save!")
            return

        file_path = await self.native_dialog(
            'SAVE', save_filename='result.jpg',
            file_types=("JPEG (*.jpg)", "PNG (*.png)", "All files (*.*)"))
        if not file_path:
            return
        if not Path(file_path).suffix:
            file_path += '.jpg'

        try:
            cv2.imwrite(file_path, page.last_result)
            self.alert('success', 'Success', f"Result saved to:\n{file_path}")
            self.update_status(f"✓ Result saved: {Path(file_path).name}", 'success')
        except Exception as e:
            self.alert('error', 'Error', f"Failed to save result:\n{str(e)}")

    def clear_display(self):
        """Clear display and results of the active page"""
        page = self.active_page
        if self.is_processing:
            self.paused = False
            self.stop_video = True

        if page.stream is not None:
            page.stream_active = False
            page.jpeg = None
            page.stream.props(remove='src')
            page.stream.set_visibility(False)
        else:
            page.image.set_visibility(False)
            page.image.set_source('')
        page.placeholder.set_visibility(True)

        if page.result_col is not None:
            self.show_no_results(page)

        page.last_result = None
        if page.mode == 'folder':
            self.folder_path = None
            self.folder_images = []
            page.shuffle_btn.set_enabled(False)
        else:
            self.current_file = None
            if page.mode == 'webcam':
                self.webcam_source = None
        page.file_label.set_text("No file selected")
        page.file_label.style(replace=f"color:{COLORS['error']}")
        if page.save_btn and page.mode != 'webcam':
            page.save_btn.set_enabled(False)
        self.update_status("Display cleared", 'fg_dim')


# ============================================================
# Entry point
# ============================================================
@ui.page('/')
def index():
    DetectorApp().build()


def main():
    """Main application entry point"""
    disable_power_throttling()
    app.native.window_args['min_size'] = (1100, 700)
    ui.run(
        title="YOLO Object Detection - NiceGUI Desktop (ONNX)",
        native=True,
        window_size=(1400, 850),
        reload=False,
        dark=False,
        favicon='📟',
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()