"""Leafora - Kivy front end (UI + camera capture only).

Run on desktop:  python main.py
The layout lives in leafora.kv; theme.py holds colors/icons;
api_client.py is the only file that talks to the Flask back end.
"""
import math
import os
import threading
import time

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, PopMatrix, PushMatrix, Rotate, Scale
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import (BooleanProperty, ColorProperty, ListProperty,
                             NumericProperty, ObjectProperty, StringProperty)
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image as KImage
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.widget import Widget
from kivy.utils import platform

import api_client
import dummy_sensors
import theme as T
from kivy.resources import resource_add_path

# Makes paths like 'Icons/bell.png' in the .kv work no matter which folder
# you launch python from (VS Code often runs from the parent folder).
resource_add_path(os.path.dirname(os.path.abspath(__file__)))

# The cards are nested and size themselves to their content, so the first layout
# needs more passes than Kivy's default (20) allows in one frame. This removes
# the "too much iteration" CRITICAL warning; it does not change how the app looks.
Clock.max_iteration = 60

# Windows webcams: MSMF backend is often very slow / returns no frames.
# Must be set before OpenCV is imported (Kivy imports it lazily).
if platform == 'win':
    os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_MSMF', '0')

T.setup_fonts()
Window.clearcolor = T.CREAM

# Desktop preview roughly the shape of the Infinix Hot 50 (20:9 phone).
# On the phone, dp() + size_hint make everything scale automatically.
if platform not in ('android', 'ios'):
    Window.size = (390, 800)

# --- camera settings you may need to tweak per phone -------------------------
CAMERA_INDEX = 0                                   # 0 = back, 1 = front
CAMERA_ROTATION = -90 if platform == 'android' else 0
FLIP_SAVED = True                                  # set False if saved photo is upside-down


# =============================================================== small widgets
class Card(BoxLayout):
    bg_color = ColorProperty(T.CREAM)
    edge_color = ColorProperty([0, 0, 0, 0])
    radius = NumericProperty(dp(24))
    fixed_height = NumericProperty(0)      # >0 = force this height, 0 = fit content
    outline = NumericProperty(0)           # >0 = draw a full dark outline of this width


class TapCard(ButtonBehavior, Card):
    pass


class Circle(AnchorLayout):
    bg_color = ColorProperty(T.MINT)


class TapCircle(ButtonBehavior, Circle):
    pass


class Pill(Label):
    bg_color = ColorProperty(T.MINT_L)
    edge_color = ColorProperty([0, 0, 0, 0])


def icon_texture(name):
    """Texture of Icons/<name>.png, or None when the file does not exist."""
    path = T.icon_image(name) if name else ''
    if not path:
        return None
    try:
        return CoreImage(path).texture
    except Exception:
        return None


class Btn(ButtonBehavior, Label):
    bg_color = ColorProperty(T.PINK)
    edge_color = ColorProperty(T.DARK)
    outline = NumericProperty(0)
    icon = StringProperty('')            # optional Icons/<icon>.png drawn left of the text
    icon_tex = ObjectProperty(None, allownone=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(icon=self._set_icon)
        self._set_icon()

    def _set_icon(self, *a):
        self.icon_tex = icon_texture(self.icon)


class Chip(ToggleButtonBehavior, Label):
    icon = StringProperty('')            # optional Icons/<icon>.png
    icon_tex = ObjectProperty(None, allownone=True)
    icon_pos = StringProperty('top')     # 'top' = icon above text, 'left' = icon beside text
    sel_color = ColorProperty(T.MINT)    # fill color when selected

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(icon=self._set_icon)
        self._set_icon()

    def _set_icon(self, *a):
        self.icon_tex = icon_texture(self.icon)


class Swatch(ToggleButtonBehavior, Widget):
    tint = ColorProperty(T.MINT)


class TabBtn(ToggleButtonBehavior, Label):
    pass


class ShutterButton(ButtonBehavior, FloatLayout):
    pass


class Bar(Widget):
    value = NumericProperty(0.5)
    fill = ColorProperty(T.MINT)
    track = ColorProperty(T.MINT_L)


class Icon(Label):
    name = StringProperty('')
    source = StringProperty('')          # path of Icons/<name>.png if it exists
    recolor = ListProperty([])           # e.g. [1, 1, 1, 1] paints the PNG white
    tex = ObjectProperty(None, allownone=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(name=self._refresh, recolor=self._refresh)
        self._refresh()

    def _load(self, path):
        """Texture of the PNG; optionally recolored (keeps the PNG's shape/alpha)."""
        try:
            if len(self.recolor) >= 3:
                from PIL import Image as PILImage, ImageOps
                img = PILImage.open(path).convert('RGBA')
                alpha = img.getchannel('A')
                if alpha.getextrema()[0] == 255:      # PNG has no transparency:
                    alpha = ImageOps.invert(img.convert('L'))   # use darkness as shape
                rgb = tuple(int(c * 255) for c in self.recolor[:3])
                solid = PILImage.new('RGBA', img.size, rgb + (255,))
                solid.putalpha(alpha)
                tex = Texture.create(size=solid.size, colorfmt='rgba')
                tex.blit_buffer(solid.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
                tex.flip_vertical()
                return tex
            return CoreImage(path).texture
        except Exception:
            return None

    def _refresh(self, *a):
        self.source = T.icon_image(self.name)
        if self.source:                  # your own image wins over font/fallback
            self.text = ''
            self.tex = self._load(self.source)
            return
        self.tex = None
        glyph, fallback = T.ICONS.get(self.name, ('', ''))
        if T.ICON_FONT:
            self.font_name = T.ICON_FONT
            self.text = glyph
        else:
            self.text = fallback


class NavItem(ButtonBehavior, BoxLayout):
    icon = StringProperty('home')
    text = StringProperty('')
    key = StringProperty('')
    selected = BooleanProperty(False)


class NavBar(BoxLayout):
    active = StringProperty('garden')


class ToggleSwitch(Widget):
    """Pill switch (dark-green track, white knob). Drawn in leafora.kv."""
    active = BooleanProperty(False)
    knob = NumericProperty(0)            # 0 = left, 1 = right (animated)

    def on_active(self, inst, value):
        Animation.cancel_all(self, 'knob')
        Animation(knob=1. if value else 0., d=.16, t='out_quad').start(self)


class SettingRow(ButtonBehavior, BoxLayout):
    """One row of a Settings card: icon, title, subtitle, then a value / chevron / switch."""
    icon = StringProperty('')            # Icons/<icon>.png
    title = StringProperty('')
    subtitle = StringProperty('')
    value = StringProperty('')           # small bold text before the chevron, e.g. 'Light'
    chevron = BooleanProperty(True)
    has_switch = BooleanProperty(False)  # True = show the on/off switch instead of a chevron
    active = BooleanProperty(False)      # the switch state
    divider = BooleanProperty(True)      # thin line under the row

    def on_release(self):
        if self.has_switch:
            self.active = not self.active


class LinkLabel(ButtonBehavior, Label):
    """Small tappable text, e.g. 'Privacy policy'."""


class GuideOverlay(Widget):
    """Corner brackets + guide rings over the camera preview (drawn in leafora.kv)."""


class VitalityChart(Widget):
    values = ListProperty([95, 91, 78, 85, 92])
    lo = NumericProperty(70)
    hi = NumericProperty(100)
    threshold = NumericProperty(90)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(pos=self.draw, size=self.draw, values=self.draw)

    def draw(self, *a):
        self.canvas.clear()
        n = len(self.values)
        if n < 2 or self.width < 10:
            return
        padv = dp(14)
        h = self.height - 2 * padv

        def pt(i, v):
            return (self.x + self.width * (i + 0.5) / n,
                    self.y + padv + h * (v - self.lo) / (self.hi - self.lo))

        pts = [pt(i, v) for i, v in enumerate(self.values)]
        low = self.values.index(min(self.values))
        ty = pt(0, self.threshold)[1]
        with self.canvas:
            Color(*T.GUIDE)
            Line(points=[self.x, ty, self.right, ty], width=dp(1),
                 dash_length=dp(4), dash_offset=dp(3))
            Color(*T.DARK)
            Line(points=[c for p in pts for c in p], width=dp(2.2), joint='round')
            for i, (x, y) in enumerate(pts):
                r = dp(7)
                Color(*(T.RED if i == low else T.DARK))
                Ellipse(pos=(x - r, y - r), size=(2 * r, 2 * r))
                Color(*T.CREAM)
                r2 = dp(3.5)
                Ellipse(pos=(x - r2, y - r2), size=(2 * r2, 2 * r2))


# ============================================================ camera (the task)
def ensure_camera_permission(callback):
    """callback(bool) on the main thread. Desktop: always True."""
    if platform != 'android':
        callback(True)
        return
    from android.permissions import Permission, check_permission, request_permissions
    if check_permission(Permission.CAMERA):
        callback(True)
        return
    request_permissions(
        [Permission.CAMERA],
        lambda perms, grants: Clock.schedule_once(lambda dt: callback(all(grants))))


USE_CV = platform not in ('android', 'ios')


class LeafCamera(FloatLayout):
    """Live preview + capture.

    Desktop : plain OpenCV (Kivy 2.3.0's own OpenCV camera provider is broken).
    Android : Kivy's Camera widget.
    """
    status = StringProperty('Tap here to open the camera')
    zoom = NumericProperty(1.0)      # 1.0 = normal, 1.2 = macro (closer crop)
    _scale = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cam = None            # android: kivy Camera widget
        self._rot = None
        self._tries = 0
        self._cap = None           # desktop: cv2.VideoCapture
        self._frame = None
        self._tex = None
        self._img = None
        self._ev = None
        self._opening = False
        self._misses = 0
        self.bind(size=self._layout, pos=self._layout)

    @property
    def is_on(self):
        if USE_CV:
            return self._ev is not None or self._opening
        return self.cam is not None and bool(self.cam.play)

    # ------------------------------------------------------- flash / macro / light
    def on_zoom(self, inst, value):
        if self._scale is not None:            # android preview
            self._scale.x = self._scale.y = value

    def set_macro(self, on):
        """Macro = 1.2x closer crop (preview + saved photo) and close-focus where supported."""
        self.zoom = 1.2 if on else 1.0
        if on:
            return self._set_android_mode('focus', 'macro', ('auto',))
        return self._set_android_mode('focus', 'continuous-picture', ('auto',))

    def set_torch(self, on):
        """Phone flashlight. Returns False when there is no flash (desktop webcams)."""
        return self._set_android_mode('flash', 'torch' if on else 'off')

    def brightness(self):
        """Average brightness 0-255 of the live picture, or None when unknown."""
        try:
            if USE_CV:
                return float(self._frame.mean()) if self._frame is not None else None
            if self.cam is None or self.cam.texture is None:
                return None
            sample = self.cam.texture.pixels[::997]          # cheap sample of RGBA bytes
            rgb = [b for i, b in enumerate(sample) if i % 4 != 3]
            return sum(rgb) / len(rgb) if rgb else None
        except Exception:
            return None

    def _android_cam(self):
        """The raw android.hardware.Camera behind Kivy's Camera widget (or None)."""
        try:
            return self.cam._camera._android_camera
        except Exception:
            return None

    def _set_android_mode(self, kind, wanted, fallbacks=()):
        cam = self._android_cam()
        if cam is None:
            return False
        try:
            params = cam.getParameters()
            supported = (params.getSupportedFlashModes() if kind == 'flash'
                         else params.getSupportedFocusModes())
            if supported is None:
                return False
            for mode in (wanted,) + tuple(fallbacks):
                if supported.contains(mode):
                    if kind == 'flash':
                        params.setFlashMode(mode)
                    else:
                        params.setFocusMode(mode)
                    cam.setParameters(params)
                    return True
        except Exception as exc:
            print('CAMERA PARAM ERROR:', exc)
        return False

    # ------------------------------------------------------------ public API
    def start(self):
        if USE_CV:
            self._cv_start()
        elif self.cam is not None:
            self.cam.play = True
        else:
            ensure_camera_permission(self._create)

    def stop(self):
        if USE_CV:
            self._cv_stop()
        elif self.cam is not None:
            self.cam.play = False

    def capture(self, folder):
        """Save the current frame as JPG. Returns the file path or None."""
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, 'scan_' + time.strftime('%Y%m%d_%H%M%S'))
        if USE_CV:
            if self._frame is None:
                return None
            import cv2
            ok, buf = cv2.imencode('.jpg', self._frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return None
            with open(base + '.jpg', 'wb') as f:
                f.write(buf.tobytes())
            return base + '.jpg'
        if self.cam is None or self.cam.texture is None:
            return None
        self.cam.texture.save(base + '.png', flipped=FLIP_SAVED)
        try:
            from PIL import Image
            img = Image.open(base + '.png').convert('RGB')
            if self.zoom > 1.01:                       # macro: keep the same area you saw
                w, h = img.size
                cw, ch = int(w / self.zoom), int(h / self.zoom)
                left, top = (w - cw) // 2, (h - ch) // 2
                img = img.crop((left, top, left + cw, top + ch))
            if CAMERA_ROTATION:
                img = img.rotate(CAMERA_ROTATION, expand=True)
            img.save(base + '.jpg', quality=88)
            os.remove(base + '.png')
            return base + '.jpg'
        except Exception:
            return base + '.png'

    # ------------------------------------------------------- desktop (OpenCV)
    def _cv_start(self):
        if self._ev is not None or self._opening:
            return
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.status = 'OpenCV not installed.\nRun: pip install opencv-python'
            return
        self._opening = True
        self.status = 'Opening camera...'
        threading.Thread(target=self._cv_open, daemon=True).start()

    def _cv_open(self):
        """Runs in a thread so the window never freezes while Windows wakes the webcam."""
        import cv2
        cap = None
        for api in (getattr(cv2, 'CAP_DSHOW', None), None):   # DirectShow first, then default
            c = cv2.VideoCapture(CAMERA_INDEX, api) if api is not None else cv2.VideoCapture(CAMERA_INDEX)
            if c.isOpened():
                cap = c
                break
            c.release()
        Clock.schedule_once(lambda dt: self._cv_ready(cap))

    def _cv_ready(self, cap):
        self._opening = False
        if cap is None:
            self.status = ('Cannot open webcam.\n'
                           'Close Zoom/Teams/Camera app and check\n'
                           'Windows Settings > Privacy > Camera')
            return
        self._cap = cap
        self._misses = 0
        if self._img is None:
            self._img = KImage(fit_mode='cover', size_hint=(1, 1), pos_hint={'x': 0, 'y': 0})
            self.add_widget(self._img)
        self.status = 'Waiting for frames...'
        self._ev = Clock.schedule_interval(self._cv_tick, 1 / 30.)

    def _cv_tick(self, dt):
        import cv2
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._misses += 1
            if self._misses > 90:                 # ~3 seconds of nothing
                self.status = 'Webcam opened but sends no frames.\nIs another app using it?'
            return
        if self.zoom > 1.01:                              # macro: centre crop
            fh, fw = frame.shape[:2]
            ch, cw = int(fh / self.zoom), int(fw / self.zoom)
            y0, x0 = (fh - ch) // 2, (fw - cw) // 2
            frame = frame[y0:y0 + ch, x0:x0 + cw].copy()
        h, w = frame.shape[:2]
        self._frame = frame
        if self._tex is None or self._tex.size != (w, h):
            self._tex = Texture.create(size=(w, h), colorfmt='bgr')
        self._tex.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        self._img.texture = self._tex
        self._img.canvas.ask_update()
        if self.status:
            self.status = ''

    def _cv_stop(self):
        if self._ev is not None:
            self._ev.cancel()
            self._ev = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------ Android (Kivy cam)
    def _create(self, granted):
        if not granted:
            self.status = 'Camera permission denied.\nEnable it in phone Settings.'
            return
        try:
            from kivy.uix.camera import Camera
            self.cam = Camera(index=CAMERA_INDEX, resolution=(640, 480), play=True,
                              size_hint=(None, None), fit_mode='cover')
        except Exception as exc:
            self.cam = None
            self.status = f'Camera unavailable\n({exc})'
            return
        with self.cam.canvas.before:
            PushMatrix()
            self._rot = Rotate(angle=CAMERA_ROTATION)
            self._scale = Scale(self.zoom, self.zoom, 1)
        with self.cam.canvas.after:
            PopMatrix()
        self.add_widget(self.cam)
        self.status = 'Waiting for camera...'
        self._layout()
        self._tries = 0
        Clock.schedule_interval(self._check_frames, 0.5)

    def _check_frames(self, dt):
        if self.cam is not None and self.cam.texture is not None:
            self.status = ''
            return False
        self._tries += 1
        if self._tries > 30:
            self.status = 'No camera frames.\nAllow Camera permission and restart.'
            return False
        return True

    def _layout(self, *a):
        if self.cam is None:
            return
        # a rotated camera needs its width/height swapped to still fill the box
        self.cam.size = (self.height, self.width) if CAMERA_ROTATION % 180 else self.size
        self.cam.center = self.center
        self._rot.origin = self.cam.center
        if self._scale is not None:
            self._scale.origin = self.cam.center


# ==================================================================== screens
class GardenScreen(Screen):
    pass


FLASH_ORDER = ('auto', 'on', 'off')
FLASH_TIPS = {                      # mode -> (title, body, badge)
    'auto': ('Flash: Auto', 'Light fires only when the leaf is too dark.', 'Auto'),
    'on': ('Flash: On', 'Light fires with every photo.', 'On'),
    'off': ('Flash: Off', 'No light. Use natural daylight.', 'Off'),
}
DARK_BRIGHTNESS = 70                # Auto flash fires below this (0-255)


class ScanScreen(Screen):
    tip_title = StringProperty('Center your leaf!')
    tip_body = StringProperty('Hold steady 4-6 inches from the plant.')
    badge = StringProperty('Ready')
    lux_text = StringProperty('Natural Light: 780 Lux')       # hook up a sensor later
    stability_text = StringProperty('Stability: 96%')
    busy = BooleanProperty(False)
    flash_mode = StringProperty('auto')      # 'auto' | 'on' | 'off'
    macro = BooleanProperty(False)

    def on_leave(self, *a):
        self.ids.cam.stop()
        self.ids.cam.status = 'Tap here to open the camera'

    def open_camera(self):
        # The camera only opens when the user taps it, not when the tab opens.
        if not self.ids.cam.is_on:
            self.ids.cam.start()

    def _tip(self, title, body, badge):
        self.tip_title, self.tip_body, self.badge = title, body, badge

    # ---------------------------------------------------------- flash / macro
    def cycle_flash(self):
        """Tap the flash button: Auto -> On -> Off -> Auto."""
        i = FLASH_ORDER.index(self.flash_mode)
        self.flash_mode = FLASH_ORDER[(i + 1) % len(FLASH_ORDER)]
        self._tip(*FLASH_TIPS[self.flash_mode])

    def toggle_macro(self):
        """Tap the macro button: closer 1.2x crop + close focus."""
        self.macro = not self.macro
        self.ids.cam.set_macro(self.macro)
        if self.macro:
            self._tip('Macro: On', 'Close-up mode. Hold 2-4 inches from the leaf.', 'Macro')
        else:
            self._tip('Macro: Off', 'Normal distance. Hold 4-6 inches from the plant.', 'Ready')

    def _wants_flash(self, cam):
        if self.flash_mode == 'on':
            return True
        if self.flash_mode == 'auto':
            level = cam.brightness()
            return level is not None and level < DARK_BRIGHTNESS
        return False

    def _screen_flash(self):
        """No flash hardware (PC webcam): flash the camera box white instead."""
        fx = self.ids.flash_fx
        Animation.cancel_all(fx, 'opacity')
        (Animation(opacity=.9, d=.08) + Animation(opacity=.9, d=.2)
         + Animation(opacity=0, d=.3)).start(fx)

    # ------------------------------------------------------------------ capture
    def capture(self):
        if self.busy:
            return
        cam = self.ids.cam
        if not cam.is_on:
            self._tip('Camera is off', 'Tap the camera box to open it first.', 'Off')
            return
        if self._wants_flash(cam):
            self.busy = True
            self._tip('Flash!', 'Hold still...', 'Busy')
            lit = cam.set_torch(True)            # real flashlight on phones
            if not lit:
                self._screen_flash()             # PC webcam fallback
            Clock.schedule_once(lambda dt: self._take(True), .5 if lit else .3)
        else:
            self._take(False)

    def _take(self, flashed):
        app = App.get_running_app()
        cam = self.ids.cam
        path = cam.capture(app.scan_dir)
        if flashed:
            cam.set_torch(False)
        if not path:
            self.busy = False
            self._tip('Camera not ready', 'Allow camera access, then try again.', 'Wait')
            return
        app.last_photo = path
        self.busy = True
        self._tip('Analyzing leaf...', 'Sending your photo to the Leafora server.', 'Busy')
        api_client.upload_scan(path, self._on_result, self._on_error)

    def _on_result(self, data):
        self.busy = False
        app = App.get_running_app()
        name = str(data.get('diagnosis', 'Unknown'))
        conf = data.get('confidence')
        body = 'Saved to your garden log.'
        if conf is not None:
            conf = float(conf)
            conf = conf / 100 if conf > 1 else conf
            app.diagnosis_conf = conf
            body = f'{conf * 100:.0f}% match - saved to your garden log.'
        app.diagnosis_title = name.upper()
        app.diagnosis_icon = 'healthytomato' if 'healthy' in name.lower() else 'unhealthytomato'
        self._tip(name, body, 'Done')

    def _on_error(self, msg):
        self.busy = False
        self._tip('Saved on phone', 'Server not reachable yet. Photo kept locally.', 'Offline')


class TagPill(BoxLayout):
    """Small rounded tag with an optional icon (Icons/<icon>.png) before the text."""
    text = StringProperty('')
    icon = StringProperty('')
    bg_color = ColorProperty(T.CREAM)
    edge_color = ColorProperty(T.DARK)


class EventRow(BoxLayout):
    icon = StringProperty('camera')
    when = StringProperty('')
    tag = StringProperty('')
    title = StringProperty('')
    body = StringProperty('')
    bg = ColorProperty(T.CREAM)
    dot = ColorProperty(T.PINK_L)          # color of the round icon badge


EVENTS = [
    dict(icon='camera', when='TODAY \u2022 09:15 AM', tag='Camera Scan',
         title='Scan Confirmed Recovery', bg=T.MINT_L, dot=T.MINT,
         body='Fresh top shoot growth detected! Vitality score increased by +7%. '
              'Lesion density dropped by 85% after targeted pruning.'),
    dict(icon='scissors', when='YESTERDAY \u2022 04:30 PM', tag='Gardener Care',
         title='Care Action: Pruned & Bottom Watered', bg=T.CREAM, dot=T.PINK_L,
         body='You clipped 3 infected lower leaves and placed tray irrigation to keep '
              'foliage crisp & dry. Completed via "I did it!" prompt.'),
    dict(icon='alert', when='2 DAYS AGO \u2022 10:12 AM', tag='Pathology Alert',
         title='Early Blight Suspected (88% Match)', bg=T.PEACH, dot=T.PEACH,
         body='Diagnostic scan identified concentric brown rings on 2 lower foliage '
              'stems. Isolation and lower canopy pruning recommended.'),
    dict(icon='antenna', when='4 DAYS AGO \u2022 02:00 AM', tag='IoT Sensor',
         title='Microclimate: Night Moisture Peak (92%)', bg=T.CREAM, dot=T.MINT,
         body='Persistent condensation recorded for 6 continuous night hours. Ideal '
              'incubation environment for fungal spore germination.'),
    dict(icon='milestone', when='DAY 35 \u2022 08:00 AM', tag='Milestone',
         title='Height Milestone: 24cm & Tiny Buds', bg=T.CREAM, dot=T.PINK_L,
         body='Sturdy main stalk logged. First tiny yellow floral cluster forming on '
              'the tertiary upper node.'),
]


class HistoryScreen(Screen):
    def on_kv_post(self, base):
        for ev in EVENTS:
            self.add_event(ev, top=False)

    def add_event(self, ev, top=True):
        tl = self.ids.timeline
        tl.add_widget(EventRow(**ev), index=len(tl.children) if top else 0)
        self.ids.event_count.text = f'{len(tl.children)} Events Logged'


class NoteNewScreen(Screen):
    hue = ColorProperty(T.MINT)
    activity = StringProperty('Pruning')
    tag = StringProperty('Gardener Care')

    def save(self):
        title = self.ids.title_in.text.strip() or 'Untitled note'
        body = self.ids.body_in.text.strip()
        hist = self.manager.get_screen('history')
        hist.add_event(dict(icon='scissors', when='TODAY \u2022 ' + time.strftime('%I:%M %p'),
                            tag=self.tag, title=title, body=body, bg=self.hue))
        self.manager.current = 'history'


class NoteDetailScreen(Screen):
    pass


class SettingsScreen(Screen):
    pass


class WipScreen(Screen):
    """Placeholder page for controls that are not built yet."""
    feature = StringProperty('This feature')


class RootSM(ScreenManager):
    pass


# ======================================================================== app
class LeaforaApp(App):
    title = 'Leafora'
    # These three are what the Flask result updates, so the dashboard reacts.
    diagnosis_title = StringProperty('EARLY BLIGHT SUSPECTED')
    diagnosis_conf = NumericProperty(0.88)
    diagnosis_icon = StringProperty('unhealthytomato')   # Icons/healthytomato.png | unhealthytomato.png
    last_photo = StringProperty('')

    # Weather card (filled from live weather; icon file = Icons/<weather_kind>.png)
    weather_kind = StringProperty('sunny')            # sunny | cloudy | rainy
    weather_temp = StringProperty('--')
    weather_desc = StringProperty('Checking weather...')
    weather_tag = StringProperty('...')
    # Vitals card (your IoT/Flask side can set these; the UI updates by itself)
    soil_moisture = NumericProperty(68)
    sensor_temp = NumericProperty(23)
    humidity = NumericProperty(62)

    # Settings screen (the .kv reads these; change them from Python and the UI follows)
    plot_name = StringProperty('Plot A \u2022 Sunny Balcony')
    pref_watering = BooleanProperty(True)
    pref_health = BooleanProperty(True)
    checkin_time = StringProperty('8:00 AM')
    quiet_hours = StringProperty('9:00 PM \u2013 7:00 AM')
    appearance = StringProperty('Light')
    units = StringProperty('\u00b0C \u00b7 Metric')
    language = StringProperty('English')
    # Connected garden device (fill from the ESP32 / backend later)
    node_name = StringProperty('Plot A node')
    node_synced = StringProperty('Last synced just now')
    node_battery = NumericProperty(86)
    node_wifi = StringProperty('Wi-Fi connected')
    node_live = BooleanProperty(True)

    @property
    def scan_dir(self):
        return os.path.join(self.user_data_dir, 'scans')

    def on_start(self):
        Window.softinput_mode = 'below_target'      # keyboard pushes the form up
        Window.bind(on_keyboard=self._on_key)
        self.load_prefs()
        self.refresh_weather()
        Clock.schedule_interval(lambda dt: self.refresh_weather(), 15 * 60)   # every 15 min
        if dummy_sensors.USE_DUMMY_SENSORS:
            self._dummy_feed = dummy_sensors.DummySensorFeed()
            self.refresh_dummy_vitals(0)
            Clock.schedule_interval(self.refresh_dummy_vitals, dummy_sensors.DUMMY_INTERVAL_SECONDS)

    # ---------------------------------------------------------- saved settings
    def _prefs_store(self):
        from kivy.storage.jsonstore import JsonStore
        return JsonStore(os.path.join(self.user_data_dir, 'prefs.json'))

    def load_prefs(self):
        """Restore the two reminder switches, then start saving them on every change."""
        try:
            store = self._prefs_store()
            if store.exists('toggles'):
                saved = store.get('toggles')
                self.pref_watering = bool(saved.get('watering', True))
                self.pref_health = bool(saved.get('health', True))
        except Exception as exc:
            print('PREFS LOAD ERROR:', exc)
        self.bind(pref_watering=self.save_prefs, pref_health=self.save_prefs)

    def save_prefs(self, *a):
        try:
            self._prefs_store().put('toggles', watering=self.pref_watering,
                                    health=self.pref_health)
        except Exception as exc:
            print('PREFS SAVE ERROR:', exc)

    def refresh_dummy_vitals(self, dt):
        """Fill the Vitals card with simulated readings (see dummy_sensors.py)."""
        reading = self._dummy_feed.next_reading()
        self.soil_moisture = reading['soil_moisture']
        self.sensor_temp = reading['temperature']
        self.humidity = reading['humidity']

    def refresh_weather(self):
        api_client.fetch_weather(self._weather_ok, self._weather_failed)

    def _weather_failed(self, msg):
        print('WEATHER ERROR:', msg)
        self.weather_desc = 'Weather unavailable'
        self.weather_temp = '--'
        self.weather_tag = '--'

    def _weather_ok(self, data):
        code, temp = data['code'], data['temp_c']
        print('WEATHER CODE:', code, temp)
        # WMO weather codes -> the 3 icons you have
        if code in (0, 1):
            self.weather_kind, self.weather_desc = 'sunny', 'Clear & Sunny'
        elif code == 2:                               # partly cloudy = still mostly sunny
            self.weather_kind, self.weather_desc = 'sunny', 'Partly Sunny'
        elif code in (3, 45, 48) or 71 <= code <= 77 or code in (85, 86):
            self.weather_kind, self.weather_desc = 'cloudy', 'Cloudy Skies'
        else:                                         # drizzle, rain, showers, thunderstorm
            self.weather_kind, self.weather_desc = 'rainy', 'Rainy Weather'
        self.weather_temp = f'{temp:.0f}\u00b0C'
        self.weather_tag = 'Cool' if temp < 24 else ('Mild' if temp <= 31 else 'Hot')

    def _on_key(self, window, key, *args):
        if key == 27 and self.root.current != 'garden':   # Android back button
            self.root.current = 'garden'
            return True
        return False

    def go(self, name):
        if self.root.has_screen(name):
            self.root.current = name

    def sign_out(self):
        """Sign out = close Leafora (desktop window or Android app)."""
        self.stop()

    def show_wip(self, feature):
        self.root.get_screen('wip').feature = feature
        self.root.current = 'wip'


if __name__ == '__main__':
    LeaforaApp().run()
