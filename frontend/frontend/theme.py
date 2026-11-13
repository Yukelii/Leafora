"""Leafora theme: colors, optional fonts, optional icon font.

Drop these files in assets/fonts/ (all optional - the app runs without them):
  - MaterialIcons-Regular.ttf          -> real icons (Google "Material Icons")
  - Quicksand-Regular.ttf / -Bold.ttf  -> rounded Figma-like text
    (or Nunito-Regular.ttf / -Bold.ttf). Use the *static* files, not variable.
"""
import os

from kivy.core.text import LabelBase
from kivy.utils import get_color_from_hex as hx

# ---------- palette (taken from the Figma screens) ----------
CREAM = hx('#FAF4EC')
MINT = hx('#A8D5BA')
MINT_SOFT = hx('#B9DCC8')
MINT_L = hx('#E6F4EA')
DARK = hx('#24453A')
PINK = hx('#FDB9BE')
PINK_L = hx('#FFD9DB')
PEACH = hx('#FDE4CF')
RED = hx('#A5432F')
MUTED = hx('#5C7367')
SUN = hx('#F2B84B')
CAM_BG = hx('#244537')
CAM_BLOB = hx('#45695A')
GUIDE = hx('#8DBB9F')
PILL_GRAY = hx('#DCE5DE')

# ---------- fonts ----------
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, 'assets', 'fonts')

# ---------- your own icon images ----------
# Put PNGs in the "Icons" folder next to main.py, named after the icon, e.g.
#   Icons/leaforaplanticon.png   (header logo)
#   Icons/home.png, Icons/camera.png ...  (any name in ICONS below)
# If a PNG exists it is used; otherwise the font glyph / fallback letter is shown.
ICON_DIR = os.path.join(HERE, 'Icons')


def icon_image(name):
    for ext in ('.png', '.jpg', '.jpeg', '.webp'):
        path = os.path.join(ICON_DIR, name + ext)
        if os.path.exists(path):
            return path
    return ''


ICON_FONT = os.path.join(FONT_DIR, 'MaterialIcons-Regular.ttf')
if not os.path.exists(ICON_FONT):
    ICON_FONT = None


def setup_fonts():
    """Replace Kivy's default font (Roboto) if a rounded font is provided."""
    for family in ('Quicksand', 'Nunito'):
        reg = os.path.join(FONT_DIR, f'{family}-Regular.ttf')
        bold = os.path.join(FONT_DIR, f'{family}-Bold.ttf')
        if os.path.exists(reg) and os.path.exists(bold):
            LabelBase.register(name='Roboto', fn_regular=reg, fn_bold=bold)
            return


# name -> (Material Icons codepoint, fallback text when the font is missing)
ICONS = {
    'leaforaplanticon': ('\uea35', '*'),      # header logo (Icons/leaforaplanticon.png)
    'leaf': ('\uea35', '*'),
    'bell': ('\ue7f4', '!'),
    'person': ('\ue7fd', 'o'),
    'home': ('\ue88a', 'H'),
    'camera': ('\ue3b0', 'O'),
    'history': ('\ue889', 'L'),
    'settings': ('\ue8b8', 'S'),
    'close': ('\ue5cd', 'x'),
    'back': ('\ue5c4', '<'),
    'bulb': ('\ue0f0', 'i'),
    'check': ('\ue86c', 'v'),
    'warn': ('\ue002', '!'),
    'add': ('\ue145', '+'),
    'edit': ('\ue3c9', '/'),
    'more': ('\ue5d4', ':'),
    'share': ('\ue80d', '>'),
    'sun': ('\ue430', '*'),
    'flash': ('\ue3a1', 'A'),
    'scissors': ('\ue14e', 'x'),
    'drop': ('\ue798', 'o'),
    'sensors': ('\ue51e', '~'),
    'thermo': ('\uf076', 't'),
    'flag': ('\ue153', 'P'),
    'photo': ('\ue439', '+'),
    'palette': ('\ue40a', '#'),
    'book': ('\ue865', '='),
    'macro': ('\ue3b4', '[]'),
    'activitycategory': ('\ue8b6', '#'),
    'eye': ('\ue8f4', 'o'),
    'stickertag': ('\ue893', '#'),
    'trash': ('\ue872', 'x'),
    'warningicon': ('\ue002', '!'),
}
