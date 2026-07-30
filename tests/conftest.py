import os

# Qt's offscreen backend lets desktop-monitor tests run without opening windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

