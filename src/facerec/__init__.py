"""Real-time face recognition and attendance."""

import cv2

# OpenCV 5's new DNN engine logs a harmless "Targets are not supported" warning per model load.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:  # very old OpenCV builds
    pass

__version__ = "2.0.0"
