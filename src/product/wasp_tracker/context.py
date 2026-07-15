from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np

Detection = namedtuple("Detection", ["cx", "cy", "x1", "y1", "w", "h", "conf"])


@dataclass
class FrameContext:
    frame_full: np.ndarray
    frame_disp: np.ndarray
    full_w: int
    full_h: int
    disp_w: int
    disp_h: int
    sky_mask: Optional[np.ndarray]
