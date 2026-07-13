import cv2
import numpy as np

from .renderer import Renderer


class HudRenderer(Renderer):
    _WIN = "WASP DRONE TRACKER  [SKY + ADAPTIVE ZOOM]"

    _COLOR = {
        "SEARCH": (0, 0, 255),
        "LOCK": (0, 255, 0),
        "PREDICT": (0, 165, 255),
    }

    C_LOCK = (96, 255, 120)
    C_SEARCH = (70, 180, 255)
    C_PREDICT = (230, 200, 100)
    C_HUD = (200, 230, 205)
    C_DIM = (110, 120, 110)
    C_WARN = (70, 70, 240)
    C_INK = (22, 26, 22)
    _ACCENT = {"SEARCH": C_SEARCH, "LOCK": C_LOCK, "PREDICT": C_PREDICT}
    _HUD_FONT = cv2.FONT_HERSHEY_DUPLEX

    INSET_SRC = 100
    INSET_DISP = 180

    def __init__(self, config, commander, skip_frames):
        self.show_window = config.show_window
        self.show_sky = config.show_sky
        self.show_inset = config.show_inset
        self.commander = commander
        self.skip_frames = skip_frames

    def open(self, size):
        if not self.show_window:
            return
        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WIN, *size)

    def present(self, frame):
        if not self.show_window:
            return False
        cv2.imshow(self._WIN, frame)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self):
        if self.show_window:
            cv2.destroyAllWindows()

    def draw(self, ctx, tracker, fused, roi, cmd, sent):
        frame = ctx.frame_disp
        h, w = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2
        accent = self._ACCENT.get(tracker.state, self.C_HUD)

        if self.show_sky:
            self._draw_sky(frame, ctx.sky_mask, accent)

        self._draw_frame_ticks(frame, self.C_DIM)
        self._reticle(frame, cx_f, cy_f, self.C_HUD)

        if roi and tracker.state == "PREDICT":
            self._corner_box(frame, roi[0], roi[1], roi[2], roi[3], self.C_DIM, 10, 1)

        if fused:
            px, py = int(fused[0]), int(fused[1])
            self._dashed(frame, (cx_f, cy_f), (px, py), accent, thick=1)

            if tracker.state == "PREDICT":
                sx, sy = tracker.kalman.pos_sigma()
                cv2.ellipse(frame, (px, py),
                            (max(12, int(sx * 3)), max(12, int(sy * 3))),
                            0, 0, 360, self.C_PREDICT, 1, cv2.LINE_AA)

            bw, bh = (tracker.last_bbox[2], tracker.last_bbox[3]) if tracker.last_bbox else (20, 20)
            half = max(16, max(bw, bh) // 2 + 8)
            self._corner_box(frame, px - half, py - half, px + half, py + half,
                             accent, length=min(16, half // 2), thick=2)
            cv2.circle(frame, (px, py), 2, accent, -1, cv2.LINE_AA)

            vx, vy = tracker.kalman.velocity()
            if np.hypot(vx, vy) > 0.5:
                tip = (int(px + vx * 8), int(py + vy * 8))
                cv2.arrowedLine(frame, (px, py), tip, self.C_HUD, 1,
                                tipLength=0.3, line_type=cv2.LINE_AA)

        self._draw_header(frame, tracker, accent)
        self._draw_flightcmd(frame, tracker, cmd, fused, cx_f, cy_f, sent, accent)

        if self.show_inset:
            self._draw_inset(frame, ctx, tracker, fused)

    def _draw_sky(self, frame, mask, color):
        if mask is None:
            return
        tint = frame.copy()
        tint[mask > 0] = (120, 90, 40)
        cv2.addWeighted(tint, 0.12, frame, 0.88, 0, frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            self._dashed_poly(frame, c, color, dash=8, gap=7, thick=1)

    def _draw_inset(self, frame, ctx, tracker, pos):
        if pos is None or ctx.frame_full is None:
            return
        src = self.INSET_SRC
        ncx = int(pos[0] * ctx.full_w / ctx.disp_w)
        ncy = int(pos[1] * ctx.full_h / ctx.disp_h)
        x1 = min(max(0, ncx - src // 2), max(0, ctx.full_w - src))
        y1 = min(max(0, ncy - src // 2), max(0, ctx.full_h - src))
        crop = ctx.frame_full[y1:y1 + src, x1:x1 + src]
        if crop.size == 0:
            return

        z = cv2.resize(crop, (self.INSET_DISP, self.INSET_DISP),
                       interpolation=cv2.INTER_LINEAR)
        color = self._COLOR.get(tracker.state, (255, 255, 255))
        rel_x = int((ncx - x1) * self.INSET_DISP / src)
        rel_y = int((ncy - y1) * self.INSET_DISP / src)
        cv2.drawMarker(z, (rel_x, rel_y), color, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        cv2.rectangle(z, (0, 0), (self.INSET_DISP - 1, self.INSET_DISP - 1), color, 2)
        cv2.putText(z, "ZOOM", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        h, w = frame.shape[:2]
        m = 10
        y0 = h - self.INSET_DISP - m
        x0 = w - self.INSET_DISP - m
        if y0 >= 0 and x0 >= 0:
            frame[y0:y0 + self.INSET_DISP, x0:x0 + self.INSET_DISP] = z

    @staticmethod
    def _panel(frame, p1, p2, fill, alpha, border=None):
        x1, y1 = p1; x2, y2 = p2
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        sub = frame[y1:y2, x1:x2]
        cv2.addWeighted(np.full_like(sub, fill), alpha, sub, 1 - alpha, 0, sub)
        if border is not None:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border, 1, cv2.LINE_AA)

    def _text(self, frame, txt, org, scale, color, thick=1, font=None):
        font = font or self._HUD_FONT
        cv2.putText(frame, txt, (org[0] + 1, org[1] + 1), font, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, org, font, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _corner_box(frame, x1, y1, x2, y2, color, length=16, thick=2):
        L = max(5, int(min(length, (x2 - x1) / 2, (y2 - y1) / 2)))
        for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, thick, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, thick, cv2.LINE_AA)

    def _reticle(self, frame, cx, cy, color):
        gap, arm = 9, 16
        cv2.line(frame, (cx - gap - arm, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + gap + arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - gap - arm), (cx, cy - gap), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + gap + arm), color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

    @staticmethod
    def _dashed(frame, p1, p2, color, dash=9, gap=7, thick=1):
        p1 = np.array(p1, float); p2 = np.array(p2, float)
        d = p2 - p1; L = float(np.hypot(*d))
        if L < 1:
            return
        u = d / L
        n = int(L // (dash + gap)) + 1
        for i in range(n):
            a = p1 + u * (i * (dash + gap))
            b = p1 + u * min(i * (dash + gap) + dash, L)
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)),
                     color, thick, cv2.LINE_AA)

    @staticmethod
    def _dashed_poly(frame, pts, color, dash=8, gap=7, thick=1):
        pts = pts.reshape(-1, 2).astype(float)
        if len(pts) < 2:
            return
        period = dash + gap
        carry = 0.0
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            d = b - a
            seg = float(np.hypot(*d))
            if seg < 1e-6:
                continue
            u = d / seg
            pos = 0.0
            while pos < seg:
                phase = carry % period
                remain = (dash - phase) if phase < dash else (period - phase)
                step = min(remain, seg - pos)
                if phase < dash:
                    p0 = a + u * pos
                    p1 = a + u * (pos + step)
                    cv2.line(frame, tuple(p0.astype(int)), tuple(p1.astype(int)),
                             color, thick, cv2.LINE_AA)
                pos += step
                carry += step

    def _draw_frame_ticks(self, frame, color):
        h, w = frame.shape[:2]
        m, L = 6, 22
        for (cx, cy, sx, sy) in ((m, m, 1, 1), (w - m, m, -1, 1),
                                 (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, 1, cv2.LINE_AA)

    def _draw_header(self, frame, tracker, accent):
        w = frame.shape[1]
        self._panel(frame, (0, 0), (w, 30), self.C_INK, 0.45)
        cv2.line(frame, (0, 30), (w, 30), accent, 1, cv2.LINE_AA)

        title = "WASP  //  SKY-TRACK"
        self._text(frame, title, (12, 21), 0.5, self.C_HUD, 1)

        (tw, _), _ = cv2.getTextSize(title, self._HUD_FONT, 0.5, 1)
        self._text(frame, "IKH - DEFENSE LABS", (12 + tw + 20, 21),
                   0.5, self.C_DIM, 1)

        (sw, _), _ = cv2.getTextSize(tracker.state, self._HUD_FONT, 0.5, 1)
        x0 = w - sw - 30
        cv2.circle(frame, (x0 - 2, 15), 5, accent, -1, cv2.LINE_AA)
        self._text(frame, tracker.state, (x0 + 10, 21), 0.5, accent, 1)

        if tracker.conf is not None and tracker.state == "LOCK":
            ctxt = f"CONF {tracker.conf * 100:.0f}%"
            (cw, _), _ = cv2.getTextSize(ctxt, self._HUD_FONT, 0.5, 1)
            self._text(frame, ctxt, (x0 - cw - 18, 21), 0.5, accent, 1)

    def _hbar(self, frame, x, y, w, h, frac, color, bidirectional=True):
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.C_DIM, 1, cv2.LINE_AA)
        frac = float(np.clip(frac, -1.0, 1.0))
        if bidirectional:
            cx = x + w // 2
            cv2.line(frame, (cx, y - 2), (cx, y + h + 2), self.C_DIM, 1, cv2.LINE_AA)
            fill = int(frac * (w // 2 - 2))
            if fill > 0:
                cv2.rectangle(frame, (cx + 1, y + 2), (cx + fill, y + h - 2),
                              color, -1, cv2.LINE_AA)
            elif fill < 0:
                cv2.rectangle(frame, (cx + fill, y + 2), (cx - 1, y + h - 2),
                              color, -1, cv2.LINE_AA)
        else:
            fill = int(max(0.0, frac) * (w - 4))
            if fill > 0:
                cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill, y + h - 2),
                              color, -1, cv2.LINE_AA)

    def _draw_flightcmd(self, frame, tracker, cmd, pos, cx_f, cy_f, sent, accent):
        h, w = frame.shape[:2]
        pw, ph = 246, 118
        x1 = 10
        y1 = h - 10 - ph
        x2, y2 = x1 + pw, y1 + ph
        self._panel(frame, (x1, y1), (x2, y2), self.C_INK, 0.5, border=self.C_DIM)

        tx = x1 + 12
        self._text(frame, "FLIGHT CMD", (tx, y1 + 18), 0.42, self.C_DIM, 1)

        if not sent:
            lamp, tag = self.C_DIM, "HOLD"
        elif self.commander.master is None:
            lamp, tag = self.C_SEARCH, "SIM"
        else:
            lamp, tag = self.C_LOCK, "TX"
        (tw, _), _ = cv2.getTextSize(tag, self._HUD_FONT, 0.42, 1)
        cv2.circle(frame, (x2 - tw - 24, y1 + 13), 4, lamp, -1, cv2.LINE_AA)
        self._text(frame, tag, (x2 - tw - 14, y1 + 18), 0.42, lamp, 1)

        if pos:
            ex = pos[0] - cx_f; ey = pos[1] - cy_f
            az = np.degrees(ex * (self.commander.H_FOV / w))
            el = np.degrees(-ey * (self.commander.V_FOV / h))
            dev = f"DEV {ex:+04d},{ey:+04d}px  AZ{az:+05.1f} EL{el:+05.1f}"
        else:
            dev = "DEV ----,----px  AZ----- EL-----"
        self._text(frame, dev, (tx, y1 + 36), 0.36, self.C_HUD, 1)

        bar_x, bar_w, bar_h = tx + 42, 110, 8
        rows = (
            ("YAW", cmd["yaw_rate"] / self.commander.MAX_YAW, f"{cmd['yaw_rate']:+.2f}r/s", True),
            ("CLB", -cmd["vz"] / self.commander.MAX_VZ, f"{-cmd['vz'] + 0.0:+.2f}m/s", True),
            ("FWD", cmd["vx"] / self.commander.MAX_FWD, f"{cmd['vx']:.2f}m/s", False),
        )
        for i, (label, frac, val, bidir) in enumerate(rows):
            y = y1 + 48 + i * 20
            self._text(frame, label, (tx, y + bar_h), 0.38, self.C_HUD, 1)
            self._hbar(frame, bar_x, y, bar_w, bar_h, frac, accent, bidir)
            self._text(frame, val, (bar_x + bar_w + 8, y + bar_h), 0.38, self.C_HUD, 1)

        rate = f"TX RATE = DET RATE  (skip {self.skip_frames})"
        self._text(frame, rate, (tx, y1 + ph - 8), 0.34, self.C_DIM, 1)
