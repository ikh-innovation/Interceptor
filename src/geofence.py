"""Geofence: a polygon of GPS points the WASP must stay inside.

Leaving it during a WASP flight starts SafeDown, which flies back to the GO
spot and lands (mission.py).  The polygon is drawn or typed on the phone and
kept in ~/.config/wasp/geofence.json.
"""
import math

EARTH_R = 6371000.0
MIN_POINTS = 3
MAX_POINTS = 200


def _local(origin, p):
    """(east, north) metres of p from origin: flat earth, fine for a few km."""
    lat0 = math.radians(origin[0])
    return (math.radians(p[1]-origin[1])*math.cos(lat0)*EARTH_R,
            math.radians(p[0]-origin[0])*EARTH_R)


def _segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx-ax, by-ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0]-ax)*dx + (p[1]-ay)*dy)/(dx*dx + dy*dy)))
    return math.hypot(p[0]-(ax+t*dx), p[1]-(ay+t*dy))


def parse_points(points):
    """[(lat, lon), ...] from a list of pairs, or ValueError."""
    if not isinstance(points, (list, tuple)):
        raise ValueError("geofence: a list of [lat, lon] points")
    out = []
    for p in points:
        try:
            lat, lon = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError("geofence: every point needs a latitude and a longitude")
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("geofence: a point is not a valid latitude, longitude")
        out.append((lat, lon))
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if not MIN_POINTS <= len(out) <= MAX_POINTS:
        raise ValueError(f"geofence: {MIN_POINTS} to {MAX_POINTS} points")
    o = out[0]
    xy = [_local(o, p) for p in out]
    area = 0.5*abs(sum(x0*y1 - x1*y0 for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1])))
    if area < 100.0:
        raise ValueError("geofence: the area is too small (under 100 m²)")
    return out


class Geofence:
    def __init__(self):
        self.enabled = False
        self.points = []

    @property
    def armed(self):
        """On and with a usable polygon."""
        return self.enabled and len(self.points) >= MIN_POINTS

    def load(self, saved):
        try:
            self.points = parse_points(saved.get("points", []))
        except (ValueError, AttributeError):
            self.points = []
        # A saved fence is a live fence; "enabled": false only from a script.
        self.enabled = bool(self.points) and (saved.get("enabled", True) is not False)

    def to_json(self):
        return dict(enabled=self.enabled, points=[list(p) for p in self.points])

    def check(self, pos):
        """(inside, metres to the edge) for pos (lat, lon); (None, None) when it
        can't tell: no fence or no position."""
        if not self.points or pos is None:
            return None, None
        o = self.points[0]
        xy = [_local(o, p) for p in self.points]
        q = _local(o, pos)
        inside = False
        for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1]):
            if (y0 > q[1]) != (y1 > q[1]) and q[0] < x0 + (q[1]-y0)*(x1-x0)/(y1-y0):
                inside = not inside
        edge = min(_segment_distance(q, a, b) for a, b in zip(xy, xy[1:] + xy[:1]))
        return inside, edge
