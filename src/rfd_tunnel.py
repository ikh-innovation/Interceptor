"""Our $RFD... lines inside MAVLink TUNNEL messages, for an RFD wired to the flight controller.

  laptop RFD ~~~ drone RFD -> FC TELEM port (MAVLink2)   FC   <- MAVLink -> Jetson

ArduPilot forwards a TUNNEL with target_system 0 (broadcast) to every other
MAVLink port on which it has heard a system, so the laptop's lines reach the
Jetson and the Jetson's come back, unchanged.  A line longer than one TUNNEL
payload (128 bytes) is split; each piece starts with [message id, index, count].
"""
from __future__ import annotations

import time

PAYLOAD_TYPE = 41257          # ours: in the private range of TUNNEL payload types
PAYLOAD_MAX = 128
CHUNK = PAYLOAD_MAX - 3
LAPTOP_SYSID = 250            # never 255: the FC's GCS failsafe listens to SYSID_MYGCS (the Jetson)
LAPTOP_COMPID = 191


def fragments(data, msg_id):
    """The TUNNEL payloads (bytes, <= 128 each) that carry one line."""
    data = bytes(data)
    count = max(1, -(-len(data) // CHUNK))
    if count > 255:
        raise ValueError("line too long for the tunnel")
    return [bytes((msg_id & 0xFF, i, count)) + data[i * CHUNK:(i + 1) * CHUNK] for i in range(count)]


def encoder(out, sysid, compid):
    """A MAVLink 2 encoder writing onto out (a pymavlink connection or any .write()).
    TUNNEL exists only in MAVLink 2, whatever the main connection decoded so far."""
    from pymavlink.dialects.v20 import ardupilotmega as v20
    return v20.MAVLink(out, srcSystem=int(sysid), srcComponent=int(compid))


def padded(payload):
    return list(payload) + [0] * (PAYLOAD_MAX - len(payload))


class Reassembler:
    """Pieces in, whole lines out; an incomplete line is dropped after keep_s."""

    def __init__(self, keep_s=3.0):
        self.keep_s = keep_s
        self._parts = {}
        self.dropped = 0

    def feed(self, payload, now=None):
        now = time.monotonic() if now is None else now
        payload = bytes(payload)
        if len(payload) < 3:
            return None
        mid, idx, count = payload[0], payload[1], payload[2]
        body = payload[3:]
        if count == 0 or idx >= count:
            return None
        if count == 1:
            return body
        for k in [k for k, v in self._parts.items() if now - v[0] > self.keep_s]:
            del self._parts[k]
            self.dropped += 1
        slot = self._parts.get(mid)
        if slot is None or slot[1] != count:
            slot = self._parts[mid] = [now, count, {}]
        slot[2][idx] = body
        if len(slot[2]) == count:
            del self._parts[mid]
            return b"".join(slot[2][i] for i in range(count))
        return None
