"""MAVLink TCP proxy between the app and SITL for SafeDown fault injection.

    deny_modes  DO_SET_MODE for these mode numbers never reaches the FC; the
                proxy answers COMMAND_ACK DENIED as the FC itself would.
    frozen      everything from the app is dropped, as if the Jetson hung:
                its RC overrides lapse after RC_OVERRIDE_TIME.
"""
import socket
import threading

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav2


class MavProxy:
    def __init__(self, upstream_port, listen_port=0):
        self.upstream_port = upstream_port
        self.deny_modes = set()
        self.frozen = False
        self.denied = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", listen_port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def url(self):
        return f"tcp:127.0.0.1:{self.port}"

    def _accept(self):
        self._srv.settimeout(.5)
        while not self._stop.is_set():
            try:
                app, _ = self._srv.accept()
            except OSError:
                continue
            fc = socket.create_connection(("127.0.0.1", self.upstream_port))
            self._app, self._fc = app, fc
            # The FC side keeps its own sequence numbers for injected ACKs.
            self._enc = mav2.MAVLink(None, srcSystem=1, srcComponent=1)
            threading.Thread(target=self._pump, args=(fc, app, False), daemon=True).start()
            threading.Thread(target=self._pump, args=(app, fc, True), daemon=True).start()

    def _pump(self, src, dst, from_app):
        parser = mav2.MAVLink(None)
        parser.robust_parsing = True
        while not self._stop.is_set():
            try:
                data = src.recv(4096)
            except OSError:
                return
            if not data:
                return
            if not from_app:
                with self._lock:
                    self._send(dst, data)
                continue
            if self.frozen:
                continue
            out = bytearray()
            for byte in data:
                msg = parser.parse_char(bytes([byte]))
                if msg is None:
                    continue
                raw = msg.get_msgbuf()
                if (msg.get_type() == "COMMAND_LONG"
                        and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE
                        and int(msg.param2) in self.deny_modes):
                    self.denied.append(int(msg.param2))
                    ack = self._enc.command_ack_encode(msg.command, mavutil.mavlink.MAV_RESULT_DENIED)
                    with self._lock:
                        self._send(src, ack.pack(self._enc))
                    continue
                out += raw
            if out:
                self._send(dst, bytes(out))

    @staticmethod
    def _send(sock, data):
        try:
            sock.sendall(data)
        except OSError:
            pass

    def close(self):
        self._stop.set()
        for s in (self._srv, getattr(self, "_app", None), getattr(self, "_fc", None)):
            try:
                s and s.close()
            except OSError:
                pass
