import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path


class FlightLogs:
    ACTIVE_PHASES = frozenset(("AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN",
                              "PUSH", "AI", "STOP", "WASP_FLIGHT", "IDLE",
                              "SAFEDOWN"))
    COLUMNS = ("utc_time", "time_ref", "flight_mode", "flight_id",
               "flight_elapsed_s",
               "idle_limit_s", "idle_remaining_s", "idle_stage",
               "idle_height_target_m", "idle_heading_target_deg",
               "detection_fresh", "wasp_stage", "wasp_steering_blend",
               "wasp_target_acquired", "cmd_roll_rate_dps",
               "cmd_pitch_rate_dps", "cmd_yaw_rate_dps", "cam_fps", "processing_fps")

    def __init__(self, directory, stamp, header, utc_fn=None, uploader=None):
        self.directory = Path(directory)
        self.stamp = stamp
        self.header = header + "," + ",".join(self.COLUMNS)
        self.counter = 0
        self.active = None
        self.last = None
        self.error = None
        self._csv = self._events = None
        self._previous = {}
        self._last_now = None
        self._utc_fn = utc_fn
        self._uploader = uploader

    def _start(self, now, state, metadata):
        kind = ("wasp" if state["wasp_enabled"] else
                "auto" if state["auto_run"] or state["phase"].startswith("AUTO_")
                else "semi_auto")
        flights_dir = self.directory / "flights"
        flights_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self.counter += 1
            name = f"flight_{self.stamp}_{self.counter:03d}_{kind}"
            flight_dir = flights_dir / name
            if not flight_dir.exists():
                break
        flight_dir.mkdir(parents=True, exist_ok=True)
        path = flight_dir / f"{name}.csv"
        self._csv = path.open("x", buffering=1)
        self._csv.write(self.header + "\n")
        self._events = path.with_suffix(".events.jsonl").open("x", buffering=1)
        self.active = dict(flight_id=name, flight_mode=kind, started=now,
                           flight_dir=str(flight_dir),
                           csv=str(path), events=str(path.with_suffix(".events.jsonl")),
                           metadata=str(path.with_suffix(".json")))
        self._previous = {}
        self._metadata = dict(schema_version=1, flight_id=name, flight_mode=kind,
                              status="recording", started_utc=self._utc(),
                              start_t_s=state["t_s"], rows=0,
                              settings=metadata())
        self._save_metadata()
        if self._uploader is not None:
            self._uploader.push_metadata(name, self._metadata)
        self._event(now, "start", phase=state["phase"])


    def _utc(self):
        if self._utc_fn is not None:
            utc_str, ref = self._utc_fn()
            self._last_time_ref = ref
            return utc_str
        self._last_time_ref = "SYS_UTC"
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _save_metadata(self):
        path = Path(self.active["metadata"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def _event(self, now, kind, **data):
        event = dict(utc_time=self._utc(), event=kind,
                     flight_id=self.active["flight_id"],
                     flight_mode=self.active["flight_mode"],
                     elapsed_s=round(max(0.0, now-self.active["started"]), 3), **data)
        self._events.write(json.dumps(event, allow_nan=False) + "\n")
        if self._uploader is not None:
            self._uploader.push_event(self.active["flight_id"], event)

    def _changes(self, now, state):
        for field in ("phase", "detection_fresh", "wasp_stage", "idle_stage"):
            value = state.get(field)
            if value != self._previous.get(field):
                self._event(now, field, previous=self._previous.get(field), value=value)
                self._previous[field] = value
        event_key = (state.get("mission_event_t"), state.get("mission_event"))
        if state.get("mission_event") and event_key != self._previous.get("mission_event"):
            self._event(now, "mission", message=state["mission_event"])
            self._previous["mission_event"] = event_key

    def record(self, now, row, state, metadata):
        self._last_now = now
        running = (state["phase"] in self.ACTIVE_PHASES
                   and state["armed"] and state["linked"])
        if running and self.active is None and self.error is None:
            try:
                self._start(now, state, metadata)
            except Exception as exc:
                self._fail(exc)
        elapsed = "" if self.active is None else round(max(0.0, now-self.active["started"]), 3)
        utc_now = self._utc()
        values = dict(state, utc_time=utc_now,
                      time_ref=getattr(self, "_last_time_ref", "SYS_UTC"),
                      flight_mode="standby" if self.active is None else self.active["flight_mode"],
                      flight_id="" if self.active is None else self.active["flight_id"],
                      flight_elapsed_s=elapsed,
                      idle_remaining_s="" if elapsed == "" else round(
                          max(0.0, state["idle_limit_s"]-elapsed), 3))
        suffix = io.StringIO()
        csv.writer(suffix, lineterminator="").writerow(
            values.get(key, "") if values.get(key) is not None else "" for key in self.COLUMNS)
        row += "," + suffix.getvalue()
        if self.active is not None and self.error is None:
            try:
                self._changes(now, state)
                self._csv.write(row + "\n")
                self._metadata["rows"] += 1
                if self._uploader is not None and self._metadata["rows"] % 50 == 0:
                    self._uploader.push_row(self.active["flight_id"], values)
                if not running:
                    reason = ("disarmed" if not state["armed"] else
                              "FC link lost" if not state["linked"] else
                              state.get("gate_reason") or state["phase"])
                    self.close(now, reason, state["phase"])
            except Exception as exc:
                self._fail(exc)
        elif self.active is not None and not running:
            self.close(now, "logging error", state["phase"])
        return row

    def _fail(self, exc):
        self.error = str(exc)

        self._close_handles()

    def _close_handles(self):
        for name in ("_csv", "_events"):
            handle = getattr(self, name)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, name, None)

    def close(self, now=None, reason="service stopped", phase=None):
        if self.active is None:
            return
        now = self._last_now if now is None else now
        try:
            if self._events is not None:
                self._event(now, "end", reason=reason, phase=phase)
            self._metadata.update(status="complete" if self.error is None else "incomplete",
                                  logging_error=self.error, ended_utc=self._utc(),
                                  duration_s=round(max(0.0, now-self.active["started"]), 3),
                                  end_reason=reason, end_phase=phase)
            self._save_metadata()
            if self._uploader is not None:
                self._uploader.push_metadata(self.active["flight_id"], self._metadata)
        except Exception as exc:
            self.error = str(exc)

        finally:
            self._close_handles()
            self.last, self.active = self.active, None

    def snapshot(self):
        flight = self.active or self.last
        return dict(active=self.active is not None and self.error is None,
                    flight_mode=None if flight is None else flight["flight_mode"],
                    flight_dir=None if flight is None else flight.get("flight_dir"),
                    csv=None if flight is None else flight["csv"],
                    events=None if flight is None else flight["events"],
                    metadata=None if flight is None else flight["metadata"], error=self.error)

    @staticmethod
    def list_flights(data_dir):
        flights_dir = Path(data_dir) / "flights"
        if not flights_dir.is_dir():
            return []
        result = []
        for d in sorted(flights_dir.iterdir(), reverse=True):
            if not d.is_dir() or not d.name.startswith("flight_"):
                continue
            meta_path = d / f"{d.name}.json"
            info = dict(flight_id=d.name, flight_dir=str(d))
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    info.update(flight_mode=meta.get("flight_mode"),
                                status=meta.get("status"),
                                started_utc=meta.get("started_utc"),
                                ended_utc=meta.get("ended_utc"),
                                duration_s=meta.get("duration_s"),
                                rows=meta.get("rows"),
                                end_reason=meta.get("end_reason"))
                except Exception:
                    info["status"] = "metadata_error"
            raw_files = [f.name for f in d.iterdir() if f.is_file()]
            info["raw_files"] = raw_files
            info["has_converted"] = (d / "ELKAK").is_dir()
            result.append(info)
        return result


class BatteryLog:
    """L-BAT: session-wide battery CSV at >= 1 Hz.

    Columns: t_s, utc_time, voltage_v, current_a, battery_pct, consumed_wh
    The file is opened once per session (same stamp as the control CSV)
    and closed when the commander shuts down.
    """
    HEADER = "t_s,utc_time,time_ref,voltage_v,current_a,battery_pct,consumed_wh\n"
    MIN_INTERVAL_S = 1.0          # write at most once per second

    def __init__(self, directory, stamp, utc_fn=None):
        self.path = str(Path(directory) / f"battery_{stamp}.csv")
        self._fh = open(self.path, "w", buffering=1)
        self._fh.write(self.HEADER)
        self._utc_fn = utc_fn
        self._prev_t = None
        self._next_write = None
        self._wh = 0.0
        self.error = None
        self.rows = 0

    def record(self, t_s, voltage_v, current_a, battery_pct=None):
        """Call every cycle (~50 Hz).  Integrates energy continuously,
        writes a row at >= 1 Hz.  ``voltage_v`` or ``current_a`` may be
        None when the FC hasn't reported yet; the row is skipped."""
        if self._fh is None or self.error is not None:
            return
        if voltage_v is None or current_a is None:
            return
        try:
            dt = 0.0 if self._prev_t is None else max(0.0, t_s - self._prev_t)
            self._prev_t = t_s
            self._wh += voltage_v * current_a * dt / 3600.0
            if self._next_write is None or t_s >= self._next_write:
                self._next_write = (t_s + self.MIN_INTERVAL_S
                                    if self._next_write is None
                                    else self._next_write + self.MIN_INTERVAL_S)
                if self._utc_fn is not None:
                    utc, ref = self._utc_fn()
                else:
                    utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                    ref = "SYS_UTC"
                pct = "" if battery_pct is None else f"{battery_pct}"
                self._fh.write(f"{t_s:.3f},{utc},{ref},{voltage_v:.2f},"
                               f"{current_a:.1f},{pct},{self._wh:.3f}\n")
                self.rows += 1
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def snapshot(self):
        return dict(csv=self.path, rows=self.rows, consumed_wh=round(self._wh, 3),
                    error=self.error)
