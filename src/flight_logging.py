import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path


class FlightLogs:
    ACTIVE_PHASES = frozenset(("AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN",
                              "PUSH", "AI", "STOP", "WASP_FLIGHT", "IDLE"))
    COLUMNS = ("utc_time", "flight_mode", "flight_id", "flight_elapsed_s",
               "idle_limit_s", "idle_remaining_s", "idle_stage",
               "idle_height_target_m", "idle_heading_target_deg",
               "detection_fresh", "wasp_stage", "wasp_steering_blend",
               "wasp_target_acquired", "cmd_roll_rate_dps",
               "cmd_pitch_rate_dps", "cmd_yaw_rate_dps", "cam_fps", "processing_fps",
               "wasp_motion_assist", "target_estimate_source", "target_motion_confidence",
               "target_flow_quality", "target_noise_rad", "target_outliers_rejected")

    def __init__(self, directory, stamp, header):
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

    def _start(self, now, state, metadata):
        kind = ("wasp" if state["wasp_enabled"] else
                "auto" if state["auto_run"] or state["phase"].startswith("AUTO_")
                else "semi_auto")
        while True:
            self.counter += 1
            name = f"flight_{self.stamp}_{self.counter:03d}_{kind}"
            path = self.directory / f"{name}.csv"
            if not any(self.directory.glob(name + ".*")):
                break
        self._csv = path.open("x", buffering=1)
        self._csv.write(self.header + "\n")
        self._events = path.with_suffix(".events.jsonl").open("x", buffering=1)
        self.active = dict(flight_id=name, flight_mode=kind, started=now,
                           csv=str(path), events=str(path.with_suffix(".events.jsonl")),
                           metadata=str(path.with_suffix(".json")))
        self._previous = {}
        self._metadata = dict(schema_version=1, flight_id=name, flight_mode=kind,
                              status="recording", started_utc=self._utc(),
                              start_t_s=state["t_s"], rows=0,
                              settings=metadata())
        self._save_metadata()
        self._event(now, "start", phase=state["phase"])


    @staticmethod
    def _utc():
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
        values = dict(state, utc_time=self._utc(),
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
        except Exception as exc:
            self.error = str(exc)

        finally:
            self._close_handles()
            self.last, self.active = self.active, None

    def snapshot(self):
        flight = self.active or self.last
        return dict(active=self.active is not None and self.error is None,
                    flight_mode=None if flight is None else flight["flight_mode"],
                    csv=None if flight is None else flight["csv"],
                    events=None if flight is None else flight["events"],
                    metadata=None if flight is None else flight["metadata"], error=self.error)
