"""Tests for the EKETA CAT062 parser and encoder (SP CH v1 format)."""
import math
import struct
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asterix_cue import (
    parse_eketa_block, parse_eketa_record, encode_eketa_cat062,
    AsterixCue, FT_TO_M, EKETA_FSPEC,
)

# The exact hex example from the EKETA-EAB PDF (page 7).
PDF_HEX = (
    "3E 00 31 9B 0D 01 01 02 00 01 58 78 40 00 6B 85 9C 00 43 AD 8A 00"
    " 1C 00 1C 04 D2 80 15 43 48 01 40 34 BC 01 F4 00 01 86 A0 00 00 22"
    " 0C 00 00 35 9B"
)
PDF_BYTES = bytes(int(x, 16) for x in PDF_HEX.split())


class ParsePdfExample(unittest.TestCase):
    def test_pdf_example_decodes_correctly(self):
        tracks = parse_eketa_block(PDF_BYTES)
        self.assertEqual(len(tracks), 1)
        t = tracks[0]
        self.assertEqual(t["sac"], 0)
        self.assertEqual(t["sic"], 1)
        self.assertEqual(t["track_no"], 1234)
        self.assertAlmostEqual(t["lat"], 37.80067, places=4)
        self.assertAlmostEqual(t["lon"], 23.79301, places=4)
        self.assertAlmostEqual(t["vx"], 7.0, places=1)
        self.assertAlmostEqual(t["vy"], 7.0, places=1)
        speed = math.hypot(t["vx"], t["vy"])
        self.assertAlmostEqual(speed, 9.90, delta=0.05)
        self.assertAlmostEqual(t["track_deg"], 45.0, delta=0.1)
        self.assertAlmostEqual(t["tod_s"], 45296.5, delta=0.01)
        self.assertEqual(t["status"], 0x80)
        self.assertEqual(t["key"], "TRK1234")
        self.assertEqual(t["callsign"], "TRK1234")

    def test_sp_field_decoded(self):
        t = parse_eketa_block(PDF_BYTES)[0]
        sp = t["sp"]
        self.assertEqual(sp["ver"], 1)
        self.assertEqual(sp["classif"], "unknown")
        self.assertAlmostEqual(sp["azimuth_deg"], 135.0, places=1)
        self.assertAlmostEqual(sp["elevation_deg"], 5.0, places=1)
        self.assertAlmostEqual(sp["slant_range_m"], 1000.0, places=1)
        self.assertAlmostEqual(sp["hrel_m"], 87.16, places=1)
        self.assertAlmostEqual(sp["h_msl_m"], 137.23, places=1)

    def test_altitude_from_sp_h_msl(self):
        t = parse_eketa_block(PDF_BYTES)[0]
        self.assertIsNotNone(t["alt_ft"])
        alt_m = t["alt_ft"] * FT_TO_M
        self.assertAlmostEqual(alt_m, 137.23, delta=0.1)


class RoundTrip(unittest.TestCase):
    def test_encode_then_parse(self):
        tracks_in = [dict(
            lat=37.80067, lon=23.79301, vx=7.0, vy=7.0,
            track_no=1234, h_msl_m=137.23, classif="unknown",
            azimuth_deg=135.0, elevation_deg=5.0,
            slant_range_m=1000.0, hrel_m=87.16,
        )]
        data = encode_eketa_cat062(tracks_in, sac=0, sic=1, tod_s=45296.5)
        tracks_out = parse_eketa_block(data)
        self.assertEqual(len(tracks_out), 1)
        t = tracks_out[0]
        self.assertAlmostEqual(t["lat"], 37.80067, places=4)
        self.assertAlmostEqual(t["lon"], 23.79301, places=4)
        self.assertAlmostEqual(t["vx"], 7.0, places=1)
        self.assertAlmostEqual(t["vy"], 7.0, places=1)
        self.assertEqual(t["track_no"], 1234)
        self.assertAlmostEqual(t["sp"]["h_msl_m"], 137.23, delta=0.02)
        self.assertAlmostEqual(t["sp"]["azimuth_deg"], 135.0, delta=0.02)

    def test_multiple_tracks(self):
        tracks_in = [
            dict(lat=37.80, lon=23.79, vx=5.0, vy=3.0, track_no=1,
                 h_msl_m=100.0, azimuth_deg=45.0, elevation_deg=2.0,
                 slant_range_m=500.0, hrel_m=17.4),
            dict(lat=38.00, lon=24.00, vx=-2.0, vy=8.0, track_no=2,
                 h_msl_m=200.0, azimuth_deg=90.0, elevation_deg=10.0,
                 slant_range_m=800.0, hrel_m=50.0),
        ]
        data = encode_eketa_cat062(tracks_in, tod_s=0.0)
        tracks_out = parse_eketa_block(data)
        self.assertEqual(len(tracks_out), 2)
        self.assertEqual(tracks_out[0]["track_no"], 1)
        self.assertEqual(tracks_out[1]["track_no"], 2)
        self.assertAlmostEqual(tracks_out[1]["lat"], 38.00, places=3)

    def test_negative_values(self):
        tracks_in = [dict(
            lat=-33.86, lon=151.21, vx=-10.0, vy=-5.0, track_no=99,
            h_msl_m=50.0, azimuth_deg=225.0, elevation_deg=-3.0,
            slant_range_m=300.0, hrel_m=-15.0,
        )]
        data = encode_eketa_cat062(tracks_in, tod_s=100.0)
        t = parse_eketa_block(data)[0]
        self.assertAlmostEqual(t["lat"], -33.86, places=3)
        self.assertAlmostEqual(t["lon"], 151.21, places=3)
        self.assertAlmostEqual(t["vx"], -10.0, places=1)
        self.assertAlmostEqual(t["vy"], -5.0, places=1)
        self.assertAlmostEqual(t["sp"]["elevation_deg"], -3.0, delta=0.02)
        self.assertAlmostEqual(t["sp"]["hrel_m"], -15.0, delta=0.02)


class CueIntegration(unittest.TestCase):
    def test_eketa_packet_into_cue(self):
        cue = AsterixCue(port=9999)
        tracks_in = [dict(
            lat=37.80067, lon=23.79301, vx=7.0, vy=7.0,
            track_no=42, h_msl_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
            slant_range_m=0.0, hrel_m=0.0,
        )]
        data = encode_eketa_cat062(tracks_in, tod_s=0.0)
        n = cue.ingest(data, now=1000.0)
        self.assertEqual(n, 1)
        self.assertEqual(cue.packets, 1)
        self.assertEqual(cue.records, 1)
        tgt = cue.target(1000.0, home_amsl=50.0)
        self.assertIsNotNone(tgt)
        self.assertEqual(tgt["key"], "TRK42")
        self.assertAlmostEqual(tgt["lat"], 37.80067, places=4)
        self.assertAlmostEqual(tgt["alt_amsl"], 100.0, delta=0.5)
        self.assertAlmostEqual(tgt["alt_rel"], 50.0, delta=0.5)

    def test_stale_track_expires(self):
        cue = AsterixCue(port=9999)
        data = encode_eketa_cat062(
            [dict(lat=37.0, lon=23.0, vx=0, vy=0, track_no=1,
                  h_msl_m=50.0, azimuth_deg=0, elevation_deg=0,
                  slant_range_m=0, hrel_m=0)], tod_s=0.0)
        cue.ingest(data, now=100.0)
        self.assertIsNotNone(cue.target(100.0))
        self.assertIsNone(cue.target(120.0))


class RejectNonEketa(unittest.TestCase):
    def test_non_cat062_ignored(self):
        data = struct.pack(">BH", 48, 5) + b"\x00\x00"
        tracks = parse_eketa_block(data)
        self.assertEqual(tracks, [])

    def test_wrong_fspec_falls_through(self):
        data = struct.pack(">BH", 62, 8) + b"\xFF\x00\x00\x00\x00"
        tracks = parse_eketa_block(data)
        self.assertEqual(tracks, [])

    def test_truncated_packet(self):
        tracks = parse_eketa_block(PDF_BYTES[:20])
        self.assertEqual(tracks, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
