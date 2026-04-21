"""Tests for the NeXus data loader (data_io.py — RSMDataLoader_Nexus)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from napari_resview.data_io import (  # noqa: E402
    NexusFileInspector,
    RSMDataLoader_Nexus,
    ScanScheme,
    _resolve_motor_name,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic NeXus files
# ---------------------------------------------------------------------------


def _write_minimal_nexus(path, *, n_frames=5, ny=64, nx=128, energy_keV=12.0):
    """Create a minimal NeXus/HDF5 file with detector frames and motors."""
    with h5py.File(str(path), "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        # Instrument
        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        inst.create_dataset("name", data="test_beamline")

        # Monochromator
        mono = inst.create_group("monochromator")
        mono.attrs["NX_class"] = "NXmonochromator"
        e_ds = mono.create_dataset("energy", data=energy_keV)
        e_ds.attrs["units"] = "keV"
        wl = 12.398419843320026 / energy_keV
        w_ds = mono.create_dataset("wavelength", data=wl)
        w_ds.attrs["units"] = "angstrom"

        # Detector
        det = inst.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        frames = (
            np.random.RandomState(42).rand(n_frames, ny, nx).astype(np.float32)
        )
        det.create_dataset("data", data=frames)
        dist_ds = det.create_dataset("distance", data=0.5)
        dist_ds.attrs["units"] = "m"
        px_ds = det.create_dataset("x_pixel_size", data=75e-6)
        px_ds.attrs["units"] = "m"
        py_ds = det.create_dataset("y_pixel_size", data=75e-6)
        py_ds.attrs["units"] = "m"
        det.create_dataset("beam_center_x", data=64 * 75e-6)
        det.create_dataset("beam_center_y", data=32 * 75e-6)

        # Motors (4-circle)
        pos = inst.create_group("positioners")
        pos.create_dataset("omega", data=np.linspace(10, 20, n_frames))
        pos.create_dataset("chi", data=np.full(n_frames, 90.0))
        pos.create_dataset("phi", data=np.linspace(0, 5, n_frames))
        pos.create_dataset("tth", data=np.linspace(20, 40, n_frames))

        # Sample
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data="test_crystal")
        ub = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        sample.create_dataset("orientation_matrix", data=ub)
        sample.create_dataset("unit_cell", data=[5.43, 5.43, 5.43, 90, 90, 90])

        # NXdata group (with signal link)
        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"
        data_grp.attrs["signal"] = "data"
        data_grp["data"] = det["data"]

    return path


def _write_sixcircle_nexus(path, *, n_frames=4, ny=32, nx=32):
    """Create a 6-circle NeXus file."""
    with h5py.File(str(path), "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"

        mono = inst.create_group("monochromator")
        mono.attrs["NX_class"] = "NXmonochromator"
        mono.create_dataset("energy", data=8.0)

        det = inst.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        frames = np.ones((n_frames, ny, nx), dtype=np.float32)
        det.create_dataset("data", data=frames)
        det.create_dataset("distance", data=1.0)
        det.create_dataset("x_pixel_size", data=172e-6)
        det.create_dataset("y_pixel_size", data=172e-6)

        # 6-circle motors
        pos = inst.create_group("positioners")
        pos.create_dataset("omega", data=np.linspace(5, 15, n_frames))
        pos.create_dataset("chi", data=np.full(n_frames, 0.0))
        pos.create_dataset("phi", data=np.zeros(n_frames))
        pos.create_dataset("mu", data=np.full(n_frames, 2.0))
        pos.create_dataset("delta", data=np.linspace(10, 30, n_frames))
        pos.create_dataset("nu", data=np.full(n_frames, 0.0))

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data="sixcircle_sample")

    return path


def _write_angular_scan_nexus(path, *, n_frames=10, ny=100, nx=100):
    """Create a SAXS/WAXS angular-scan NeXus file (only omega motor)."""
    with h5py.File(str(path), "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"

        mono = inst.create_group("mono")
        mono.attrs["NX_class"] = "NXmonochromator"
        e_ds = mono.create_dataset("energy", data=13500.0)
        e_ds.attrs["units"] = "eV"  # in eV, should be converted

        det = inst.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset(
            "data", data=np.zeros((n_frames, ny, nx), dtype=np.float32)
        )
        det.create_dataset("distance", data=3.0)
        px = det.create_dataset("x_pixel_size", data=0.172)
        px.attrs["units"] = "mm"  # in mm
        py = det.create_dataset("y_pixel_size", data=0.172)
        py.attrs["units"] = "mm"

        # Only one motor — angular scan
        inst.create_dataset("omega", data=np.linspace(-5, 5, n_frames))

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data="polymer_film")

    return path


@pytest.fixture
def minimal_nexus(tmp_path):
    return _write_minimal_nexus(tmp_path / "minimal.nxs")


@pytest.fixture
def sixcircle_nexus(tmp_path):
    return _write_sixcircle_nexus(tmp_path / "sixcircle.nxs")


@pytest.fixture
def angular_nexus(tmp_path):
    return _write_angular_scan_nexus(tmp_path / "angular.nxs")


# ---------------------------------------------------------------------------
# Unit tests — motor name resolution
# ---------------------------------------------------------------------------


class TestMotorAliases:
    def test_standard_names(self):
        assert _resolve_motor_name("omega") == "omega"
        assert _resolve_motor_name("chi") == "chi"
        assert _resolve_motor_name("phi") == "phi"
        assert _resolve_motor_name("tth") == "tth"

    def test_aliases(self):
        assert _resolve_motor_name("th") == "omega"
        assert _resolve_motor_name("sample_theta") == "omega"
        assert _resolve_motor_name("two_theta") == "tth"
        assert _resolve_motor_name("det_delta") == "delta"
        assert _resolve_motor_name("VTH") == "omega"

    def test_unknown(self):
        assert _resolve_motor_name("unknown_motor") is None


# ---------------------------------------------------------------------------
# Integration tests — NexusFileInspector
# ---------------------------------------------------------------------------


class TestNexusFileInspector:
    def test_list_entries(self, minimal_nexus):
        with h5py.File(str(minimal_nexus), "r") as f:
            inspector = NexusFileInspector(f)
            entries = inspector.list_entries()
            assert len(entries) == 1
            assert "/entry" in entries[0]

    def test_summarise(self, minimal_nexus):
        with h5py.File(str(minimal_nexus), "r") as f:
            inspector = NexusFileInspector(f)
            summary = inspector.summarise()
            assert len(summary) == 1
            entry_info = list(summary.values())[0]
            assert "instrument" in entry_info
            assert "motors" in entry_info
            assert "omega" in entry_info["motors"]

    def test_motor_discovery(self, sixcircle_nexus):
        with h5py.File(str(sixcircle_nexus), "r") as f:
            inspector = NexusFileInspector(f)
            summary = inspector.summarise()
            motors = list(summary.values())[0]["motors"]
            assert "omega" in motors
            assert "mu" in motors
            assert "delta" in motors
            assert "nu" in motors


# ---------------------------------------------------------------------------
# Integration tests — RSMDataLoader_Nexus (4-circle)
# ---------------------------------------------------------------------------


class TestRSMDataLoaderNexus4Circle:
    def test_load_returns_tuple(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        setup, ub, df = loader.load()
        assert setup is not None
        assert ub.shape == (3, 3)
        assert isinstance(df, pd.DataFrame)

    def test_setup_fields(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        setup, _, _ = loader.load()
        assert setup.distance == pytest.approx(0.5)
        assert setup.pitch == pytest.approx(75e-6)
        assert setup.energy_keV == pytest.approx(12.0)
        assert setup.xpixels == 128
        assert setup.ypixels == 64

    def test_ub_extracted(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        _, ub, _ = loader.load()
        np.testing.assert_array_almost_equal(ub, np.eye(3))

    def test_dataframe_columns(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        _, _, df = loader.load()
        for col in (
            "scan_number",
            "data_number",
            "intensity",
            "tth",
            "th",
            "chi",
            "phi",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_dataframe_frame_count(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        _, _, df = loader.load()
        assert len(df) == 5

    def test_intensity_shape(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        _, _, df = loader.load()
        assert df["intensity"].iloc[0].shape == (64, 128)

    def test_motor_values(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        _, _, df = loader.load()
        assert df["th"].iloc[0] == pytest.approx(10.0)
        assert df["th"].iloc[-1] == pytest.approx(20.0)
        assert df["chi"].iloc[0] == pytest.approx(90.0)

    def test_scheme_detection(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        _, _, df = loader.load()
        assert df["scan_scheme"].iloc[0] == ScanScheme.FOUR_CIRCLE.value

    def test_selected_scans(self, minimal_nexus):
        # All frames default to scan_number=1, so selecting 1 gives all
        loader = RSMDataLoader_Nexus(minimal_nexus, selected_scans=1)
        _, _, df = loader.load()
        assert len(df) == 5

    def test_selected_scans_no_match(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus, selected_scans=999)
        with pytest.raises(ValueError, match="No frames match"):
            loader.load()

    def test_inspect(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus)
        info = loader.inspect()
        assert len(info) >= 1


# ---------------------------------------------------------------------------
# Integration tests — 6-circle
# ---------------------------------------------------------------------------


class TestRSMDataLoaderNexus6Circle:
    def test_scheme_detected(self, sixcircle_nexus):
        loader = RSMDataLoader_Nexus(sixcircle_nexus)
        _, _, df = loader.load()
        assert df["scan_scheme"].iloc[0] == ScanScheme.SIX_CIRCLE.value

    def test_extra_columns(self, sixcircle_nexus):
        loader = RSMDataLoader_Nexus(sixcircle_nexus)
        _, _, df = loader.load()
        for col in ("mu", "delta", "nu"):
            assert col in df.columns

    def test_mu_values(self, sixcircle_nexus):
        loader = RSMDataLoader_Nexus(sixcircle_nexus)
        _, _, df = loader.load()
        np.testing.assert_array_almost_equal(df["mu"].values, 2.0)


# ---------------------------------------------------------------------------
# Integration tests — angular scan
# ---------------------------------------------------------------------------


class TestRSMDataLoaderNexusAngular:
    def test_scheme_detected(self, angular_nexus):
        loader = RSMDataLoader_Nexus(angular_nexus)
        _, _, df = loader.load()
        assert df["scan_scheme"].iloc[0] == ScanScheme.ANGULAR_SCAN.value

    def test_energy_ev_converted(self, angular_nexus):
        loader = RSMDataLoader_Nexus(angular_nexus)
        setup, _, _ = loader.load()
        assert setup.energy_keV == pytest.approx(13.5, rel=1e-3)

    def test_pixel_mm_converted(self, angular_nexus):
        loader = RSMDataLoader_Nexus(angular_nexus)
        setup, _, _ = loader.load()
        # 0.172 mm → 0.000172 m
        assert setup.pitch == pytest.approx(0.000172, rel=1e-3)

    def test_omega_values(self, angular_nexus):
        loader = RSMDataLoader_Nexus(angular_nexus)
        _, _, df = loader.load()
        assert df["th"].iloc[0] == pytest.approx(-5.0)
        assert df["th"].iloc[-1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_file_not_found(self):
        loader = RSMDataLoader_Nexus("/nonexistent/path.nxs")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_explicit_entry(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus, entry="/entry")
        setup, ub, df = loader.load()
        assert len(df) == 5

    def test_bad_entry_path(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus, entry="/nonexistent")
        with pytest.raises(KeyError):
            loader.load()

    def test_forced_scheme(self, minimal_nexus):
        loader = RSMDataLoader_Nexus(minimal_nexus, scan_scheme="6-circle")
        _, _, df = loader.load()
        assert df["scan_scheme"].iloc[0] == "6-circle"
        assert "mu" in df.columns

    def test_single_frame_nexus(self, tmp_path):
        path = tmp_path / "single.nxs"
        with h5py.File(str(path), "w") as f:
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            inst = entry.create_group("instrument")
            inst.attrs["NX_class"] = "NXinstrument"
            mono = inst.create_group("monochromator")
            mono.attrs["NX_class"] = "NXmonochromator"
            mono.create_dataset("energy", data=10.0)
            det = inst.create_group("detector")
            det.attrs["NX_class"] = "NXdetector"
            det.create_dataset(
                "data", data=np.ones((16, 16), dtype=np.float32)
            )
            det.create_dataset("distance", data=1.0)
            det.create_dataset("x_pixel_size", data=100e-6)
            det.create_dataset("y_pixel_size", data=100e-6)

        loader = RSMDataLoader_Nexus(path)
        setup, ub, df = loader.load()
        assert len(df) == 1
        assert df["intensity"].iloc[0].shape == (16, 16)

    def test_no_ub_returns_identity(self, tmp_path):
        path = tmp_path / "noub.nxs"
        with h5py.File(str(path), "w") as f:
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            inst = entry.create_group("instrument")
            inst.attrs["NX_class"] = "NXinstrument"
            mono = inst.create_group("monochromator")
            mono.attrs["NX_class"] = "NXmonochromator"
            mono.create_dataset("energy", data=10.0)
            det = inst.create_group("detector")
            det.attrs["NX_class"] = "NXdetector"
            det.create_dataset(
                "data", data=np.ones((2, 8, 8), dtype=np.float32)
            )
            det.create_dataset("distance", data=1.0)
            det.create_dataset("x_pixel_size", data=75e-6)
            det.create_dataset("y_pixel_size", data=75e-6)

        loader = RSMDataLoader_Nexus(path)
        _, ub, _ = loader.load()
        np.testing.assert_array_equal(ub, np.eye(3))
