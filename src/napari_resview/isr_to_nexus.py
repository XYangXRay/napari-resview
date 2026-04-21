#!/usr/bin/env python3
"""
isr_to_nexus.py

Convert ISR data (SPEC file + setup YAML + TIFF directory) into a single
self-contained NeXus/HDF5 file that can be loaded by ``RSMDataLoader_Nexus``.

Usage
-----
As a module::

    from napari_resview.isr_to_nexus import convert_isr_to_nexus
    convert_isr_to_nexus(
        spec_file="setup_6oct23",
        setup_yaml="rsm3d_defaults.yaml",
        tiff_dir="data_6oct23_tiff",
        output="experiment.nxs",
    )

From the command line::

    python -m napari_resview.isr_to_nexus \\
        --spec  setup_6oct23 \\
        --yaml  rsm3d_defaults.yaml \\
        --tiff  data_6oct23_tiff \\
        -o      experiment.nxs
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import tifffile
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIFF_PATTERN = re.compile(r"^(?:[^_]+_)*(\d+)_(\d+)_.*\.tiff?$")


def _parse_spec_motors(spec_file: str | Path) -> dict:
    """
    Parse a SPEC file and return per-frame motor positions, crystal info,
    and scan metadata.

    Returns
    -------
    dict with keys:
        "scans"   : list[dict]  – one dict per scan with keys
                      scan_number, type, frames (list of dicts)
        "crystal" : dict with "g1" (list[float]) and "ub" (3×3 ndarray)
        "o0"      : list[str] – global motor names from #O0
    """
    spec_file = Path(spec_file)
    if not spec_file.is_file():
        raise FileNotFoundError(f"SPEC file not found: {spec_file}")

    o0_names: list[str] = []
    global_g1: list[float] | None = None
    global_ub: np.ndarray | None = None

    scans: list[dict] = []
    cur_scan: int | None = None
    cur_type: str = ""
    p0_map: dict[str, float] = {}
    data_idx: dict = {}
    in_data = False
    counter = 0
    current_ub: np.ndarray | None = None
    frames: list[dict] = []

    ASCAN_AXES = ("VTTH", "VTH", "Phi", "Chi")
    HKL_AXES = ("VTTH", "VTH", "Chi", "Phi")

    def _flush_scan():
        nonlocal frames
        if cur_scan is not None and frames:
            # Attach scan-level UB to every frame in this scan
            scan_ub = current_ub.copy() if current_ub is not None else None
            for fr in frames:
                fr["ub"] = scan_ub
            scans.append(
                {
                    "scan_number": cur_scan,
                    "type": cur_type,
                    "frames": list(frames),
                    "ub": scan_ub,
                }
            )
        frames = []

    with open(spec_file, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            # Global motor ordering
            if line.startswith("#O0 "):
                o0_names = line[4:].split()
                continue

            # Global crystal info
            if line.startswith("#G1 ") and cur_scan is None:
                global_g1 = [float(x) for x in line.split()[1:]]
                continue
            if line.startswith("#G3 ") and cur_scan is None:
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) == 9:
                    global_ub = np.array(vals).reshape(3, 3)
                continue

            # New scan header
            if line.startswith("#S "):
                _flush_scan()
                parts = line.split()
                cur_scan = int(parts[1])
                cur_type = parts[2] if len(parts) > 2 else ""
                p0_map.clear()
                data_idx.clear()
                in_data = False
                counter = 0
                current_ub = None
                continue

            if cur_scan is None:
                continue

            # Per-scan crystal info
            if line.startswith("#G1 "):
                continue
            if line.startswith("#G3 "):
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) == 9:
                    current_ub = np.array(vals).reshape(3, 3)
                continue

            # Fixed motor positions
            if line.startswith("#P0 "):
                vals = [float(x) for x in line.split()[1:]]
                p0_map = {
                    name: vals[i]
                    for i, name in enumerate(o0_names)
                    if i < len(vals)
                }
                continue

            # Column header
            if line.startswith("#L "):
                cols = line.split()[1:]
                ctype = cur_type.lower()
                if ctype == "ascan":
                    axes = [c for c in ASCAN_AXES if c in cols]
                    if axes:
                        scan_col = axes[0]
                        data_idx["scan_col"] = cols.index(scan_col)
                        p0_map["_scan_col_name"] = scan_col
                    for hk in ("H", "K", "L"):
                        if hk in cols:
                            data_idx[hk] = cols.index(hk)
                    in_data = True
                elif ctype == "hklscan":
                    for ax in HKL_AXES:
                        if ax in cols:
                            data_idx[ax] = cols.index(ax)
                    for hk in ("H", "K", "L"):
                        if hk in cols:
                            data_idx[hk] = cols.index(hk)
                    in_data = True
                else:
                    in_data = False
                continue

            # Data rows
            if in_data:
                if not line or (
                    line.startswith("#") and not line[1:2].isdigit()
                ):
                    in_data = False
                    continue
                parts = line.split()
                rec: dict = {
                    "scan_number": cur_scan,
                    "data_number": counter,
                }
                ctype = cur_type.lower()
                if ctype == "ascan":
                    rec["tth"] = p0_map.get("VTTH", 0.0)
                    rec["th"] = p0_map.get("VTH", 0.0)
                    rec["chi"] = p0_map.get("Chi", 0.0)
                    rec["phi"] = p0_map.get("Phi", 0.0)
                    # Override the scanned axis
                    scan_name = p0_map.get("_scan_col_name")
                    scan_col_idx = data_idx.get("scan_col")
                    if scan_col_idx is not None and scan_col_idx < len(parts):
                        val = float(parts[scan_col_idx])
                        if scan_name == "VTTH":
                            rec["tth"] = val
                        elif scan_name == "VTH":
                            rec["th"] = val
                        elif scan_name == "Phi":
                            rec["phi"] = val
                        elif scan_name == "Chi":
                            rec["chi"] = val
                    for hk in ("H", "K", "L"):
                        if hk in data_idx and data_idx[hk] < len(parts):
                            rec[hk.lower()] = float(parts[data_idx[hk]])
                elif ctype == "hklscan":
                    for ax_name, motor_key in [
                        ("VTTH", "tth"),
                        ("VTH", "th"),
                        ("Chi", "chi"),
                        ("Phi", "phi"),
                    ]:
                        if ax_name in data_idx and data_idx[ax_name] < len(
                            parts
                        ):
                            rec[motor_key] = float(parts[data_idx[ax_name]])
                    for hk in ("H", "K", "L"):
                        if hk in data_idx and data_idx[hk] < len(parts):
                            rec[hk.lower()] = float(parts[data_idx[hk]])

                frames.append(rec)
                counter += 1

    _flush_scan()

    # Global #G3 is the primary UB (matches Crystal.from_spec behaviour).
    # Only fall back to the first per-scan #G3 when no global header exists.
    if global_ub is not None:
        ub = global_ub
    else:
        ub = None
        for sc in scans:
            if sc["ub"] is not None:
                ub = sc["ub"]
                break
        if ub is None:
            ub = np.eye(3)

    crystal: dict = {"ub": ub}
    g1 = global_g1
    if g1 and len(g1) >= 6:
        crystal["a"] = g1[0]
        crystal["b"] = g1[1]
        crystal["c"] = g1[2]
        crystal["alpha"] = g1[3]
        crystal["beta"] = g1[4]
        crystal["gamma"] = g1[5]

    return {"scans": scans, "crystal": crystal, "o0": o0_names}


def _load_setup_yaml(yaml_path: str | Path, profile: str = "ISR") -> dict:
    """
    Read experiment setup from the defaults YAML.

    Returns a flat dict with distance, pitch, ycenter, xcenter,
    xpixels, ypixels, energy, wavelength.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Setup YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    # Try profile-based layout
    if isinstance(doc.get("profiles"), dict):
        p_upper = profile.upper()
        for name, pdata in doc["profiles"].items():
            if name.upper() == p_upper and isinstance(pdata, dict):
                sec = pdata.get("ExperimentSetup", pdata)
                return dict(sec)

    # Flat layout
    for key in ("ExperimentSetup", "experiment", "experiment_setup"):
        if isinstance(doc.get(key), dict):
            return dict(doc[key])

    return dict(doc)


def _load_tiff_frames(
    tiff_dir: str | Path,
    scan_frame_list: list[tuple[int, int]],
) -> list[np.ndarray]:
    """
    Load TIFF frames matching (scan_number, data_number) pairs.

    Parameters
    ----------
    tiff_dir : path
        Directory containing TIFF files.
    scan_frame_list : list of (scan_number, data_number) tuples
        Ordered list of frames to load.

    Returns
    -------
    list of 2-D ndarray
    """
    tiff_dir = Path(tiff_dir)
    if not tiff_dir.is_dir():
        raise FileNotFoundError(f"TIFF directory not found: {tiff_dir}")

    # Build index: (scan, data) -> filepath
    index: dict[tuple[int, int], Path] = {}
    for p in sorted(tiff_dir.iterdir()):
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        m = _TIFF_PATTERN.match(p.name)
        if m:
            sn, dn = int(m.group(1)), int(m.group(2))
            index[(sn, dn)] = p

    frames: list[np.ndarray] = []
    for sn, dn in scan_frame_list:
        path = index.get((sn, dn))
        if path is None:
            raise FileNotFoundError(
                f"TIFF not found for scan={sn}, data={dn} in {tiff_dir}"
            )
        frames.append(tifffile.imread(str(path)))

    return frames


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def convert_isr_to_nexus(
    spec_file: str | Path,
    setup_yaml: str | Path,
    tiff_dir: str | Path,
    output: str | Path,
    *,
    selected_scans: list[int] | None = None,
    profile: str = "ISR",
    compression: str | None = "gzip",
) -> Path:
    """
    Convert ISR data to a single NeXus/HDF5 file.

    Parameters
    ----------
    spec_file : path
        SPEC metadata file.
    setup_yaml : path
        YAML file with ExperimentSetup parameters.
    tiff_dir : path
        Directory of TIFF detector frames.
    output : path
        Destination ``.nxs`` or ``.h5`` file.
    selected_scans : list[int] | None
        Limit conversion to these scan numbers.  *None* → all.
    profile : str
        Profile name to read from the YAML (default ``"ISR"``).
    compression : str | None
        HDF5 compression for the detector data dataset.

    Returns
    -------
    Path to the written NeXus file.
    """
    output = Path(output)

    # ── 1. Parse SPEC ──────────────────────────────────────────────
    spec_data = _parse_spec_motors(spec_file)
    scans = spec_data["scans"]
    crystal = spec_data["crystal"]

    if selected_scans is not None:
        wanted = set(selected_scans)
        scans = [s for s in scans if s["scan_number"] in wanted]
        if not scans:
            raise ValueError("No scans match the requested selected_scans.")

    # Flatten per-frame records across all scans
    all_frames_meta: list[dict] = []
    for scan in scans:
        for frame in scan["frames"]:
            all_frames_meta.append(frame)

    if not all_frames_meta:
        raise ValueError("No frames found in the SPEC file.")

    n_frames = len(all_frames_meta)

    # ── 2. Load setup YAML ─────────────────────────────────────────
    setup = _load_setup_yaml(setup_yaml, profile=profile)
    distance = float(setup.get("distance", 1.0))
    pitch = float(setup.get("pitch", 75e-6))
    xcenter = int(setup.get("xcenter", 512))
    ycenter = int(setup.get("ycenter", 512))
    xpixels = int(setup.get("xpixels", 1024))
    ypixels = int(setup.get("ypixels", 1024))
    energy_keV = float(setup.get("energy", 10.0))
    wavelength_raw = setup.get("wavelength")
    if wavelength_raw is not None:
        wavelength_A = float(wavelength_raw)
    else:
        wavelength_A = 12.398419843320026 / energy_keV

    # ── 3. Load TIFF images ────────────────────────────────────────
    scan_frame_pairs = [
        (int(fm["scan_number"]), int(fm["data_number"]))
        for fm in all_frames_meta
    ]
    images = _load_tiff_frames(tiff_dir, scan_frame_pairs)

    # Verify shape consistency
    shapes = {img.shape for img in images}
    if len(shapes) > 1:
        logger.warning(
            "TIFF frames have inconsistent shapes: %s. "
            "Padding/cropping to (%d, %d).",
            shapes,
            ypixels,
            xpixels,
        )
    detector_stack = np.stack(images, axis=0)  # (N, ny, nx)

    # ── 4. Collect motor arrays ────────────────────────────────────
    omega = np.array(
        [fm.get("th", 0.0) for fm in all_frames_meta], dtype=float
    )
    chi = np.array([fm.get("chi", 0.0) for fm in all_frames_meta], dtype=float)
    phi = np.array([fm.get("phi", 0.0) for fm in all_frames_meta], dtype=float)
    tth = np.array([fm.get("tth", 0.0) for fm in all_frames_meta], dtype=float)
    scan_numbers = np.array(
        [fm["scan_number"] for fm in all_frames_meta], dtype=int
    )

    # Optional HKL
    h_arr = np.array(
        [fm.get("h", np.nan) for fm in all_frames_meta], dtype=float
    )
    k_arr = np.array(
        [fm.get("k", np.nan) for fm in all_frames_meta], dtype=float
    )
    l_arr = np.array(
        [fm.get("l", np.nan) for fm in all_frames_meta], dtype=float
    )

    # ── 5. UB matrices (per-frame from per-scan #G3 lines) ─────────
    fallback_ub = crystal.get("ub", np.eye(3))
    if not isinstance(fallback_ub, np.ndarray):
        fallback_ub = np.array(fallback_ub, dtype=float).reshape(3, 3)

    ub_stack = np.empty((n_frames, 3, 3), dtype=np.float64)
    for i, fm in enumerate(all_frames_meta):
        frame_ub = fm.get("ub")
        if frame_ub is not None:
            ub_stack[i] = frame_ub
        else:
            ub_stack[i] = fallback_ub

    # Unit-cell parameters (if available from SPEC #G1)
    unit_cell = None
    if all(k in crystal for k in ("a", "b", "c", "alpha", "beta", "gamma")):
        unit_cell = np.array(
            [
                crystal["a"],
                crystal["b"],
                crystal["c"],
                crystal["alpha"],
                crystal["beta"],
                crystal["gamma"],
            ],
            dtype=float,
        )

    # ── 6. Write NeXus/HDF5 ───────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)

    comp_kwargs: dict = {}
    if compression:
        comp_kwargs["compression"] = compression

    with h5py.File(str(output), "w") as f:
        # -- NXentry
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("definition", data="NXsas")
        entry.create_dataset("scan_number", data=scan_numbers)

        # -- NXinstrument
        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        inst.create_dataset("name", data="ISR_beamline")

        # -- NXmonochromator
        mono = inst.create_group("monochromator")
        mono.attrs["NX_class"] = "NXmonochromator"
        e_ds = mono.create_dataset("energy", data=energy_keV)
        e_ds.attrs["units"] = "keV"
        w_ds = mono.create_dataset("wavelength", data=wavelength_A)
        w_ds.attrs["units"] = "angstrom"

        # -- NXdetector
        det = inst.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"

        det.create_dataset("data", data=detector_stack, **comp_kwargs)

        d_ds = det.create_dataset("distance", data=distance)
        d_ds.attrs["units"] = "m"
        px_ds = det.create_dataset("x_pixel_size", data=pitch)
        px_ds.attrs["units"] = "m"
        py_ds = det.create_dataset("y_pixel_size", data=pitch)
        py_ds.attrs["units"] = "m"
        det.create_dataset("beam_center_x", data=float(xcenter * pitch))
        det.create_dataset("beam_center_y", data=float(ycenter * pitch))
        det.create_dataset("x_pixel_number", data=xpixels)
        det.create_dataset("y_pixel_number", data=ypixels)

        # -- Positioners (motors)
        pos = inst.create_group("positioners")
        pos.create_dataset("omega", data=omega)
        pos.create_dataset("chi", data=chi)
        pos.create_dataset("phi", data=phi)
        pos.create_dataset("tth", data=tth)

        # -- NXsample
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data="ISR_sample")
        sample.create_dataset(
            "orientation_matrix",
            data=ub_stack,
        )
        if unit_cell is not None:
            sample.create_dataset("unit_cell", data=unit_cell)

        # -- NXdata (signal link)
        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"
        data_grp.attrs["signal"] = "data"
        data_grp["data"] = det["data"]

        # -- Optional HKL
        if not np.all(np.isnan(h_arr)):
            entry.create_dataset("h", data=h_arr)
            entry.create_dataset("k", data=k_arr)
            entry.create_dataset("l", data=l_arr)

    logger.info(
        "Wrote %d frames (%d scans) to %s",
        n_frames,
        len(scans),
        output,
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert ISR data (SPEC + YAML + TIFF) to a single NeXus file.",
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the SPEC metadata file."
    )
    parser.add_argument(
        "--yaml", required=True, help="Path to the setup YAML file."
    )
    parser.add_argument(
        "--tiff", required=True, help="Path to the TIFF frame directory."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output NeXus file path (.nxs)."
    )
    parser.add_argument(
        "--scans",
        default=None,
        help="Comma-separated scan numbers to include (default: all).",
    )
    parser.add_argument(
        "--profile",
        default="ISR",
        help="Profile name in the YAML file (default: ISR).",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression of detector data.",
    )

    args = parser.parse_args(argv)

    selected = None
    if args.scans:
        selected = [int(s.strip()) for s in args.scans.split(",")]

    compression = None if args.no_compress else "gzip"

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    out = convert_isr_to_nexus(
        spec_file=args.spec,
        setup_yaml=args.yaml,
        tiff_dir=args.tiff,
        output=args.output,
        selected_scans=selected,
        profile=args.profile,
        compression=compression,
    )
    print(f"NeXus file written: {out}")


if __name__ == "__main__":
    main()
