from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from typing import Tuple, Union

import numpy as np

try:
    import napari  # type: ignore
except Exception as e:
    raise ImportError(
        "napari must be installed to use RSMNapariViewer. `pip install napari`"
    ) from e


__all__ = ("RSMNapariViewer", "IntensityNapariViewer")


class RSMNapariViewer:
    """
    Napari viewer for 3D reciprocal-space maps (HKL or Q). Always shows scale bar in Å⁻¹.
    Layers (top→bottom) after reordering:
        RSM3D, Outline box, Outline corners, Corner Labels, World axes
    """

    def __init__(
        self,
        grid: np.ndarray,
        axes: Tuple[Iterable[float], Iterable[float], Iterable[float]],
        *,
        space: str = "hkl",
        name: str = "RSM",
        log_view: bool = True,
        contrast_percentiles: Tuple[float, float] = (1.0, 99.8),
        cmap: str = "viridis",
        rendering: str = "attenuated_mip",
        viewer_kwargs: dict | None = None,
    ) -> None:
        self._validate_grid_axes(grid, axes)
        self.grid, (self.xax, self.yax, self.zax) = self._ensure_ascending(
            grid, axes
        )
        self.space = space.lower()
        if self.space not in {"hkl", "q"}:
            raise ValueError("space must be 'hkl' or 'q'")
        self.name = name
        self.log_view = bool(log_view)
        self.contrast_percentiles = contrast_percentiles
        self.cmap = cmap
        self.rendering = rendering
        self.viewer_kwargs = viewer_kwargs or {}
        self.volume, self.scale, self.translate, self.is_uniform = (
            self._volume_from_grid_axes(
                self.grid, (self.xax, self.yax, self.zax)
            )
        )
        self.viewer: napari.Viewer | None = None
        self.img_layer = None
        self._hud_enabled = True
        self._corner_labels_layer = None  # restored (even if unused)

    # ------------------------------ public ------------------------------------
    def launch(self, viewer: napari.Viewer | None = None) -> napari.Viewer:
        """Launch into a new Napari viewer or reuse the provided `viewer`.

        If `viewer` is None a new `napari.Viewer` is created; otherwise layers
        are added to the supplied viewer.
        """
        v = viewer or napari.Viewer(
            title=f"{self.name} viewer", **self.viewer_kwargs
        )
        # If reusing an existing viewer, clear its current layers first so the
        # new visualization replaces the previous content.
        if viewer is not None:
            with contextlib.suppress(Exception):
                for _ly in list(v.layers):
                    with contextlib.suppress(Exception):
                        v.layers.remove(_ly)
        v.dims.ndisplay = 3
        data = self._log1p_clip(self.volume) if self.log_view else self.volume
        lo, hi = self._robust_percentiles(data, self.contrast_percentiles)

        layer = v.add_image(
            data,
            name="RSM3D",
            colormap=self.cmap,
            scale=self.scale,
            translate=self.translate,
            contrast_limits=(float(lo), float(hi)),
        )
        self._force_volume(layer)

        # REMOVE old simple camera block and RESTORE robust auto zoom
        self._auto_zoom(v, layer)

        v.axes.visible = True
        v.axes.colored = True
        v.axes.arrows = True
        v.scale_bar.visible = True
        v.scale_bar.unit = "Å⁻¹"
        v.dims.axis_labels = (
            ("Qz", "Qy", "Qx") if self.space == "q" else ("L", "K", "H")
        )

        self._add_outline_and_corners(v)
        self._add_axes_vectors(v)
        self._install_hud(v, layer)

        # Desired panel order top→bottom
        with contextlib.suppress(Exception):
            # original intended order
            self._apply_display_order(
                v,
                [
                    "RSM3D",
                    "Outline box",
                    "Outline corners",
                    "Corner Labels",
                    "World axes",
                ],
            )

        self.viewer = v
        self.img_layer = layer
        return v

    def add_grid_overlay(
        self,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        *,
        thickness_vox: int = 1,
        opacity: float = 0.25,
        color: str = "white",
        name: str = "grid-planes",
        max_planes_per_axis: int = 200,
    ) -> None:
        self._require_viewer()
        nx, ny, nz = self.grid.shape
        xax, yax, zax = self.xax, self.yax, self.zax

        def nearest_indices(ax: np.ndarray, step: float) -> np.ndarray:
            if step <= 0 or ax.size < 2:
                return np.array([], dtype=int)
            start = np.ceil(ax[0] / step) * step
            coords = np.arange(start, ax[-1] + 0.5 * step, step)
            idx = np.searchsorted(ax, coords)
            idx = idx[(idx >= 0) & (idx < ax.size)]
            return np.unique(idx)

        ix = nearest_indices(xax, spacing[0])
        iy = nearest_indices(yax, spacing[1])
        iz = nearest_indices(zax, spacing[2])
        if any(sz > max_planes_per_axis for sz in (ix.size, iy.size, iz.size)):
            raise ValueError(
                "Too many grid planes — adjust spacing or max_planes_per_axis."
            )

        mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        half = max(int(thickness_vox) // 2, 0)
        for k in ix:
            mask[max(0, k - half) : min(nx, k + half + 1), :, :] = 1
        for k in iy:
            mask[:, max(0, k - half) : min(ny, k + half + 1), :] = 1
        for k in iz:
            mask[:, :, max(0, k - half) : min(nz, k + half + 1)] = 1

        mask_vol = np.ascontiguousarray(mask.transpose(2, 1, 0))
        layer = self.viewer.add_image(
            mask_vol,
            name=name,
            opacity=float(opacity),
            blending="additive",
            colormap=color,
            scale=self.scale,
            translate=self.translate,
            contrast_limits=(0, 1),
            rendering="translucent",
        )
        with contextlib.suppress(Exception):
            layer.interpolation = "nearest"

    def flip_axis(self, axis: str) -> None:
        self._require_viewer()
        axis = axis.lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if axis == "x":
            self.grid = self.grid[::-1, :, :]
            self.xax = self.xax[::-1].copy()
        elif axis == "y":
            self.grid = self.grid[:, ::-1, :]
            self.yax = self.yax[::-1].copy()
        else:
            self.grid = self.grid[:, :, ::-1]
            self.zax = self.zax[::-1].copy()
        self.volume, self.scale, self.translate, self.is_uniform = (
            self._volume_from_grid_axes(
                self.grid, (self.xax, self.yax, self.zax)
            )
        )
        data = self._log1p_clip(self.volume) if self.log_view else self.volume
        if self.img_layer is not None:
            self.img_layer.data = data
            self.img_layer.scale = self.scale
            self.img_layer.translate = self.translate
            self._force_volume(self.img_layer)

    # ------------------------------ overlays ----------------------------------
    def _add_outline_and_corners(self, v: napari.Viewer) -> None:
        zmin, zmax = float(self.zax[0]), float(self.zax[-1])
        ymin, ymax = float(self.yax[0]), float(self.yax[-1])
        xmin, xmax = float(self.xax[0]), float(self.xax[-1])
        corners_world = np.array(
            [
                [zmin, ymin, xmin],
                [zmin, ymin, xmax],
                [zmin, ymax, xmin],
                [zmin, ymax, xmax],
                [zmax, ymin, xmin],
                [zmax, ymin, xmax],
                [zmax, ymax, xmin],
                [zmax, ymax, xmax],
            ],
            dtype=float,
        )

        mean_vox = float(np.mean(self.scale))
        corner_size = mean_vox * 0.15

        # Outline box (add first for final order control)
        box_edges = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [1, 3],
                [1, 5],
                [2, 3],
                [2, 6],
                [3, 7],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
            ],
            dtype=int,
        )
        edge_segments = [corners_world[e] for e in box_edges]
        v.add_shapes(
            edge_segments,
            shape_type="line",
            edge_color="yellow",
            edge_width=mean_vox * 0.05,
            opacity=0.9,
            blending="additive",
            name="Outline box",
        )

        v.add_points(
            corners_world,
            name="Outline corners",
            size=np.full(8, corner_size),
            face_color="red",
            opacity=0.9,
            blending="additive",
        )

        labels = []
        if self.space == "q":
            for z, y, x in corners_world:
                labels.append(f"Qx={x:.3f}, Qy={y:.3f}, Qz={z:.3f}")
        else:
            for z, y, x in corners_world:
                labels.append(f"H={x:.3f}, K={y:.3f}, L={z:.3f}")

        try:
            v.add_points(
                corners_world,
                name="Corner Labels",
                text={"string": labels, "size": 10},
                face_color="white",
                size=0.0,
                blending="additive",
            )
        except (TypeError, ValueError, AttributeError):
            v.add_points(
                corners_world,
                name="Corner Labels",
                text=labels,
                face_color="white",
                size=0.0,
                blending="additive",
            )

    def _add_axes_vectors(self, v: napari.Viewer) -> None:
        with contextlib.suppress(Exception):
            Lx = float(self.xax[-1] - self.xax[0])
            Ly = float(self.yax[-1] - self.yax[0])
            Lz = float(self.zax[-1] - self.zax[0])
            max_extent = max(Lx, Ly, Lz) or 1.0
            length = 0.10 * max_extent
            origin = np.array(
                [self.zax[0], self.yax[0], self.xax[0]], dtype=float
            )
            vectors = np.stack(
                [
                    np.vstack(
                        [origin, origin + np.array([length, 0, 0])]
                    ),  # +Z
                    np.vstack(
                        [origin, origin + np.array([0, length, 0])]
                    ),  # +Y
                    np.vstack(
                        [origin, origin + np.array([0, 0, length])]
                    ),  # +X
                ],
                axis=0,
            )
            width = float(np.mean(self.scale)) * 0.05
            kwargs = {
                "name": "World axes",
                "edge_color": ["cyan", "lime", "magenta"],
                "edge_width": width,
                "blending": "translucent_no_depth",
            }
            try:
                v.add_vectors(vectors, **kwargs)
            except TypeError:
                kwargs.pop("edge_width", None)
                kwargs["width"] = max(width, 1.0)
                v.add_vectors(vectors, **kwargs)

    def _install_hud(self, v: napari.Viewer, layer) -> None:
        def on_mouse_move(viewer, event):
            if not self._hud_enabled:
                return
            pos_world = viewer.cursor.position
            if pos_world is None:
                return
            try:
                zi, yi, xi = layer.world_to_data(pos_world)
            except (AttributeError, TypeError, ValueError):
                return
            intensity_val = np.nan
            zi_i, yi_i, xi_i = (int(np.round(q)) for q in (zi, yi, xi))
            if (
                0 <= zi_i < self.volume.shape[0]
                and 0 <= yi_i < self.volume.shape[1]
                and 0 <= xi_i < self.volume.shape[2]
            ):
                intensity_val = float(self.volume[zi_i, yi_i, xi_i])
            H = self._index_to_axis_value(self.xax, xi)
            K = self._index_to_axis_value(self.yax, yi)
            L = self._index_to_axis_value(self.zax, zi)
            if self.space == "q":
                text = f"Qx={H:.4f} Å⁻¹   Qy={K:.4f} Å⁻¹   Qz={L:.4f} Å⁻¹   I={intensity_val:.3g}"
            else:
                text = f"H={H:.4f}   K={K:.4f}   L={L:.4f}   I={intensity_val:.3g}"
            overlay = getattr(viewer, "text_overlay", None)
            if overlay is not None:
                overlay.visible = True
                overlay.position = "top_left"
                overlay.color = "white"
                overlay.font_size = 12
                overlay.text = text

        v.mouse_move_callbacks.append(on_mouse_move)

        @v.bind_key("C", overwrite=True)
        def _toggle_coords(viewer):
            self._hud_enabled = not self._hud_enabled
            overlay = getattr(viewer, "text_overlay", None)
            if overlay is not None:
                overlay.visible = self._hud_enabled

    # ------------------------------ internals ---------------------------------
    def _apply_display_order(self, v, desired_order):
        """
        Previous ordering strategy: iterate desired_order in given order and move each
        existing layer to the end so final list order matches desired_order exactly.
        """
        for name in desired_order:
            try:
                current_idx = [ly.name for ly in v.layers].index(name)
                target_idx = len(v.layers) - 1
                if current_idx != target_idx:
                    v.layers.move(current_idx, target_idx)
            except ValueError:
                continue  # layer not present; ignore

    def _force_volume(self, layer) -> None:
        if self.viewer is not None:
            self.viewer.dims.ndisplay = 3
        with contextlib.suppress(Exception):
            if hasattr(layer, "depiction"):
                layer.depiction = "volume"
        with contextlib.suppress(Exception):
            if hasattr(layer, "rendering"):
                for r in [
                    self.rendering,
                    "attenuated_mip",
                    "mip",
                    "translucent",
                ]:
                    try:
                        layer.rendering = r
                        break
                    except (AttributeError, TypeError):
                        continue

    @staticmethod
    def _validate_grid_axes(grid: np.ndarray, axes) -> None:
        if not isinstance(grid, np.ndarray) or grid.ndim != 3:
            raise ValueError("grid must be a 3D numpy array (nx, ny, nz).")
        xax, yax, zax = axes
        xax = np.asarray(xax)
        yax = np.asarray(yax)
        zax = np.asarray(zax)
        nx, ny, nz = grid.shape
        if any(a.ndim != 1 for a in (xax, yax, zax)):
            raise ValueError("All axes must be 1D.")
        if len(xax) != nx or len(yax) != ny or len(zax) != nz:
            raise ValueError("Axis lengths must match grid shape.")

    @staticmethod
    def _ensure_ascending(grid: np.ndarray, axes):
        xax, yax, zax = [np.asarray(a) for a in axes]
        G = grid
        if xax.size > 1 and xax[1] < xax[0]:
            xax = xax[::-1]
            G = G[::-1, :, :]
        if yax.size > 1 and yax[1] < yax[0]:
            yax = yax[::-1]
            G = G[:, ::-1, :]
        if zax.size > 1 and zax[1] < zax[0]:
            zax = zax[::-1]
            G = G[:, :, ::-1]
        return G, (xax, yax, zax)

    @staticmethod
    def _volume_from_grid_axes(grid, axes):
        xax, yax, zax = axes
        vol = np.ascontiguousarray(grid.transpose(2, 1, 0))

        def avg_step(a):
            return float(np.diff(a).mean()) if a.size > 1 else 1.0

        dx, dy, dz = avg_step(xax), avg_step(yax), avg_step(zax)
        translate = (float(zax[0]), float(yax[0]), float(xax[0]))
        scale = (dz, dy, dx)
        is_uniform = (
            (zax.size < 2 or np.allclose(np.diff(zax), dz))
            and (yax.size < 2 or np.allclose(np.diff(yax), dy))
            and (xax.size < 2 or np.allclose(np.diff(xax), dx))
        )
        return vol, scale, translate, is_uniform

    @staticmethod
    def _robust_percentiles(
        a: np.ndarray, prc: Tuple[float, float]
    ) -> Tuple[float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(a, prc)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
            if hi <= lo:
                hi = lo + 1.0
        return float(lo), float(hi)

    @staticmethod
    def _log1p_clip(a: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(np.asarray(a), 0.0))

    @staticmethod
    def _index_to_axis_value(ax: np.ndarray, idx: float) -> float:
        n = ax.size
        if n == 0:
            return np.nan
        if idx <= 0:
            return float(ax[0])
        if idx >= n - 1:
            return float(ax[-1])
        i0 = int(np.floor(idx))
        t = float(idx - i0)
        return float((1 - t) * ax[i0] + t * ax[i0 + 1])

    def _require_viewer(self) -> None:
        if self.viewer is None:
            raise RuntimeError("Call launch() first.")

    def _auto_zoom(self, v: napari.Viewer, layer) -> None:
        """
        Robust auto-zoom to fit the full data extent, restoring earlier behavior.
        Attempts (in order):
          1. reset_view()
          2. camera.set_range(...)
          3. set camera.center manually
          4. set an oblique viewing angle + reasonable zoom
        All steps are guarded; silently continues if not supported.
        """
        # Reset to a known baseline
        with contextlib.suppress(Exception):
            v.reset_view()

        cam = getattr(v, "camera", None)
        if cam is None:
            return

        # Get data extent (Z,Y,X)
        try:
            (z0, z1), (y0, y1), (x0, x1) = layer.extent.data
        except (AttributeError, TypeError, ValueError, IndexError):
            return

        # Try the direct napari convenience if available
        with contextlib.suppress(Exception):
            cam.set_range(range_z=(z0, z1), range_y=(y0, y1), range_x=(x0, x1))

        # Center camera explicitly (older napari sometimes needs this)
        with contextlib.suppress(Exception):
            cam.center = ((z0 + z1) / 2.0, (y0 + y1) / 2.0, (x0 + x1) / 2.0)

        # Provide a stable oblique angle
        with contextlib.suppress(Exception):
            cam.angles = (30, 30, 0)

        # Heuristic zoom: inversely related to max span (kept mild to avoid over-zoom)
        with contextlib.suppress(Exception):
            spans = np.array([z1 - z0, y1 - y0, x1 - x0], dtype=float)
            span_max = float(np.max(spans)) or 1.0
            # Normalize zoom to typical scale; clamp to sensible bounds
            target_zoom = 1.0
            if span_max > 0:
                target_zoom = min(
                    1.5, max(0.2, 1.0 * (100.0 / (50.0 + span_max)))
                )
            cam.zoom = target_zoom


# ------------------------------ Utilities -------------------------------------
def _maybe_series_to_list(obj):
    if (
        hasattr(obj, "to_numpy")
        and hasattr(obj, "values")
        and hasattr(obj, "iloc")
    ):
        try:
            return list(obj.to_numpy())
        except (AttributeError, TypeError, ValueError):
            try:
                return list(obj.values)
            except (AttributeError, TypeError, ValueError):
                return list(obj)
    return obj


def _stack_list_of_2d(
    frames: Sequence[np.ndarray],
    pad_value: float = np.nan,
    dtype=np.float32,
) -> np.ndarray:
    frames = [np.asarray(f) for f in frames if f is not None]
    if not frames:
        raise ValueError("Empty intensity list.")
    H = max(f.shape[0] for f in frames)
    W = max(f.shape[1] for f in frames)
    if all(f.shape == (H, W) for f in frames):
        return np.stack([f.astype(dtype, copy=False) for f in frames], 0)
    out = np.full((len(frames), H, W), pad_value, dtype=dtype)
    for i, f in enumerate(frames):
        h, w = f.shape
        out[i, :h, :w] = f
    return out


def _to_tyx_any(
    intensity: Union[np.ndarray, Sequence[np.ndarray]],
) -> np.ndarray:
    intensity = _maybe_series_to_list(intensity)
    if isinstance(intensity, (list, tuple)):
        return _stack_list_of_2d(intensity)
    a = np.asarray(intensity)
    if a.ndim == 3 and a.dtype != object:
        return a
    if a.ndim == 2 and a.dtype != object:
        return a[None, ...]
    if a.ndim == 1 and a.dtype == object:
        return _stack_list_of_2d(list(a))
    raise ValueError("Unsupported intensity input.")


class IntensityNapariViewer:
    """
    Simple intensity viewer with adjustable rectangular ROI.
    Layers (top→bottom after reorder): Intensity, ROI
    """

    def __init__(
        self,
        intensity: Union[np.ndarray, Sequence[np.ndarray]],
        *,
        name: str = "Intensity",
        log_view: bool = True,
        contrast_percentiles: Tuple[float, float] = (1.0, 99.8),
        cmap: str = "inferno",
        rendering: str = "attenuated_mip",  # kept for API compatibility
        add_timeseries: bool = True,  # NEW: backward compat (unused)
        add_volume: bool = False,  # NEW: backward compat (unused)
        scale_tzyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pad_value: float = np.nan,
    ):
        self._name = name
        self._log = bool(log_view)
        self._p_lo, self._p_hi = map(float, contrast_percentiles)
        self._cmap = cmap
        self._rendering = (
            rendering  # <— stored (not used for 2D stack, but kept)
        )
        self._scale = tuple(map(float, scale_tzyx))
        self._raw_tyx = _to_tyx_any(intensity).astype(np.float32, copy=False)
        self._viewer: napari.Viewer | None = None
        self._layer_ts = None
        self._pad_value = float(pad_value)
        self._add_timeseries = bool(add_timeseries)
        self._add_volume = bool(add_volume)
        self._roi_drag_callback = None
        self._roi_label_points = None
        self._roi_data_callback = None

    def launch(self, viewer: napari.Viewer | None = None) -> napari.Viewer:
        """Launch into a new Napari viewer or reuse the provided `viewer`.

        If `viewer` is None a new `napari.Viewer` is created; otherwise layers
        are added to the supplied viewer.
        """
        v = viewer or napari.Viewer(title=self._name)
        self._viewer = v
        # If reusing a provided viewer, clear its layers so intensity replaces
        # any existing content in the current viewer.
        if viewer is not None:
            with contextlib.suppress(Exception):
                for _ly in list(v.layers):
                    with contextlib.suppress(Exception):
                        v.layers.remove(_ly)
        data = self._prepare(self._raw_tyx)
        finite = data[np.isfinite(data)]
        lo, hi = (
            np.percentile(finite, [self._p_lo, self._p_hi])
            if finite.size
            else (0.0, 1.0)
        )
        if lo == hi:
            hi = lo + 1e-6
        self._layer_ts = v.add_image(
            data,
            name="Intensity",
            contrast_limits=(float(lo), float(hi)),
            colormap=self._cmap,
            blending="translucent",
            scale=self._scale,
        )
        v.dims.ndisplay = 2

        _, H, W = data.shape
        rect = np.array(
            [
                [H / 4, W / 4],
                [H / 4, 3 * W / 4],
                [3 * H / 4, 3 * W / 4],
                [3 * H / 4, W / 4],
            ]
        )
        shapes = v.add_shapes(
            [rect],
            shape_type="rectangle",
            edge_color="red",
            face_color="transparent",
            name="ROI",
        )
        shapes.editable = True
        with contextlib.suppress(Exception):
            shapes.mode = "transform"

        self._roi_label_points = self._create_roi_label_points(v, rect)
        self._update_roi_corner_labels(shapes)

        if self._roi_drag_callback is None:

            def _on_roi_drag(layer, event):
                if getattr(layer, "mode", None) != "transform":
                    return
                self._update_roi_corner_labels(layer)
                yield
                while True:
                    if event.type == "mouse_move":
                        self._update_roi_corner_labels(layer)
                    if event.type == "mouse_release":
                        break
                    yield
                self._update_roi_corner_labels(layer)

            self._roi_drag_callback = _on_roi_drag
        if self._roi_drag_callback not in shapes.mouse_drag_callbacks:
            shapes.mouse_drag_callbacks.append(self._roi_drag_callback)

        if self._roi_data_callback is None:

            def _on_roi_data(_event):
                self._update_roi_corner_labels(shapes)

            self._roi_data_callback = _on_roi_data
        with contextlib.suppress(Exception):
            shapes.events.data.connect(self._roi_data_callback, weak=False)

        with contextlib.suppress(Exception):
            names = [ly.name for ly in v.layers]
            for name in reversed(["Intensity", "ROI", "ROI Corner Labels"]):
                if name in names:
                    idx = names.index(name)
                    if idx != 0:
                        v.layers.move(idx, 0)
                        names = [ly.name for ly in v.layers]
        return v

    def _create_roi_label_points(self, viewer, coords: np.ndarray):
        labels = [
            f"({y:.2f}, {x:.2f})" for y, x in np.asarray(coords, dtype=float)
        ]
        try:
            return viewer.add_points(
                coords,
                name="ROI Corner Labels",
                text={"string": labels, "size": 10},
                face_color="white",
                size=0.0,
                blending="additive",
            )
        except (TypeError, ValueError, AttributeError):
            return viewer.add_points(
                coords,
                name="ROI Corner Labels",
                text=labels,
                face_color="white",
                size=0.0,
                blending="additive",
            )

    def _update_roi_corner_labels(self, shapes_layer) -> None:
        if (
            self._roi_label_points is None
            or shapes_layer is None
            or not shapes_layer.data
        ):
            return
        coords = np.asarray(shapes_layer.data[0], dtype=float)
        try:
            self._roi_label_points.data = coords
        except (AttributeError, TypeError):
            self._roi_label_points._data = coords
        labels = [f"({y:.2f}, {x:.2f})" for y, x in coords]
        text_obj = getattr(self._roi_label_points, "text", None)
        if text_obj is not None and hasattr(text_obj, "values"):
            try:
                text_obj.values = labels
            except (AttributeError, TypeError, ValueError):
                text_obj = None
        if text_obj is None:
            try:
                self._roi_label_points.text = {"string": labels, "size": 10}
            except (TypeError, AttributeError, ValueError):
                self._roi_label_points.text = labels
        self._roi_label_points.refresh()

    def _prepare(self, a: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(a, 0.0)) if self._log else a

    def close(self):
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.close()
            self._viewer = None

    def _set_roi_corner_text(self, shapes_layer, coords=None) -> None:
        if shapes_layer is None or (coords is None and not shapes_layer.data):
            return
        coords_np = np.asarray(
            coords if coords is not None else shapes_layer.data[0], dtype=float
        )
        labels = [f"({y:.2f}, {x:.2f})" for y, x in coords_np]
        text_obj = getattr(shapes_layer, "text", None)

        if text_obj is not None and hasattr(text_obj, "values"):
            try:
                text_obj.values = labels
                shapes_layer.refresh()
                return
            except (AttributeError, TypeError, ValueError):
                pass

        try:
            shapes_layer.text = {
                "string": labels,
                "size": 10,
                "color": "white",
                "anchor": "upper_left",
            }
        except (TypeError, AttributeError, ValueError):
            shapes_layer.text = labels
        shapes_layer.refresh()
