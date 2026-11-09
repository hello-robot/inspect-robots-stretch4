"""Readings and configs on the wire. Pure: no sockets.

Images travel in one of two modes, chosen by the client at `hello`:

- `forward` (default): each camera's JPEG exactly as the camera produced it. The client runs the
  same `decode()` as a local robot, so both robots' images go through one lossy pass.
- `lossless`: the service decodes and resizes, then sends PNG. Pixel-identical to `forward`, and
  smaller only when the resized frame compresses below the camera's JPEG.

Depth is already preprocessed, as uint16 millimetres, zlib-compressed: exact both ways.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots_stretch4.config import IMAGE_MODES, Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import EncodedReading, Reading, decode, decode_image
from inspect_robots_stretch4.net.framing import ProtocolError, expect_frames

_TUPLES = {f.name for f in dataclasses.fields(Stretch4EmbodimentConfig) if "tuple" in str(f.type)}


def config_to_json(cfg: Stretch4EmbodimentConfig) -> dict[str, Any]:
    return {
        f.name: list(v) if isinstance(v := getattr(cfg, f.name), tuple) else v
        for f in dataclasses.fields(cfg)
    }


def config_from_json(data: Any) -> Stretch4EmbodimentConfig:
    if not isinstance(data, dict):
        raise ProtocolError("`cfg` is not a JSON object")
    names = {f.name for f in dataclasses.fields(Stretch4EmbodimentConfig)}
    unknown = set(data) - names
    if unknown:
        raise ProtocolError(f"`cfg` has fields this side doesn't know: {sorted(unknown)}")
    try:
        return Stretch4EmbodimentConfig(
            **{k: tuple(v) if k in _TUPLES and isinstance(v, list) else v for k, v in data.items()}
        )
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"`cfg` is not a valid Stretch 4 config: {e}") from None


def pack_depth(depth_m: npt.NDArray[np.float32]) -> bytes:
    mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype("<u2")
    return zlib.compress(mm.tobytes(), 1)


def unpack_depth(blob: bytes, shape: Sequence[int]) -> npt.NDArray[np.float32]:
    try:
        mm = np.frombuffer(zlib.decompress(blob), dtype="<u2").reshape(tuple(shape))
    except (zlib.error, ValueError) as e:
        raise ProtocolError(f"a depth frame doesn't decode to {tuple(shape)}: {e}") from None
    return mm.astype(np.float32) / 1000.0


def encode_reading(
    encoded: EncodedReading, cfg: Stretch4EmbodimentConfig, mode: str
) -> tuple[dict[str, Any], list[bytes]]:
    """Service side: a reading -> (header fields, frames)."""
    cams = list(cfg.cameras)
    if mode == "forward":
        frames = [encoded.jpegs[c] for c in cams]
    elif mode == "lossless":
        images = decode(encoded, cfg).images
        frames = [_png(images[c]) for c in cams]
    else:
        raise ProtocolError(f"unknown image_mode {mode!r}; one of {IMAGE_MODES}")
    depth = list(encoded.depth)
    frames += [pack_depth(encoded.depth[d]) for d in depth]
    fields = {
        "t_robot": encoded.t,
        "joints": [float(v) for v in encoded.joints],
        "cams": cams,
        "depth": depth,
        "depth_shapes": [list(encoded.depth[d].shape) for d in depth],
    }
    return fields, frames


def decode_reading(
    header: dict[str, Any],
    frames: Sequence[bytes],
    cfg: Stretch4EmbodimentConfig,
    mode: str,
    to_local: Callable[[float], float],
) -> Reading:
    """Client side: (header, frames) -> a Reading on this machine's clock."""
    cams, depth = header.get("cams"), header.get("depth", [])
    if cams != list(cfg.cameras):
        raise ProtocolError(f"the reading has cameras {cams}, expected {list(cfg.cameras)}")
    shapes = header.get("depth_shapes", [])
    if not isinstance(depth, list) or len(shapes) != len(depth):
        raise ProtocolError("the reading's depth names and shapes don't match")
    expect_frames(header, frames, len(cams) + len(depth))
    joints = np.asarray(header.get("joints"), dtype=np.float64)
    n = len(cfg.joints)
    if joints.shape != (n,):
        raise ProtocolError(f"the reading has joints of shape {joints.shape}, expected ({n},)")
    if mode == "forward":
        images = {c: decode_image(c, f, cfg) for c, f in zip(cams, frames, strict=False)}
    else:
        images = {c: _unpng(c, f) for c, f in zip(cams, frames, strict=False)}
    blobs = frames[len(cams) :]
    return Reading(
        joints=joints,
        images=images,
        t=to_local(float(header["t_robot"])),
        depth={d: unpack_depth(b, s) for d, b, s in zip(depth, blobs, shapes, strict=True)},
    )


def _png(rgb: npt.NDArray[np.uint8]) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _unpng(camera: str, blob: bytes) -> npt.NDArray[np.uint8]:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ProtocolError(f"camera {camera!r}'s PNG does not decode")
    return np.ascontiguousarray(bgr[..., ::-1])
