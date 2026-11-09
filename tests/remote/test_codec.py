"""Readings and configs on the wire: forwarded JPEGs decode exactly as a local read would."""

from __future__ import annotations

import json

import numpy as np
import pytest

from inspect_robots_stretch4 import Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import EncodedReading, decode
from inspect_robots_stretch4.net.framing import ProtocolError
from inspect_robots_stretch4.remote.codec import (
    config_from_json,
    config_to_json,
    decode_reading,
    encode_reading,
    pack_depth,
    unpack_depth,
)

CFG = Stretch4EmbodimentConfig(base="none", max_image_side=8, gripper_depth=True)


@pytest.fixture
def encoded(driver) -> EncodedReading:
    depth_mm = np.random.default_rng(3).integers(0, 4000, (5, 8)).astype(np.uint16)
    driver.depth = {"gripper_right": depth_mm.astype(np.float32) / 1000.0}
    driver.t = 12.5
    return driver.read_encoded()


def test_config_round_trips_through_json() -> None:
    cfg = Stretch4EmbodimentConfig(cameras=("head_left", "gripper_right"), home_pose_file="h.json")
    assert config_from_json(json.loads(json.dumps(config_to_json(cfg)))) == cfg


@pytest.mark.parametrize(
    ("data", "match"),
    [([], "not a JSON object"), ({"warp": 9}, "doesn't know"), ({"gripper": "hook"}, "valid")],
)
def test_bad_configs_are_protocol_errors(data, match) -> None:
    with pytest.raises(ProtocolError, match=match):
        config_from_json(data)


def test_depth_round_trips_exactly() -> None:
    depth = np.random.default_rng(0).integers(0, 65535, (5, 8)).astype(np.float32) / 1000.0
    assert np.array_equal(unpack_depth(pack_depth(depth), (5, 8)), depth)
    with pytest.raises(ProtocolError, match="doesn't decode"):
        unpack_depth(pack_depth(depth), (4, 8))


@pytest.mark.parametrize("mode", ["forward", "lossless"])
def test_remote_images_equal_a_local_decode_exactly(encoded, mode) -> None:
    fields, frames = encode_reading(encoded, CFG, mode)
    if mode == "forward":  # the camera's bytes, untouched
        assert frames[:2] == [encoded.jpegs["head_right"], encoded.jpegs["gripper_right"]]
    reading = decode_reading(fields, frames, CFG, mode, to_local=lambda t: t - 10.0)
    local = decode(encoded, CFG)
    for camera in CFG.cameras:
        assert np.array_equal(reading.images[camera], local.images[camera])
    assert reading.images["head_right"].shape == (5, 8, 3)
    assert np.array_equal(reading.depth["gripper_right"], encoded.depth["gripper_right"])
    assert np.array_equal(reading.joints, encoded.joints)
    assert reading.t == pytest.approx(2.5)


def test_malformed_readings_are_protocol_errors(encoded) -> None:
    fields, frames = encode_reading(encoded, CFG, "forward")
    with pytest.raises(ProtocolError, match="binary frames"):
        decode_reading(fields, frames[:-1], CFG, "forward", float)
    with pytest.raises(ProtocolError, match="cameras"):
        decode_reading({**fields, "cams": ["head_left"]}, frames, CFG, "forward", float)
    with pytest.raises(ProtocolError, match="joints"):
        decode_reading({**fields, "joints": [0.0]}, frames, CFG, "forward", float)
    with pytest.raises(ProtocolError, match="image_mode"):
        encode_reading(encoded, CFG, "gif")
