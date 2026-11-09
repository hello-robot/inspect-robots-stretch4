"""The hardware seam: the small surface the embodiment needs from a Stretch 4.

`Stretch4Driver` is what the embodiment talks to, so the whole embodiment runs in tests with
a fake. `BodyDriver` is the real one, on stretch4_body alone (no ROS, no LeRobot):

- Joints: a thread polls stretch_body_server's status (status only, so it takes no command
  lease and coexists with other clients) into a per-joint history.
- Cameras: one stretch4_body camera pipeline per OAK device (head, gripper), drained by a
  thread into a per-eye buffer of timestamped JPEGs.
- Each observation is clocked by the first camera's next frame: the other cameras' frames
  nearest its capture time are waited for, and the joints are interpolated to that time, so
  the state is the robot at the moment the images were taken.
- Commands: `move_to` absolute targets on a second client, pushed at a fixed priority.
- Link: the server silently drops commands from a client that doesn't hold its command lease,
  so every read and push checks that status is still arriving and that the lease is ours.
- Base: a thread closes a turn-rate loop on wheel odometry toward the latest heading target,
  so a turn finishes while the policy is thinking, as `move_to` does for the arm. The base only
  turns in place (it is never sent a sideways or forward velocity), and its heading is reported
  relative to the last `zero_base()`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import statistics
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4.config import (
    BASE_JOINTS,
    CAMERA_CROP,
    CAMERA_NATIVE_HW,
    CAMERA_ROTATE,
    DEPTH_CAMERA,
    JOINTS,
    Stretch4EmbodimentConfig,
    crop_box,
)

log = logging.getLogger(__name__)

# Modules the real driver imports that package metadata cannot express.
# `inspect-robots doctor` prints these remedies when a module is missing.
RUNTIME_REQUIREMENTS: dict[str, str] = {
    "stretch4_body": "stretch4_body ships with the Stretch 4 software install (stretch4_install)",
    "cv2": "pip install opencv-python-headless (or use the robot's system OpenCV)",
}

WRIST_JOINTS = ("wrist_yaw", "wrist_pitch", "wrist_roll")
GRIPPER_JOINTS = ("stretch_gripper", "parallel_gripper")  # SG4, PG4

# camera name -> (stretch4_body device, eye)
CAMERAS: dict[str, tuple[str, str]] = {
    "head_left": ("head", "left"),
    "head_right": ("head", "right"),
    "gripper_left": ("gripper", "left"),
    "gripper_right": ("gripper", "right"),
}

# move_to speed/acceleration limits during a run, and slower ones for driving home.
RUN_LIMITS = {"v_m": 0.10, "a_m": 0.25, "v_r": 1.5, "a_r": 4.0}
HOME_LIMITS = {"v_m": 0.04, "a_m": 0.10, "v_r": 0.6, "a_r": 2.0}
ARRIVAL_TOL = {
    "lift": 0.006,
    "arm": 0.006,
    "wrist_yaw": 0.03,
    "wrist_pitch": 0.03,
    "wrist_roll": 0.03,
    "grip_mm": 3.0,
}

# Pair camera frames at most this far apart in capture time, and wait at most this long for a
# camera's matching frame.
MAX_SKEW_S = 0.020
MATCH_TIMEOUT_S = 0.060
JOINT_POLL_HZ = 200.0

# stretch_body_server publishes status at its 100 Hz control rate: this much silence means it
# has stopped, and the joints and run-stop state would otherwise freeze at their last values.
STATUS_STALE_S = 0.25
# The server's command lease lapses this long after the holder's last push (LEASE_TIMEOUT).
LEASE_TIMEOUT_S = 1.1
# A status received this long after the first push of a burst shows whether the server took it.
LEASE_CONFIRM_S = 0.15
# Another client's lease is reported only after it has been held this long: a software
# run-stop from another client takes the lease a few status cycles before the run-stop shows.
TAKEOVER_SETTLE_S = 0.1
# Above any client's normal priority, so a software run-stop is never dropped for want of the
# lease (a strictly higher priority takes it).
RUNSTOP_PRIORITY = 100
# The base turn loop: proportional gain (1/s), turn-rate cap (rad/s), and when a heading
# counts as reached (rad). A target the loop has chased for BASE_TARGET_TIMEOUT_S is dropped
# (blocked or in contact) and the base stops. The wheel motors' own watchdog stops them ~1 s
# after the last velocity command, should this process die.
BASE_HZ = 20.0
BASE_GAIN = 2.0
BASE_MAX_W = 0.3
BASE_TOL = 0.02
BASE_TARGET_TIMEOUT_S = 3.0


class LinkLost(EmbodimentFault):
    """A robot reached over the network has gone quiet: no heartbeat, or a request timed out.

    The robot holds where it is, so the caller may wait and try to reconnect (a served driver's
    `reconnect`) instead of ending the run at once.
    """


@dataclass(frozen=True)
class Reading:
    """One synced observation: joints in the config's action order (`cfg.joints`), images by
    camera name, and when.

    `t` is the capture time on the monotonic clock (`time.monotonic()`) of the process that
    called `read()`. A driver that reads a remote robot converts the robot's clock to this one.
    """

    joints: npt.NDArray[np.float64]
    images: Mapping[str, npt.NDArray[np.uint8]]
    t: float
    # Metric depth (m, 0 = none) by camera name, at the image size; only with gripper_depth.
    depth: Mapping[str, npt.NDArray[np.float32]] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedReading:
    """A `Reading` before its images are decoded: each camera's JPEG exactly as the camera sent
    it. A remote robot sends these, so both robots' images go through one `decode()`."""

    joints: npt.NDArray[np.float64]
    jpegs: Mapping[str, bytes]
    t: float
    depth: Mapping[str, npt.NDArray[np.float32]] = field(default_factory=dict)


def decode(encoded: EncodedReading, cfg: Stretch4EmbodimentConfig) -> Reading:
    """Decode and preprocess every camera's JPEG: the same images on either robot."""
    images = {c: decode_image(c, jpeg, cfg) for c, jpeg in encoded.jpegs.items()}
    return Reading(joints=encoded.joints, images=images, t=encoded.t, depth=encoded.depth)


def decode_image(camera: str, jpeg: bytes, cfg: Stretch4EmbodimentConfig) -> npt.NDArray[np.uint8]:
    """One camera's JPEG -> upright RGB at the camera's configured shape."""
    import cv2

    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise EmbodimentFault(f"camera {camera!r} sent a frame that does not decode")
    return preprocess(bgr, CAMERA_ROTATE[camera], cfg.image_shape(camera), CAMERA_CROP.get(camera))


@runtime_checkable
class Stretch4Driver(Protocol):
    """What the embodiment needs from the robot."""

    def read(self) -> Reading:
        """Block for the next synced camera frame and return it with the joints."""
        ...

    def command(self, targets: Mapping[str, float]) -> Mapping[str, float]:
        """Send absolute joint targets; return the targets sent."""
        ...

    def goto(self, pose: Mapping[str, float], timeout_s: float) -> tuple[bool, Any]:
        """Drive slowly to `pose`; return (arrived, worst remaining error)."""
        ...

    def is_runstopped(self) -> bool:
        """True while the robot's run-stop is engaged."""
        ...

    def trigger_runstop(self) -> None:
        """Engage the run-stop from software (to stop this robot when another one stops)."""
        ...

    def release_base(self) -> None:
        """Stop chasing any base target and let the wheels spin freely, so the operator can
        push the robot into place."""
        ...

    def zero_base(self) -> None:
        """Hold the wheels and make the base's current pose the new zero."""
        ...

    def disconnect(self) -> None:
        """Release cameras, robot clients and threads."""
        ...


DriverFactory = Callable[[Stretch4EmbodimentConfig], Stretch4Driver]


# -- pure helpers (tested without a robot) ----------------------------------


class Timeline:
    """Timestamped values in a ring buffer: newest, nearest to a time, or interpolated."""

    def __init__(self, maxlen: int = 600) -> None:
        self._lock = threading.Lock()
        self._items: deque[tuple[float, Any]] = deque(maxlen=maxlen)

    def put(self, t: float, value: Any) -> None:
        """Append, ignoring anything not newer than the last item."""
        with self._lock:
            if not self._items or t > self._items[-1][0]:
                self._items.append((t, value))

    def latest(self) -> tuple[float, Any] | None:
        """The newest (t, value), or None if empty."""
        with self._lock:
            return self._items[-1] if self._items else None

    def nearest(self, t: float) -> tuple[float, Any] | None:
        """The (t, value) captured closest to `t`, or None if empty."""
        with self._lock:
            return min(self._items, key=lambda item: abs(item[0] - t), default=None)

    def interpolate(self, t: float) -> float | None:
        """A numeric value linearly interpolated at `t` (held at the ends), or None if empty."""
        with self._lock:
            items = list(self._items)
        if not items:
            return None
        times = [ti for ti, _ in items]
        i = int(np.searchsorted(times, t))
        if i == 0:
            return float(items[0][1])
        if i >= len(items):
            return float(items[-1][1])
        (t0, v0), (t1, v1) = items[i - 1], items[i]
        return float(v0 + (t - t0) / (t1 - t0) * (v1 - v0))


def preprocess(
    bgr: npt.NDArray[np.uint8],
    rotate: int,
    out_hw: tuple[int, int],
    crop_hw: tuple[int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    """A decoded camera frame -> upright RGB at `out_hw`: rotate by quarter turns, take the
    centred `crop_hw`-aspect box if given (else the whole frame), then resize. out_hw has the
    result's aspect; see Stretch4EmbodimentConfig.image_shape."""
    import cv2

    img = _upright(bgr, rotate, crop_hw)
    out_h, out_w = out_hw
    if img.shape[:2] != (out_h, out_w):
        interp = cv2.INTER_AREA if img.shape[1] > out_w else cv2.INTER_LINEAR
        img = cv2.resize(img, (out_w, out_h), interpolation=interp)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_depth(
    depth_mm: npt.NDArray[np.uint16],
    rotate: int,
    out_hw: tuple[int, int],
    crop_hw: tuple[int, int] | None = None,
) -> npt.NDArray[np.float32]:
    """A raw depth map (uint16 mm, 0 = no depth) -> meters with the same rotation and crop as
    `preprocess`, resized nearest-neighbour so no pixel blends a near and a far surface."""
    import cv2

    img = _upright(depth_mm, rotate, crop_hw)
    out_h, out_w = out_hw
    if img.shape[:2] != (out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.float32) / 1000.0


def _upright(
    img: npt.NDArray[Any], rotate: int, crop_hw: tuple[int, int] | None
) -> npt.NDArray[Any]:
    """Rotate by quarter turns, then the centred `crop_hw`-aspect box if one is given."""
    img = np.ascontiguousarray(np.rot90(img, k=rotate)) if rotate else img
    if crop_hw is None:
        return img
    top, bottom, left, right = crop_box(img.shape[:2], crop_hw)
    return img[top:bottom, left:right]


def check_camera_config(camera: str, image_size: Any, rotate: int) -> None:
    """Fail if stretch4_body captures `camera` at another size or orientation than
    CAMERA_NATIVE_HW and CAMERA_ROTATE say, which would distort every image."""
    found = (tuple(int(v) for v in image_size), int(rotate))
    expected = (CAMERA_NATIVE_HW[camera], CAMERA_ROTATE[camera])
    if found != expected:
        raise EmbodimentFault(
            f"stretch4_body captures camera {camera!r} as (height, width), quarter turns = "
            f"{found}, but this package expects {expected}; update CAMERA_NATIVE_HW and "
            f"CAMERA_ROTATE in config.py"
        )


def _as_bytes(jpeg: Any) -> bytes:
    """A camera frame's JPEG (stretch4_body gives a uint8 array or bytes) as bytes."""
    return jpeg.tobytes() if isinstance(jpeg, np.ndarray) else bytes(jpeg)


def gripper_joint(end_of_arm_joints: list[str]) -> str:
    """The installed gripper's joint name."""
    name = next((j for j in GRIPPER_JOINTS if j in end_of_arm_joints), None)
    if name is None:
        raise EmbodimentFault(
            f"no {' or '.join(GRIPPER_JOINTS)} on the end of arm (joints: {end_of_arm_joints})"
        )
    return name


def arrived(pose: Mapping[str, float], current: Mapping[str, float]) -> tuple[bool, dict]:
    """Whether every joint is within ARRIVAL_TOL of `pose`, and the errors."""
    err = {j: abs(current[j] - pose[j]) for j in JOINTS}
    return all(err[j] <= ARRIVAL_TOL[j] for j in JOINTS), err


def drive_to(
    pose: Mapping[str, float],
    timeout_s: float,
    *,
    send: Callable[[Mapping[str, float]], None],
    current: Callable[[], Mapping[str, float]],
    runstopped: Callable[[], bool],
    cancelled: Callable[[], bool] = lambda: False,
    period_s: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, Any]:
    """Re-send `pose` every `period_s` until it is reached on two checks in a row; return
    (arrived, worst remaining error). Re-sending is what keeps the server's lease.

    An engaged run-stop ends the drive at once: re-sending through it would restart the motion
    the moment it is released. Cancelling ends it too, holding where the robot is:
    (False, "cancelled").
    """
    deadline, settled, err = clock() + timeout_s, 0, {}
    while clock() < deadline:
        if runstopped():
            raise SafetyAbort("the run-stop was engaged while driving to the home pose")
        if cancelled():
            send(current())
            return False, "cancelled"
        send(pose)
        sleep(period_s)
        ok, err = arrived(pose, current())
        settled = settled + 1 if ok else 0
        if settled >= 2:
            return True, err
    return False, max(err.items(), key=lambda kv: kv[1] / ARRIVAL_TOL[kv[0]], default=None)


class LeaseWatch:
    """Whether stretch_body_server is still publishing status and still obeying us.

    The server obeys one client at a time and drops everyone else's commands without a reply,
    so a command can only be confirmed through status: `lease_holder` must name us once the
    server has had time to see our pushes. The lease lapses LEASE_TIMEOUT_S after the last push
    (during a long LLM call, say), and the next push starts a new burst that must be confirmed
    again. While the run-stop is engaged the lease is not reported: the run-stop is.
    """

    def __init__(self, client_id: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.client_id = client_id
        self._clock = clock
        self._lock = threading.Lock()
        self._status_t = clock()  # fresh at connect; the first status follows within ms
        self._holder: tuple[float, str | None] | None = None  # (received at, lease_holder)
        self._foreign_since: float | None = None  # first status of another client's lease
        self._runstopped = False
        self._burst_start: float | None = None
        self._last_push: float | None = None

    def status(self, lease_holder: str | None, runstopped: bool = False) -> None:
        """A status message arrived."""
        with self._lock:
            self._status_t = self._clock()
            self._holder = (self._status_t, lease_holder)
            self._runstopped = runstopped
            if lease_holder == self.client_id:
                self._foreign_since = None
            elif self._foreign_since is None:
                self._foreign_since = self._status_t

    def pushed(self) -> None:
        """We pushed a command."""
        with self._lock:
            now = self._clock()
            if self._last_push is None or now - self._last_push > LEASE_TIMEOUT_S:
                self._burst_start = now
            self._last_push = now

    def check(self) -> None:
        """EmbodimentFault if status has stopped; SafetyAbort if another client holds the lease
        while ours should be live."""
        with self._lock:
            now = self._clock()
            age = now - self._status_t
            holder, burst_start, last_push = self._holder, self._burst_start, self._last_push
            foreign_since, runstopped = self._foreign_since, self._runstopped
        if age > STATUS_STALE_S:
            raise EmbodimentFault(
                f"no status from stretch_body_server for {age:.2f}s; is it still running?"
            )
        if runstopped or holder is None or burst_start is None or last_push is None:
            return
        seen_at, name = holder
        if name == self.client_id or foreign_since is None:
            return
        live = now - last_push < LEASE_TIMEOUT_S
        settled = seen_at - foreign_since >= TAKEOVER_SETTLE_S
        if live and seen_at - burst_start >= LEASE_CONFIRM_S and settled:
            raise SafetyAbort(
                f"another program ({name}) holds Stretch's command lease, so our commands are "
                "being ignored; stop it, or raise -E priority"
            )


def load_home_pose(path: str) -> dict[str, float]:
    """A pose file: {"joints": {"lift": ..., ..., "grip_mm": ...}} (the stretch4_pose format)."""
    data = json.loads(Path(path).expanduser().read_text())
    joints = data.get("joints") or {}
    missing = [j for j in JOINTS if j not in joints]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return {j: float(joints[j]) for j in JOINTS}


def base_turn_rate(target: float, current: float) -> float | None:
    """One step of the base loop: the turn rate (rad/s, counter-clockwise) that closes on the
    `target` heading, or None once it is within BASE_TOL."""
    error = target - current
    if abs(error) <= BASE_TOL:
        return None
    return max(-BASE_MAX_W, min(BASE_MAX_W, BASE_GAIN * error))


def wall_minus_monotonic(samples: int = 51) -> float:
    """time.time() - time.monotonic(), median of sandwiched reads to reject scheduling noise.
    Server status is stamped with wall time, camera frames with monotonic time."""
    offsets = []
    for _ in range(samples):
        m0 = time.monotonic()
        w = time.time()
        m1 = time.monotonic()
        offsets.append(w - (m0 + m1) / 2.0)
    return statistics.median(offsets)


# -- the real driver ----------------------------------------------------------


class _CameraDevice:  # pragma: no cover - needs a Stretch 4
    """One OAK device (both eyes), opened once and drained by a thread into per-eye Timelines
    of (monotonic capture time, JPEG)."""

    def __init__(self, device: str) -> None:
        from stretch4_body.subsystem.cameras.controllers.camera_pipeline_controller import (
            RGBPipelineController,
        )
        from stretch4_body.subsystem.cameras.enums.rgb_camera import RGBCameras

        camera_type = (
            RGBCameras.head_left_right_compressed
            if device == "head"
            else RGBCameras.gripper_rgbd_compressed
        )
        extra = {} if device == "head" else {"enable_pointcloud": False}
        self.device = device
        self.eyes = {"left": Timeline(32), "right": Timeline(32)}
        self.depth = Timeline(8)  # gripper only: uint16 mm, aligned to the right eye
        self._stop = threading.Event()
        # The device re-enumerates for a few seconds after a previous close.
        log.info("connecting to the %s camera...", device)
        for attempt in range(6):
            try:
                self._controller = RGBPipelineController(
                    camera_type=camera_type,
                    recording_directory=None,
                    show_image_in=None,
                    is_rotate=False,
                    is_rectify=False,
                    is_crop=False,
                    ai_models_to_use=[],
                    detect_aruco_marker_size=None,
                    **extra,
                )
                break
            except Exception as e:
                if attempt == 5:
                    raise
                log.warning(
                    "%s camera not ready (attempt %d/6), retrying: %s", device, attempt + 1, e
                )
                time.sleep(3.0)
        log.info("%s camera connected", device)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"oak-{device}")
        self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self._controller.get_frame_synced(is_run_pipeline=False):
                if self._stop.is_set():
                    break
                if frame is None:
                    continue
                for eye, timeline in self.eyes.items():
                    img = getattr(frame, eye, None)
                    if img is not None and img.timestamp > 0:
                        timeline.put(img.timestamp, img.image)
                depth = getattr(frame, "depth", None)
                if depth is not None and frame.timestamp > 0:
                    self.depth.put(frame.timestamp, depth)
        except Exception as e:
            if not self._stop.is_set():
                log.error("%s camera thread stopped: %s", self.device, e)

    def close(self) -> None:
        # Leave the capture loop before closing: closing under a blocked read wedges the
        # device for the next open.
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._controller.stop()
        finally:
            time.sleep(1.0)  # the device reboots on close


class BodyDriver:  # pragma: no cover - needs a Stretch 4
    """`Stretch4Driver` on stretch4_body: status thread, OAK camera threads, move_to commands."""

    def __init__(self, cfg: Stretch4EmbodimentConfig) -> None:
        import cv2  # noqa: F401 - fail at connect, not mid-run
        from stretch4_body.subsystem.cameras.enums.rgb_camera import RGBCameras
        from stretch4_body.utils.tool_metadata import get_tool_metadata

        unknown = [c for c in cfg.cameras if c not in CAMERAS]
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; Stretch 4 has {sorted(CAMERAS)}")
        self._cfg = cfg
        self._stop = threading.Event()
        self._runstopped = False
        self._on_runstop: Callable[[], None] | None = None
        self._joints = {j: Timeline() for j in JOINTS}
        self._odom = Timeline()  # raw odometry heading
        self._base_ref: float | None = None  # the zero heading; set at the first odometry
        self._base_target: dict[str, float] | None = None
        self._base_target_t = 0.0
        self._base_moving = False
        self._last_sync_t: float | None = None
        self._devices: dict[str, _CameraDevice] = {}
        self._status = self._cmd = None
        self._threads: list[threading.Thread] = []
        # Unique per process: the server treats clients sharing an id as one lease holder, so
        # two evals with a fixed id would both drive the robot, interleaved.
        tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._watch = LeaseWatch(f"inspect_robots_stretch4_cmd_{tag}")
        # The command client is used from the caller's thread, the base loop's thread and, for
        # trigger_runstop(), from whichever thread notices another robot stop.
        self._cmd_lock = threading.Lock()
        self._cancel_goto = threading.Event()
        try:
            # Two clients: the status thread owns the first one's socket.
            self._status = self._client(f"inspect_robots_stretch4_status_{tag}")
            if not self._status.is_homed():
                raise EmbodimentFault("the robot is not homed; run stretch_robot_home first")
            self._cmd = self._client(self._watch.client_id)
            self._gripper = gripper_joint(list(self._status.end_of_arm.joints))
            if self._gripper != cfg.gripper:
                raise EmbodimentFault(
                    f"this robot has a {self._gripper}, but the embodiment is configured for "
                    f"a {cfg.gripper}; pass -E gripper={self._gripper}"
                )
            self._grip_meta = get_tool_metadata(self._gripper)

            t = threading.Thread(
                target=self._poll_status,
                args=(wall_minus_monotonic(),),
                daemon=True,
                name="stretch4-status",
            )
            t.start()
            self._threads.append(t)
            if cfg.base != "none":
                t = threading.Thread(target=self._run_base, daemon=True, name="stretch4-base")
                t.start()
                self._threads.append(t)

            for c in cfg.cameras:
                camera = RGBCameras[c].config
                check_camera_config(c, camera.image_size, camera.rotate_number_of_times)
            for device in {CAMERAS[c][0] for c in cfg.cameras}:
                self._devices[device] = _CameraDevice(device)
            self._wait_for_first_frames()
        except BaseException:
            self.disconnect()
            raise

    @staticmethod
    def _client(client_id: str) -> Any:
        from stretch4_body.robot.robot_client import RobotClient

        robot = RobotClient(client_id=client_id)
        if not robot.startup(verbose=False):
            raise EmbodimentFault(
                "could not connect to stretch_body_server; start it with "
                "`stretch_body_server --launch`"
            )
        return robot

    # -- threads ----------------------------------------------------------

    def _poll_status(self, offset: float) -> None:
        """Pull status flat out into the joint timelines (monotonic time)."""
        period, failures = 1.0 / JOINT_POLL_HZ, 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self._status.pull_status(blocking=False):
                    self._record(self._status.status, offset)
            except Exception as e:
                # A status missing a field must not silently end the thread.
                failures += 1
                if failures in (1, 10) or failures % 1000 == 0:
                    log.warning(
                        "stretch4 status read failed (%s: %s); %d so far",
                        type(e).__name__,
                        e,
                        failures,
                    )
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _record(self, s: dict, offset: float) -> None:
        was, self._runstopped = self._runstopped, bool(self._status.is_runstopped())
        if self._runstopped and not was and self._on_runstop is not None:
            try:
                self._on_runstop()
            except Exception as e:
                log.error("the run-stop callback failed: %s", e)
        eoa = s["end_of_arm"]
        for joint, status in (
            ("lift", s["lift"]),
            ("arm", s["arm"]),
            *((j, eoa[j]) for j in WRIST_JOINTS),
        ):
            self._joints[joint].put(status["timestamp_pc"] - offset, float(status["pos"]))
        g = eoa[self._gripper]
        aperture_mm = self._grip_meta.status_to_metadata(g)["aperture_m"] * 1000.0
        self._joints["grip_mm"].put(g["timestamp_pc"] - offset, float(aperture_mm))
        # The omnibase's own timestamp_pc is never updated; odometry is stamped with wall
        # time in pose_time_s each time it is integrated.
        b = s["omnibase"]
        t_odom = b.get("pose_time_s") or b["wheel_0"]["timestamp_pc"]
        self._odom.put(float(t_odom) - offset, float(b["theta"]))
        if self._base_ref is None:
            self._base_ref = float(b["theta"])
        # Last, so a status that failed to parse doesn't count as fresh.
        self._watch.status(s.get("server", {}).get("lease_holder"), self._runstopped)

    def _run_base(self) -> None:
        """Chase the latest base target with velocity commands; stop when it is reached,
        dropped, or the run-stop is engaged."""
        period = 1.0 / BASE_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._base_tick(t0)
            except Exception as e:
                # Includes a lost lease or link; the main thread's next read or push reports it.
                log.warning(
                    "stretch4 base loop failed (%s: %s); stopping the base", type(e).__name__, e
                )
                self._base_target = None
                # If this fails too, the wheel watchdog stops the base ~1 s after the last command.
                with contextlib.suppress(Exception):
                    self._send_base(0.0)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _base_tick(self, now: float) -> None:
        target = self._base_target
        if self._runstopped or target is None or now - self._base_target_t > BASE_TARGET_TIMEOUT_S:
            self._base_target = None
            if self._base_moving:
                self._send_base(0.0)
            return
        w = base_turn_rate(target["base_theta"], self._heading_at(time.monotonic()))
        if w is None:
            if self._base_moving:
                self._send_base(0.0)
            return
        self._send_base(w)

    def _send_base(self, w: float) -> None:
        """Turn in place: no forward or sideways velocity, ever."""
        self._base_push(lambda base: base.set_velocity(0.0, 0.0, w))
        self._base_moving = w != 0.0

    def _base_push(self, queue: Callable[[Any], None]) -> None:
        """Check the link, then queue one base command and push it."""
        self._watch.check()
        with self._cmd_lock:
            if self._cmd is None:
                return
            queue(self._cmd.omnibase)
            self._push()
        self._watch.pushed()

    def _wait_for_first_frames(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        for c in self._cfg.cameras:
            device, eye = CAMERAS[c]
            while self._devices[device].eyes[eye].latest() is None:
                if time.monotonic() > deadline:
                    raise EmbodimentFault(f"no frames from camera {c!r} after {timeout_s:.0f}s")
                time.sleep(0.02)

    # -- Stretch4Driver ---------------------------------------------------

    def read(self) -> Reading:
        return decode(self.read_encoded(), self._cfg)

    def read_encoded(self) -> EncodedReading:
        """`read()` without decoding the images, for sending them to another machine."""
        self._watch.check()
        cams = self._cfg.cameras
        while True:  # a sync frame without close-enough partners is skipped for the next one
            t_ref, jpegs = self._next_sync_frame(cams[0])
            for c in cams[1:]:
                match = self._match(c, t_ref)
                if match is None:
                    break
                jpegs[c] = match
            else:
                break
        depth = {}
        if self._cfg.gripper_depth:
            nearest = self._devices["gripper"].depth.nearest(t_ref)
            if nearest is None:
                raise EmbodimentFault("gripper_depth is on, but the gripper camera sent no depth")
            depth[DEPTH_CAMERA] = preprocess_depth(
                nearest[1],
                CAMERA_ROTATE[DEPTH_CAMERA],
                self._cfg.image_shape(DEPTH_CAMERA),
                CAMERA_CROP.get(DEPTH_CAMERA),
            )
        return EncodedReading(
            joints=self._all_joints_at(t_ref),
            jpegs={c: _as_bytes(jpeg) for c, jpeg in jpegs.items()},
            t=t_ref,
            depth=depth,
        )

    def command(self, targets: Mapping[str, float]) -> Mapping[str, float]:
        if any(j in targets for j in BASE_JOINTS):
            self._base_target = {j: float(targets[j]) for j in BASE_JOINTS}
            self._base_target_t = time.monotonic()
        self._send(targets, RUN_LIMITS)
        return dict(targets)

    def goto(self, pose: Mapping[str, float], timeout_s: float) -> tuple[bool, Any]:
        """Re-push at 10 Hz: the server's command lease expires ~1 s after the last push."""
        self._cancel_goto.clear()
        return drive_to(
            pose,
            timeout_s,
            send=lambda p: self._send(p, HOME_LIMITS),
            current=lambda: dict(zip(JOINTS, self._joints_at(time.monotonic()), strict=True)),
            runstopped=self.is_runstopped,
            cancelled=self._cancel_goto.is_set,
        )

    def cancel_goto(self) -> None:
        """End a running `goto` from another thread; the robot holds where it is."""
        self._cancel_goto.set()

    def is_runstopped(self) -> bool:
        return self._runstopped

    def on_runstop(self, callback: Callable[[], None] | None) -> None:
        """Call `callback` from the status thread each time the run-stop engages. It must
        return at once: the status thread reads the joints at JOINT_POLL_HZ."""
        self._on_runstop = callback

    def request_runstop(self) -> None:
        """`trigger_runstop()`: one push, so it doesn't block."""
        self.trigger_runstop()

    def trigger_runstop(self) -> None:
        # Pushed through the lease like any command, so at a priority that takes it.
        with self._cmd_lock:
            if self._cmd is None:
                return
            self._cmd.trigger_runstop()
            self._cmd.push_command(ignore_control_lock=True, priority=RUNSTOP_PRIORITY)

    def release_base(self) -> None:
        self._base_target = None
        self._base_push(lambda base: base.enable_freewheel_mode())
        self._base_moving = False

    def zero_base(self) -> None:
        self._base_target = None
        self._base_push(lambda base: base.enable_hold_mode())
        self._base_moving = False
        time.sleep(0.2)  # let odometry settle after the operator lets go
        self._base_ref = self._odom_at(time.monotonic())

    def disconnect(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        for device in self._devices.values():
            try:
                device.close()
            except Exception as e:
                log.error("closing the %s camera failed: %s", device.device, e)
        self._devices = {}
        with self._cmd_lock:
            for client in (self._cmd, self._status):
                if client is not None:
                    try:
                        client.stop()
                    except Exception as e:
                        log.error("closing a robot client failed: %s", e)
            self._cmd = self._status = None

    # -- internals ----------------------------------------------------------

    def _timeline(self, camera: str) -> Timeline:
        device, eye = CAMERAS[camera]
        return self._devices[device].eyes[eye]

    def _next_sync_frame(self, camera: str) -> tuple[float, dict[str, Any]]:
        deadline = time.monotonic() + 1.0
        while True:
            latest = self._timeline(camera).latest()
            if latest is not None and latest[0] != self._last_sync_t:
                self._last_sync_t = latest[0]
                return latest[0], {camera: latest[1]}
            if time.monotonic() > deadline:
                raise EmbodimentFault(f"no new frame from camera {camera!r} for 1s")
            time.sleep(0.002)

    def _match(self, camera: str, t_ref: float) -> Any | None:
        """The frame nearest t_ref, after waiting for one captured at or after t_ref - skew."""
        timeline, deadline = self._timeline(camera), time.monotonic() + MATCH_TIMEOUT_S
        while time.monotonic() < deadline:
            latest = timeline.latest()
            if latest is not None and latest[0] >= t_ref - MAX_SKEW_S:
                break
            time.sleep(0.001)
        nearest = timeline.nearest(t_ref)
        if nearest is None or abs(nearest[0] - t_ref) > MAX_SKEW_S:
            return None
        return nearest[1]

    def _all_joints_at(self, t: float) -> npt.NDArray[np.float64]:
        """The arm's joints, then the configured base joints, at time `t`."""
        arm = self._joints_at(t)
        base_joints = self._cfg.joints[len(JOINTS) :]
        if not base_joints:
            return arm
        return np.concatenate([arm, [self._heading_at(t)]])

    def _odom_at(self, t: float) -> float:
        value = self._odom.interpolate(t)
        if value is None:
            raise EmbodimentFault("no base odometry from stretch_body_server")
        return float(value)

    def _heading_at(self, t: float) -> float:
        """base_theta at time `t`: the heading since the last zero_base()."""
        if self._base_ref is None:
            raise EmbodimentFault("no base odometry from stretch_body_server")
        return self._odom_at(t) - self._base_ref

    def _joints_at(self, t: float, timeout_s: float = 1.0) -> npt.NDArray[np.float64]:
        deadline = time.monotonic() + timeout_s
        while True:
            values = [self._joints[j].interpolate(t) for j in JOINTS]
            if all(v is not None for v in values):
                return np.array(values, dtype=np.float64)
            if time.monotonic() > deadline:
                missing = [j for j, v in zip(JOINTS, values, strict=True) if v is None]
                raise EmbodimentFault(f"no joint state for {missing}")
            time.sleep(0.005)

    def _send(self, pose: Mapping[str, float], limits: Mapping[str, float]) -> None:
        """Check the link, then queue and push one target."""
        self._watch.check()
        with self._cmd_lock:
            self._queue(pose, limits)
            self._push()
        self._watch.pushed()

    def _queue(self, pose: Mapping[str, float], limits: Mapping[str, float]) -> None:
        r = self._cmd
        r.lift.move_to(float(pose["lift"]), v_m=limits["v_m"], a_m=limits["a_m"])
        r.arm.move_to(float(pose["arm"]), v_m=limits["v_m"], a_m=limits["a_m"])
        for j in WRIST_JOINTS:
            r.end_of_arm.move_to(j, float(pose[j]), v_r=limits["v_r"], a_r=limits["a_r"])
        # The installed gripper's own aperture range is the last word, whatever joint_high says.
        lo, hi = self._grip_meta.aperture_range
        aperture_m = min(max(float(pose["grip_mm"]) / 1000.0, lo), hi)
        command = self._grip_meta.aperture_to_command(aperture_m)
        r.end_of_arm.move_to(self._gripper, command, v_r=limits["v_r"], a_r=limits["a_r"])

    def _push(self) -> None:
        try:
            self._cmd.push_command(
                ignore_control_lock=self._cfg.ignore_control_lock, priority=self._cfg.priority
            )
        except SystemExit as e:
            # stretch4_body exits the process when another program holds the control lock.
            raise SafetyAbort(
                "another program is controlling Stretch; stop it, or run "
                "`stretch_body_server --free_up_control`"
            ) from e
