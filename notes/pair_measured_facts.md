# Robot and setup facts

Measured facts about two Stretch 4 robots with the parallel gripper (PG4).

## Setup

- robot 1 and robot 2 stand side by side on the same side of one table (74.9 cm high,
  60.3 cm deep), each turned in toward the table's centre. Seen by the operator, who faces
  the robots across the table, robot 1 is on the left and robot 2 on the right; left and
  right in an instruction are the operator's.
- Each robot's cameras are `robotN_head` (fixed on the head) and `robotN_wrist` (on the
  wrist).
- Nothing stops the two arms from hitting each other. While the grippers are near each other,
  check the gap between them in a head camera before each move that brings them closer, and
  close it slowly.

## Arm and gripper

- `lift` sets the wrist's height (0–1.2 m) and `arm` its extension along the arm's line
  (0–0.55 m). The arm droops slightly as it extends: at full extension the gripper sits
  about 2 cm lower than at `arm` 0.3 m for the same `lift`.
- `wrist_yaw` swings the gripper sideways; positive is toward the robot's left, and at +1.57
  it points straight to the robot's left. Each robot's range is in its joint limits.
- `wrist_pitch` 0 is level and ≈ 1.45 points the gripper straight down. Pointing down pulls
  the fingertips back toward the robot by about the gripper's length.
- `wrist_roll` turns the fingers' closing line. At `wrist_roll` 0 the wrist camera sits on
  top and the fingers below it; rolling swings the camera, with the gripper's body, round
  the fingers.
- The fingers stay parallel as they close. Fully open is 77 mm.
- On the table, an object tall and thick enough for the fingers to close on it clear of the
  table can be grasped from any direction but below; a small or flat one only from above. An
  object held up off the table can be grasped from any direction, including level or from
  below.
- Commanding `grip_mm` 0 grips as hard as the gripper can; the gripper stops short of
  breaking the object. The finger pads compress, so a held object reads below its width, a
  thin one far below (a 1–2 cm thick object held firmly can read 2–5 mm). Closed on nothing it
  reads about 0.1–0.3 mm.

## Table

- With the gripper level (`wrist_pitch` 0), the gripper reaches the table at `lift` ≈ 0.62 m
  with `arm` around 0.3 m, and at ≈ 0.64 m with the arm fully out.
- Pointing straight down (`wrist_pitch` ≈ 1.45), the fingertips touch the table at
  `lift` ≈ 0.77 m (arm fully out).

## Base

- With `robotN_base_theta` in the action, the base turns in place (positive is
  counter-clockwise seen from above); it never drives forward or sideways. It starts every
  trial at 0 and may turn up to ±3.14 rad from there. It turns only while that robot's arm
  is at 0.15 m or less; otherwise it holds. Without `robotN_base_theta`, it never moves.
- Turning swings the arm and its line about the base's centre. The head camera turns with
  the base. The base turns at up to 0.3 rad/s.

## Cameras

- Head images are 640 × 368 px, wrist images 640 × 400 px (width × height).
- The head camera's view is oblique, not a plan view. Raising `lift` moves the gripper
  image-up.
- The wrist camera sits on the gripper a little above and behind the fingers (on top at `wrist_roll` 0)
  and looks out over them along their length, so the fingertips are well out in front of the
  camera. In the wrist image the two fingertips are along the bottom edge, either side of the
  centre; a point further out along the fingers' line appears closer to the image centre. An
  object that appears between the fingertips is still a few cm beyond them, out past the
  tips. The fingertips stay fixed in the image while everything beyond them spreads outward
  from the centre as the gripper moves closer. The wrist camera shows only what is ahead of
  its own fingers.

## Controller

- Each move is interpolated at a fixed speed: the arm and lift travel ~8 cm/s and the wrist
  ~0.45 rad/s. Steps run at 30 Hz.
- A joint has arrived when its readback is within 6 mm (`lift`, `arm`), 0.03 rad (wrist,
  base 0.02 rad) or 3 mm (`grip_mm`) of the target.
- `joint_pos` is what the robot actually did. Targets beyond the joint limits are clamped to
  them; a joint that stops short without reaching a limit is in contact or blocked by
  self-collision checks.
