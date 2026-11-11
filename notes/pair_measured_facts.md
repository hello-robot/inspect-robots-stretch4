# Setup facts

Measured facts about this setup: two Stretch 4 robots with the parallel gripper (PG4) at one
table. The facts about the robots themselves are in the embodiment's built-in notes.

## Setup

- robot 1 and robot 2 stand side by side on the same side of one table (74.9 cm high,
  60.3 cm deep), each turned in toward the table's centre. Seen by the operator, who faces
  the robots across the table, robot 1 is on the left and robot 2 on the right; left and
  right in an instruction are the operator's.
- Nothing stops the two arms from hitting each other. While the grippers are near each other,
  check the gap between them in a head camera before each move that brings them closer, and
  close it slowly.

## Table

- With the gripper level (`wrist_pitch` 0), the gripper reaches the table at `lift` ≈ 0.62 m
  with `arm` around 0.3 m, and at ≈ 0.64 m with the arm fully out.
- Pointing straight down (`wrist_pitch` ≈ 1.45), the fingertips touch the table at
  `lift` ≈ 0.77 m (arm fully out).
