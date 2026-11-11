# inspect-robots-stretch4

[Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment adapter for the
Hello Robot **Stretch 4**. Once installed, any compatible policy (an LLM agent, a VLA) can be
evaluated on Stretch 4 with `--embodiment stretch4`, with Inspect Robots' logs, scoring and
Rerun visualization. It does not use ROS: it talks to `stretch_body_server` directly through
`stretch4_body`.

> [!WARNING]
> **Research preview.** This software is under active development and has not been fully
> validated. Interfaces, defaults and behavior may change without notice, and some features,
> notably the two-robot setup, are experimental. No benchmarking has been done yet; see
> [Status and known issues](#status-and-known-issues). Always supervise the robot and keep a
> hand on the run-stop.

## What the robot looks like to a policy

| | |
|---|---|
| Embodiment name | `stretch4` |
| Action | `joint_pos`, absolute targets: `lift` (m), `arm` (m), `wrist_yaw`, `wrist_pitch`, `wrist_roll` (rad), `grip_mm` (fingertip aperture, mm), then the base heading `base_theta` (rad), or no base with `-E base=none`. The base only turns in place |
| State | `joint_pos`: the same values, same order and units |
| Cameras | `head_right`, `gripper_right` (upright, longer side 640 px: head 640×368, the centre band of its fisheye; gripper 640×400, the whole frame; any of `head_left`, `head_right`, `gripper_left`, `gripper_right`) |
| Rate | 30 Hz, paced by the cameras (`self_paced`) |
| Base frame | `base_theta` is relative to the heading at the start of the trial, counter-clockwise seen from above |
| Not commanded | the head (it moves with the base) |

Safety, inside `step()` and independent of the framework's guardrails: each target is clamped
to the joint limits and to at most one `MAX_STEP` from the measured pose; joints move with
speed-limited `move_to`; the gripper is also clamped to its own aperture range; an engaged
run-stop, or another program holding the control lock, halts the eval. Episodes end when the operator
presses Esc (or Enter, outside the CLI); there is no automatic success check.

The base runs a velocity loop on wheel odometry toward its target, which is at most
0.2 rad from where it is; the loop drops a target it has not reached within 3 s, and the wheel
motors stop on their own ~1 s after the last command. At each reset the arm drives home, then
the wheels freewheel so you can push the robot onto its marks; pressing Enter holds the base
and makes that pose zero.

## Install on a new robot

Python ≥ 3.11 (Stretch 4 has 3.12). The robot-side dependencies, `stretch4_body` and OpenCV,
come with the Stretch 4 software install (`stretch4_install`).

**1. Get the code.**

```bash
mkdir -p ~/repos && cd ~/repos
git clone https://github.com/hello-robot/inspect-robots-stretch4.git
cd inspect-robots-stretch4
```

**2. Make an environment and install.** `--system-site-packages` lets the environment see the
robot's `stretch4_body`.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
pip install 'inspect-robots-agent==0.30.0'   # the LLM agent policy (--policy agent)
```

For two robots, see [Two robots](#two-robots-experimental).

**3. Add the model API key.** `inspect-robots` reads a `.env` file in the directory you run it
from; `.env` is git-ignored. Use the variable for your model's provider, such as
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`:

```bash
echo '<PROVIDER>_API_KEY=<your key>' > .env
```

**4. Start the robot.** Each time the robot boots:

```bash
stretch_robot_home                    # once per boot
stretch_body_server --launch          # leave it running in its own terminal
```

**5. Check it.** `doctor` moves nothing. The hold run moves nothing either, except that at the
start prompt the wheels freewheel: push the robot to check they roll, then press Enter to
lock them (`-E base=none` skips that).

```bash
cd ~/repos/inspect-robots-stretch4 && source .venv/bin/activate
inspect-robots doctor --embodiment stretch4
inspect-robots run --instruction "test" --policy stretch4-hold --embodiment stretch4 --max-steps 30
```

To update later: `git pull && pip install -e .` in the same environment.

## Run

```bash
# An LLM drives the arm through tool calls (needs inspect-robots-agent and the API key above).
# max_llm_calls and max_output_tokens bound the cost per episode.
inspect-robots run --instruction "pick up the cup" --policy agent \
    -P model=anthropic/claude-opus-5-5 -P wire=messages -P image_horizon=none \
    -P max_llm_calls=40 -P max_output_tokens=4000 --max-steps 1500 \
    --embodiment stretch4 -E cameras=head_left,head_right,gripper_right
```

Embodiment options (`-E key=value`):

| Key | Default | |
|---|---|---|
| `home_pose_file` | none | a pose file, `{"joints": {"lift": …, "grip_mm": …}}`, to drive to slowly at each reset |
| `auto_start` | `false` | skip the "press Enter to start" prompt |
| `cameras` | `head_right,gripper_right` | |
| `max_image_side` | `640` | Longer side of every image in pixels; the aspect is kept and frames are never enlarged |
| `control_hz` | `30` | the cameras' frame rate |
| `joint_low`, `joint_high` | see `config.py` | comma-separated, in action order |
| `gripper` | `parallel_gripper` | `parallel_gripper` (PG4, opens to 77 mm) or `stretch_gripper` (SG4, 150 mm) |
| `base` | `rotate` | `rotate` (the base turns in place, 7-D) or `none` (base held still, 6-D) |
| `base_max_theta` | `3.14` | how far the base may turn from its start heading in a trial |
| `gripper_depth` | `false` | also send the wrist stereo depth, aligned to `gripper_right` (needs that camera), as `gripper_right_depth`; the LLM agent shows it as a depth image with distances |
| `priority`, `ignore_control_lock` | `2`, `false` | `stretch_body_server` command arbitration |

### Recording a start pose for a task

1. Put the arm where the task should start, using the gamepad. Then run this, type `start` at
   the `label>` prompt, press Enter, and type `q`:

   ```bash
   stretch_inspect_record_poses --out poses/my-task --cameras head_left
   ```

2. Write the most recent `start` entry to the start pose file:

   ```bash
   python3 -c "
   import json
   rows = [json.loads(l) for l in open('poses/my-task/poses.jsonl')]
   pose = [r for r in rows if r['label'] == 'start'][-1]
   json.dump({'joints': pose['joints']}, open('poses/my-task/start.json', 'w'), indent=2)
   print(pose['joints'])"
   ```

3. Pass it to a run with `-E home_pose_file=poses/my-task/start.json`, and the arm drives
   there slowly at each reset.

For another task, replace `my-task` in each command, so each task gets its own
`poses/<task>/start.json`.

## Benchmarks

TBD. No benchmark results are included yet. The only registered task is the two-robot
`uncap-marker` ([Two robots](#two-robots-experimental)).

## Two robots (experimental)

> [!WARNING]
> Experimental. Nothing stops the two arms from hitting each other: stand within reach of both
> run-stops.

`--embodiment stretch4-pair` drives two Stretch 4s as one robot for the bimanual
`uncap-marker` task from [StationeryBench](https://openai.robocurve.org/stationerybench/). The action is each robot's six arm joints, plus
`robotN_base_theta` with `-E base=rotate` (12-D or 14-D). **Robot 1** runs the eval and drives
its own hardware; **robot 2** runs `stretch_inspect_pair_server`, reached over the network with
CURVE encryption, and only accepts robot 1's key.

### Set up each robot

On **both** robots, do steps 1, 2 and 4 of [Install on a new robot](#install-on-a-new-robot),
installing the pair extras instead (`dev` adds the hardware check):

```bash
pip install -e '.[pair,dev]'
pip install 'inspect-robots-agent==0.30.0'   # robot 1 only (runs the agent)
```

Then, on both:

1. Run the same `stretch4_body` version on both robots (`pip list | grep stretch4-body`).
2. Put both robots on the same network. Wi-Fi works well if the signal is good. Robot 1 also
   needs internet for the model's API. Over Wi-Fi, turn power saving off, because otherwise
   the Wi-Fi card sleeps between packets, and the latency spikes that causes make the robots'
   commands stutter: set `wifi.powersave = 2` in
   `/etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`, then
   `sudo systemctl restart NetworkManager`.
3. Run the hardware check, in person: `stretch_inspect_hardware_check` ([Hardware tests](#hardware-tests-on-the-robot)).

**Layout.** Stand the robots side by side at one edge of the table, each turned in toward its
centre, with robot 1 on the left as you face them (StationeryBench's layout; left and right in
its instructions are yours).

**Keys.** Each robot makes its own keypair once, and each trusts the other's public key. Quote
keys in single quotes (they can contain `!`, `>` or `^`):

```bash
stretch_inspect_pair_keys init                    # on each robot, once
stretch_inspect_pair_keys show                    # prints this robot's public key
stretch_inspect_pair_keys trust robot1 '<key>'    # on robot 2: robot 1's key
stretch_inspect_pair_keys trust robot2 '<key>'    # on robot 1: robot 2's key
```

**Replacing one robot:** set up the new one as above, then `stretch_inspect_pair_keys init` on it.
Trust the other robot's key there, and on the other robot re-run `stretch_inspect_pair_keys trust
robot1 '<new key>'` (or `robot2`), which replaces the old robot's key. Re-check the table
numbers in `notes/pair_measured_facts.md` on the new robot.

**The API key** goes in `.env` on robot 1 only (step 3 above).

**The agent's facts** are `notes/pair_measured_facts.md`, passed with
`-P 'prior_learnings="notes/pair_measured_facts.md"'`. They describe our setup, laid out as
StationeryBench lays it out: our table's height and depth, and where our gripper met it. Re-measure
and edit them for your own table and robots before you rely on them.

### Run

Each session, home both robots (`stretch_robot_home`) with their arms retracted, then:

```bash
# robot 2: leave running. robot 1 connects to ports 5600 and 5601.
stretch_inspect_pair_server --bind tcp://<robot 2's IP>:5600

# robot 1: a hold run moves nothing (check the link first)
inspect-robots run --task uncap-marker --policy stretch4-pair-hold --embodiment stretch4-pair \
    -E robot2_host=tcp://<robot 2's IP>:5600 -T max_steps=300 -T epochs=1

# robot 1: a trial
inspect-robots run --task uncap-marker --policy agent \
    -P model=anthropic/claude-opus-5-5 -P wire=messages -P effort=high -P image_horizon=none \
    -P max_llm_calls=30 -P max_output_tokens=16000 \
    -P 'prior_learnings="notes/pair_measured_facts.md"' -P transcript_echo=true \
    --embodiment stretch4-pair -E robot2_host=tcp://<robot 2's IP>:5600 -E base=rotate \
    -T epochs=1
```

Each task's setup prints at the grading prompt, with its rubric. For a left-handed robot 1,
first switch the robot itself to left-handed (run `REx_routines_run` and choose option 8), then
add `-E robot1_handedness=left`. To drive each robot to a start pose at every reset, record one
([Recording a start pose](#recording-a-start-pose-for-a-task)) and pass it with
`-E robot1_home_pose_file=…` and `-E robot2_home_pose_file=…`; a left-handed robot needs a
left-handed pose (wrist yaw, pitch, roll = π, π, π).

Pair options (`-E key=value`), besides `auto_start`, `max_image_side`, `control_hz` and
`priority` as for one robot:

| Key | Default | |
|---|---|---|
| `robot2_host` | none | `tcp://<robot 2's IP>:5600`, where `stretch_inspect_pair_server` listens (required) |
| `base` | `none` | `rotate`: each base turns in place (`robotN_base_theta`, 14-D) |
| `base_max_theta` | `3.14` | how far each base may turn from its start heading (rad, 180°) |
| `base_turn_max_arm` | `0.15` | a base turns only while its arm is retracted to this (m), now and in the target; otherwise it holds (step info `base_held`) |
| `cross_guard` | `false` | **Unfinished: not calibrated or tested on the robots; leave it off.** For robots facing each other across the table only: keeps the arms apart. Needs `rig_file`; refused with `base=rotate` or a left-handed wrist |
| `rig_file` | none | **Unfinished, for `cross_guard`.** The measured facing layout, from `stretch_inspect_pair_calibrate` |
| `gripper_depth` | `true` | each wrist's depth image (`robotN_wrist_depth`) |
| `robot1_cameras`, `robot2_cameras` | `head_right,gripper_right` | one eye per camera device |
| `robot1_home_pose_file`, `robot2_home_pose_file` | none | a pose each robot drives to at reset |
| `robotN_handedness` | `right` | `left` keeps that robot's wrist folded over (stretch4_body's left-handed home is yaw, pitch, roll = π, π, π); the model's joint values keep their right-handed meanings. Use a left-handed home pose with it |
| `robotN_wrist_yaw_min`/`_max`, `robotN_wrist_roll_min`/`_max` | the robot's limits | narrow that robot's `wrist_yaw` or `wrist_roll` range (rad, the robot's own joint values) |
| `image_mode` | `forward` | how robot 2 sends images: its camera JPEGs, or `lossless` PNG (half the step rate) |
| `link_retries`, `link_retry_s` | `3`, `5` | if robot 2 goes quiet, both robots hold and robot 1 retries this many times, this long each, before the run ends |

Safety, besides each robot's own: pressing either run-stop engages the other's within about
0.2 s, at any point in the run, including while the model is thinking between steps. A lost
link holds both robots instead of ending the run at once. **Test this on your robots before
running real trials:** during a hold run, press each robot's run-stop in turn and check that
the other robot stops too.

### Status and known issues

The two-robot setup has only been tried on `uncap-marker`, over a handful of trials. It has not
been benchmarked, and there are no reference scores. Expect failures. These are the most common:

- **Missed or slipping grasps.** The gripper closes on nothing, or holds an object by the
  fingertips and loses it when the other arm pulls. The finger pads compress, so a firm grasp
  can read only a few millimetres, and the model may decide it missed and let go.
- **The arms colliding.** Nothing stops the two arms from hitting each other.

Unfinished: the cross-robot collision guard (`cross_guard`, `rig_file` and
`stretch_inspect_pair_calibrate`) has never been calibrated or tested on the robots, so leave it
off.

## Develop

```bash
pip install -e ".[dev]"
pytest          # no robot needed: the hardware is behind a fake driver
ruff check . && ruff format --check .
```

> [!NOTE]
> **On the robot, a plain `pytest` fails** (for example `ModuleNotFoundError: No module named
> 'yaml'`). The ROS setup puts ROS's own pytest plugins (`launch_testing`, `ament_*`) on
> `PYTHONPATH`, and pytest tries to load them. This repo needs none of them, so turn them off:
>
> ```bash
> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
> ```
>
> `stretch_inspect_hardware_check` already does this for you.

## Hardware tests (on the robot)

> [!CAUTION]
> **Run these only in person, standing next to the robot with a hand on or near its run-stop.**
> They move the real robot: the arm moves up to 10 cm and the gripper opens and closes, and the
> tests engage and release the run-stop themselves. Never run them over SSH from somewhere else,
> or with nobody watching the robot.

`tests/hardware/test_robot.py` holds 8 tests that drive a real Stretch 4:

1. every camera's images have the documented sizes
2. the gripper reads open and closed correctly
3. a hold run keeps every joint still
4. the base holds still in `rotate` mode
5. pressing the run-stop during a run ends it, and the robot stays still
6. pressing the run-stop while driving to the start pose ends it, and the move doesn't resume
7. another program taking the command lease ends the run
8. `stretch_body_server` stopping mid-run is detected (only with `--server`)

A plain `pytest` skips them; `stretch_inspect_hardware_check` runs them.

Before you start, run `stretch_body_server`, home the robot (`stretch_robot_home`), release the
run-stop and clear the area around the arm. Then:

```bash
stretch_inspect_hardware_check              # all of them except the server stop (about 3 min)
stretch_inspect_hardware_check --server     # also stop and restart stretch_body_server.service
stretch_inspect_hardware_check -k runstop   # extra arguments go to pytest
```

The command needs this repo installed with `pip install -e ".[dev]"`. Some checks still need a
person:

- whether the images are upright and show the right scene
- which way the base moves in the room
