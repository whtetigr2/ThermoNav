# Compute Jitter in the Real World — What Actually Causes It and How Real Platforms Survive It

**Purpose:** Ground our simulation jitter tests (see `JITTER.md`) in reality. Paul's
question: "how much jitter are we talking here, and what real-world scenarios generate
that much jitter? How do mil-spec / hobbyist / competitive drones and Boston-Dynamics-style
bots deal with it?"

**Scope note:** This is web research, not firsthand measurement. Every number carries a
provenance label: **measured** (someone instrumented it), **company-claimed** (vendor says
so), **calculated** (arithmetic from labeled numbers), **estimated** (informed guess),
**inferred** (reasoned from evidence), or **MISSING** (could not verify — not invented).

**Claim label:** Z1-legal formulation, validated in simulation. This document is background
research; it makes no silicon or flight claims.

---

## 1. The universal pattern (read this first)

Every real platform we researched separates the stack into two clocks, and the split is
the whole trick:

- **Fast inner loop** (attitude / motor control): runs on a microcontroller under a real-time
  OS at hundreds of Hz to kHz. It *never* stalls — it is the thing everything else is allowed
  to fail around.
- **Slow outer loop** (perception / planning / navigation): runs on a bigger computer
  (companion board, Jetson, payload PC) at single-digit Hz. It *is* allowed to stall, because
  the inner loop keeps the vehicle stable while it recovers, and its outputs are
  freshness-checked before use.

Our stack already follows this pattern: classical PID + velocity servo + CBF at high rate
on the host, TSU planning at 1 Hz. The jitter tests freeze the *outer* loop (perception /
S2 / D5 see a stale map) while the inner loop keeps flying. That is exactly the right layer
to test — it matches how every real system is built.

The one collision in our tests (heavy fog + periodic 2 s freezes) happened because D5 — the
decider — consumed a stale map with no freshness bound. Section 6 covers how real platforms
put a staleness gate on exactly that path.

---

## 2. Hobbyist: Betaflight / ArduPilot / PX4

### 2a. Loop rates (inner vs outer)

| Layer | Betaflight | ArduPilot (Copter) | PX4 |
|---|---|---|---|
| Gyro sampling | 8 kHz (MPU6000, ICM-42688P) or 3.2 kHz (BMI270) — **company docs** | IMU read every loop — **open-source code** | IMU via sensor voting — **open-source code** |
| Inner PID / rate control | up to 8 kHz (F7/H7 boards); 4 kHz recommended on F4 — **company docs** | every main-loop iteration (rate PID from IMU only) — **open-source code** | ~1 kHz rate controller — **developer discussion** (PX4 GitHub issue #27514) |
| Main scheduler | n/a (bare-metal loop) | 400 Hz cooperative scheduler (AP_Scheduler) — **open-source code** | 100 Hz commander loop — **developer discussion** |
| Attitude/position outer | n/a (pilot's thumbs are the outer loop) | AHRS/EKF every iteration; nav updates 50 Hz; GPS 50 Hz; compass 10 Hz — **open-source code** | EKF2 estimator; position control above rate loop — **open-source code** |
| Housekeeping | n/a | 1 Hz loop; failsafe check 10 Hz — **open-source code** | commander state machine at 100 Hz |

Key numbers:
- Betaflight at 8 kHz = one PID correction every **125 µs** (**calculated**). Our 1 Hz TSU
  outer loop is 8,000× slower than an FPV inner loop — but that is *normal*: inner and outer
  loops live on different clocks everywhere.
- ArduPilot's `failsafe_check` runs at 10 Hz and detects a **2-second main-loop stall**,
  then reduces motors to minimum and disarms, repeating every second until recovery
  (**open-source code**, `ArduCopter/failsafe.cpp` via code audit). Two seconds of *inner-loop*
  stall is treated as a lost cause — the vehicle gives up on flying and disarms.
- Betaflight documents that **1% loop-time error puts the RPM notch filters 1% off their
  correct frequency** (**company docs**, 4.2 tuning notes) — even microsecond-scale jitter
  matters when filters are tuned to the loop clock. Their advice for lowest jitter: disable
  blackbox logging.

### 2b. Jitter-tolerance mechanisms

- **RTOS with priorities.** ArduPilot runs on ChibiOS, PX4 on NuttX — both preemptive
  real-time kernels. The flight-control thread cannot be starved by a logging thread.
  ArduPilot 4.7 release notes explicitly mention ChibiOS watchdog fixes for stuck SD-card
  threads (**company/open-source release notes**).
- **Cooperative scheduler with rate enforcement.** ArduPilot's AP_Scheduler runs fast tasks
  every iteration and rate-limits the rest (rc_loop 250 Hz, fence 25 Hz, ekf_check 10 Hz).
  A slow task delays the *next* loop, but the fast tasks (IMU, rate control, motor output)
  always run first each iteration.
- **Offboard heartbeat / freshness gate.** PX4 offboard mode requires a **2 Hz** setpoint
  stream as proof the external controller is alive (**PX4 docs**). If the stream stops for
  longer than `COM_OF_LOSS_T`, PX4 drops out of offboard and runs the failsafe chain.
  This is the direct ancestor of the "staleness gate" idea: *the decider refuses commands
  older than X*.
- **Failsafe chains, staged by severity.** PX4: Hold → Return → Land → **Descend**
  (open-loop descent, no position estimate needed — the bottom rung assumes the estimator
  itself is dead) (**PX4 docs**). ArduPilot: RTL → Land; throttle failsafe → disarm.
  ArduPlane: short failsafe action after **1.5 s** (FS_SHORT_TIMEOUT default), GCS failsafe
  after **20 s** (**parameter docs**).
- **Sensor voting and glitch rejection.** PX4 has centralized multi-IMU/GPS voting; ArduPilot
  runs `gpsglitch_check` and `ekf_check` at 10 Hz, with EKF yaw/position reset handling
  (**open-source code**). A bad sensor gets voted out rather than trusted.

### 2c. Real stall scenarios at this level

- A historical ArduPilot bug caused **1–2 seconds of RC control loss** when radio-frame timing
  aligned exactly with the loop frequency (**project news**, fixed long ago). Even the best
  hobbyist stacks have shipped multi-second control gaps — via timing bugs, not compute load.
- GPS lock loss of **1–2 seconds** is routine in hobbyist flight (forum reports, **anecdotal**);
  the position controller coasts on IMU and recovers. Longer outages (tens of seconds) trigger
  the failsafe chain.

---

## 3. Mil-spec / commercial autonomy: Skydio, Anduril-type, military UAS

### 3a. Loop rates and compute

- **Skydio 2** (company blog): the autonomy engine refines its flight plan **500 times per
  second** considering 40+ objectives (**company-claimed**). That is the *trajectory
  refinement* layer — 2 ms per pass (**calculated**).
- **Skydio edge/cloud split** (executive interview): on-drone edge processing handles vision,
  navigation, and immediate tracking at **7–10 Hz**; heavier planning and vision-language
  models run in the cloud with **1–2 s** feedback latency, accepted as the trade for
  "broad decisions about where to move" (**company-claimed** via interview). Note the shape:
  ~10 Hz local autonomy, ~0.5–1 Hz cloud deliberation — strikingly close to our 1 Hz TSU
  outer loop sitting above a fast classical inner loop.
- **Skydio X10** compute is Jetson-class (**third-party handbook**; exact SoC **MISSING** —
  marked UNVERIFIED in the source). No LiDAR by design; 6× global-shutter cameras + IMU.
- At 36 mph (~16 m/s), third-party analysis infers Skydio needs **<10 ms** control-loop
  decisions (**inferred** by a third party, not a company spec).

### 3b. Jitter-tolerance mechanisms

- **Lost-link procedures (the mil-spec answer to "what if the brain stalls").** Military UAS
  doctrine treats link/compute loss as *expected*, with preprogrammed responses: the RQ-7B
  Shadow flies to a lost-link holding point, waits for comms, then RTBs or lands via TALS
  (**US Army procedures**, via EFF archive). FAA lost-link procedure: squawk 7400, execute
  the preprogrammed flight profile (**FAA Order 7110.65**). EDA test guidelines **fail a
  platform after more than 3 short datalink losses** (**EDA MIL-UAS-SPECIFIC**).
- **Update-rate doctrine.** A UAS industry paper notes typical air-to-ground update rates of
  **1–20 Hz**: ~1 Hz is considered appropriate for a fully autonomous aircraft, ~20 Hz for a
  manually flown one (**industry paper** via NPSTC). Our 1 Hz planning rate sits exactly on
  the "fully autonomous" tick of that doctrine.
- **Hardware redundancy + voting.** DO-178C Level A (catastrophic failure) demands ~100× the
  reliability of Level B, achievable only through hardware redundancy, not better software
  (**industry analysis**). Standard answer: triple-modular redundancy with majority voting —
  a disagreeing flight computer is voted out and isolated instantly. Commercial example: the
  Veronte 4x autopilot runs multiple independent cores with automatic switchover, certified
  to DO-178C DAL B (DAL A in progress) (**company-claimed**).
- **Dissimilar redundancy.** To defeat common-mode software bugs, redundant channels are built
  by separate teams, sometimes in different languages (Ada vs C) or on different processors
  (**industry analysis**). A compute stall that hits one channel cannot hit all three the
  same way.
- **Fail-operational vs fail-safe.** Hardware redundancy keeps the mission going
  (fail-operational); software-only redundancy triggers failsafe protocols and ends the
  mission (fail-safe) (**industry analysis**). Our stack is currently fail-safe by design
  (D5 refusal → stand down), not fail-operational — worth naming honestly.
- **Real failure, honestly reported.** In a 2025 Navy exercise, 30+ Anduril drone boats
  running the Lattice autonomy stack rejected their inputs and auto-idled as a failsafe —
  "dead in the water" — when the software configuration was wrong (**Wall Street Journal**,
  Nov 2025). The failsafe *worked*; the mission didn't. That is the honest shape of
  fail-safe autonomy: the vehicle survives, the sortie is lost.

## 4. Competitive: FPV racing

### 4a. Loop rates

Same Betaflight inner loop as §2 (8 kHz gyro/PID on modern boards). The differentiator is
the *video* path: DJI's digital FPV system claims **28 ms end-to-end latency**
(**company-claimed**), marketed as matching analog. Racers obsess over single-digit
millisecond differences in camera-to-photon delay.

### 4b. How they handle jitter: they don't — they minimize it

Racing has no jitter *tolerance* strategy because the budget is the whole game. At racing
speeds (20+ m/s), a 28 ms video delay already means ~0.5 m of travel per frame
(**calculated**); a 2-second stall is not a degraded mode, it is a crash into a gate.
Mechanisms are all about *removing* latency: 8 kHz loops, DShot600, RPM filtering,
disabling blackbox logging, ELRS/Crossfire at 250–500 Hz RC update. The lesson for us is
negative: nobody in racing survives multi-second perception stalls, because nobody tries —
the inner loop is the entire system and there is no outer loop to stall.

## 5. Legged robots: Boston Dynamics Spot (and Atlas by extension)

### 5a. Loop rates

- Spot's **internal** locomotion controller rate is proprietary — **MISSING** (not published
  in the SDK docs). What *is* documented: the payload API communicates over gRPC, and an
  independent RL-on-Spot study **measured ~5 ms** of communication delay sending a command
  and receiving confirmation (**measured**, arXiv 2025).
- Third-party navigation stacks on Spot (e.g., mine-navigation research) run LiDAR-inertial
  odometry at **10 Hz LiDAR / 100 Hz IMU** through FAST-LIO2 on an Intel NUC payload PC
  (**measured** in the paper) — again the slow-outer/fast-inner split, with the fast part
  inside BD's closed controller.
- Spot exposes a **1 Hz** pulse-per-second (PPS) hardware sync pin so payload computers can
  time-align with the robot's internal clock (**BD SDK docs**).
- Spot streams ~200 telemetry channels "multiple times a second" (**company statement** via
  press test-drive).

### 5b. Jitter-tolerance mechanisms

- **The locomotion controller is a sealed box.** BD never lets payload code near the balance
  loop. Your autonomy commands are *suggestions* (velocity setpoints, body poses) streamed
  over the API; the internal controller guarantees it won't fall over. This is the most
  extreme version of the inner/outer split: the inner loop is not just fast, it's
  untouchable.
- **Safety interlock.** The payload electrical interface has a safety circuit — break
  continuity and the robot *immediately* disables motor power and sits down (**BD SDK
  docs**). Hardware-level dead-man's switch, no software in the path.
- **Comms-loss behavior.** Third-party fleet operators (Formant) report: when connectivity
  drops below a usable level, they send "Sit" and the robot powers down; video is throttled
  block-by-block as bandwidth degrades (**company statement** via press). Degrade gracefully,
  then stop — the legged equivalent of our D5 refusal.
- **Latency-aware state estimation (research).** The standard technique for fusing slow
  sensors (vision, LiDAR) with fast IMU is *delay compensation*: buffer the inertial data,
  and when a delayed visual measurement arrives, fuse it at its *original timestamp* and
  re-propagate forward (demonstrated for UAVs with delayed visual measurements, **peer-
  reviewed**, MDPI Sensors). Teleoperation research distinguishes nominal delays (25–100 ms)
  from delay *outliers* and compensates the outliers inside a predictor framework
  (**peer-reviewed**, MDPI). Our stack does not currently do timestamp-based
  re-propagation — the map simply freezes. That is a known, named gap.

## 6. Real-world scenarios that generate multi-second stalls (with magnitudes)

| Scenario | Realistic magnitude | Provenance | Our closest test regime |
|---|---|---|---|
| Python garbage-collection pause | **up to 2,200 ms measured** on Linux (Intel Ice Lake); 320 ms on Apple M4 | **measured** (lemire.me, Feb 2026) — directly relevant: our sim stack is Python on Linux | Isolated 2 s delay |
| Thermal throttling (Jetson-class, 25 W in a sealed enclosure) | sustained clock reduction; ms-scale numbers **MISSING** | **industry reporting** (throttling widely documented; no published ms figures found) | Periodic every 3rd/5th step (sustained degraded) |
| GPU contention (two models, shared memory bandwidth) | missed deadlines at 24 Hz on Orin Nano Super; "latency still spikes" even with cgroup pinning | **third-party analysis** (dev.to survey) | Periodic every 3rd step |
| RC link micro-loss | 400 ms default stage-1 failsafe (Betaflight); 1–2 s in a historical ArduPilot timing bug | **company docs** / **project history** | Isolated 0.5–1 s delay |
| GPS fault (real flight test) | **88 s** of erroneous output (t=44–132 s); fault-tolerant filter survived, naive filter diverged permanently | **measured** (GPS World flight test) | Beyond anything we tested — and survived *because* of fault detection, not raw tolerance |
| Sensor driver crash / disconnect (LiDAR, camera) | indefinite until restart; treated as sensor loss, not degraded mode | **industry practice** | Perception-died (every 2nd step) |
| GCS link loss | 20 s timeout before RTL (ArduPlane) | **parameter docs** | Longer than any regime we tested |
| Offboard companion-computer death | setpoint stream stops → PX4 failsafe after COM_OF_LOSS_T | **PX4 docs** | Isolated 5 s delay |
| Full compute hang (watchdog reboot) | seconds to reboot + re-init; ArduPilot treats 2 s inner-loop stall as disarm-worthy | **open-source code** | Isolated 5 s delay |

The honest summary: **0.5–2 s stalls are routine, real, and survivable** — GC pauses,
thermal dips, link micro-losses. Every platform above assumes they happen and is built
around them. **5 s is past the edge**: ArduPilot disarms at 2 s of inner-loop stall; our
5 s test surviving in clear air is a *stronger* result than the hobbyist envelope, and it
holds only because our inner loop never stalls.

## 7. Mapping table: our tested regimes → real-world equivalents → who survives

| Our regime (JITTER.md) | Closest real-world equivalent | Hobbyist (Betaflight/ArduPilot/PX4) | Mil-spec / Skydio-type | FPV racing | Spot-type legged |
|---|---|---|---|---|---|
| Isolated 0.5 s, clear | Python GC pause; RC micro-failsafe | **Survives.** Inner loop (400 Hz–8 kHz) never notices; EKF coasts on IMU. | **Survives.** 1 Hz autonomy doctrine already assumes this. | **Crashes.** 0.5 s at 20 m/s = 10 m blind. No tolerance strategy exists. | **Survives.** Sealed balance controller; autonomy just resumes. |
| Isolated 1–2 s, clear | Worst-case GC pause (2.2 s measured); thermal dip | **Survives, then failsafes.** 2 s is ArduPilot's loop-stall disarm threshold — at the *inner* loop. Outer-loop (companion) stall → offboard timeout → RTL. | **Survives.** Lost-link hold pattern is built for minutes, not seconds. | **Crashes.** | **Survives.** |
| Isolated 5 s, clear | Full compute hang; watchdog reboot territory | **Marginal.** Inner loop would have disarmed at 2 s; only survives if the stall is *outer-loop-only* (companion computer) while the FC flies on. PX4 offboard → failsafe chain. | **Survives** via preprogrammed lost-link profile. | **Crashes.** | **Survives** (sits down if it's the autonomy PC; balance never depended on it). |
| 2 s burst ×3, clear | Repeated GC pauses under sustained load | **Survives** (same as isolated, if outer-loop-only). | **Survives.** | **Crashes.** | **Survives.** |
| 2 s periodic every 3rd/5th, clear | Thermal-limited sustained operation; periodic background task stealing cycles | **Survives** if inner loop is isolated by RTOS priority. This is exactly what ChibiOS/NuttX priorities are for. | **Survives.** | **Crashes.** | **Survives.** |
| 2 s periodic every 3rd, **heavy fog** (our 1 collision) | Degraded sensing **and** stale compute together — e.g., fog + thermal throttle on the same board | **Would failsafe, not collide.** PX4's offboard freshness gate (2 Hz heartbeat) drops stale setpoints; ArduPilot's EKF rejects glitchy sensors at 10 Hz. No hobbyist stack feeds a 2-second-stale obstacle map to its collision checker. | **Would stand down.** Lost-link/sensor-degraded doctrine says land, not push through. | **Crashes** (long before fog matters). | **Would sit.** Degraded perception → stop, the Formant pattern. |
| Perception-died (every 2nd step), clear — 6/14, fails safe via max_steps | Sensor hardware failure (LiDAR disconnect, camera driver crash) | **Failsafes.** Treated as sensor loss → RTL/Land, not as "fly half-blind." Our max_steps timeout is the sim analogue of their RTL. | **Failsafes** per doctrine. | n/a (no such sensor path). | **Sits down.** |

## 8. The architectural lesson (what we'd steal)

1. **Freshness-gate the decider.** PX4's 2 Hz offboard heartbeat is the pattern: D5 should
   refuse to act on a map older than *N* seconds, full stop — not "act on it but
   cautiously." Our collision happened at the one place with no staleness bound. This is
   now a blocking test cell (bar 2b) in the evidence-θ round-2 validation, per Paul's call
   2026-10-08.
2. **Keep the inner loop untouchable.** Betaflight disables blackbox logging to protect
   loop timing; ArduPilot puts IMU+rate-control+motor-output first every scheduler
   iteration. Our CBF + velocity servo + PID already sit at this layer — the jitter tests
   confirm they carry the vehicle through 5 s outer-loop stalls in clear air. Don't let
   future features (logging, telemetry, the TSU itself) share that thread.
3. **Fail-safe is honest; fail-operational costs hardware.** Everything we have is
   fail-safe (D5 refusal → stand down). Mil-spec fail-*operational* needs voted redundant
   computers. Name which one we are in every claim; don't imply the other.
4. **Delay compensation is a named gap.** Research systems fuse delayed measurements at
   their original timestamps and re-propagate. We freeze the map instead. Fine for now —
   but it's on the list, not forgotten.

## 9. What we could not verify (MISSING, not invented)

- Spot's **internal** locomotion control rate — **MISSING** (proprietary; not published).
- Thermal-throttling latency magnitudes in milliseconds on Jetson-class hardware —
  **MISSING** (everyone agrees throttling happens; nobody publishes the ms numbers).
- Anduril Lattice / Skydio latency budgets and stall-handling internals — **MISSING**
  (not public).
- PX4 `COM_OF_LOSS_T` default value — **MISSING** (didn't verify; it's a tunable parameter).
- Whether Skydio's 7–10 Hz edge figure and 500 Hz refinement figure describe the same
  pipeline stage — **MISSING** (different sources, different years; treated as separate
  layers in §3a).

## 10. Sources

- Betaflight docs: gyro/PID loop configuration (betaflight.com), 4.2 tuning notes on loop
  jitter (betaflight.com)
- ArduPilot: AP_Scheduler and failsafe.cpp via third-party code audits (jeranaias/
  meridian-core, mrandreyisachenko/architecture-radar); FS_SHORT_TIMEOUT parameter docs
  (ardupilot.org); historical RC-loss bug (ardupilot project news)
- PX4: offboard mode docs and failsafe chain (px4.io docs); commander/rate-controller rates
  (PX4-Autopilot GitHub issue #27514)
- Skydio: "Inside the mind of the Skydio 2" (Skydio blog, Medium); Suchet Bargoti interview
  on edge/cloud split (BigGo Finance); third-party spatial-intelligence handbook (sou350121,
  GitHub — flagged UNVERIFIED where it flags itself)
- DJI digital FPV latency claim (dji.com newsroom)
- Boston Dynamics Spot: SDK electrical interface docs (dev.bostondynamics.com); gRPC delay
  measurement (arXiv RL-on-Spot paper, 2025); Formant fleet operations (TechBrew test-drive)
- Military UAS: RQ-7B adverse procedures (US Army, via EFF); EDA MIL-UAS-SPECIFIC
  guidelines; FAA Order 7110.65 lost-link procedures; UAS update-rate doctrine (NPSTC paper)
- DO-178C/redundancy: vectree.io FBW analysis; Veronte 4x (unmannedsystemstechnology.com);
  Embention KAI flight computer (embention.com)
- Python GC pauses: lemire.me measurements (Feb 2026)
- GPS fault flight test: GPS World, "Robustness to Faults for a UAV"
- Delayed-measurement fusion: MDPI Sensors (visual-inertial delay compensation); MDPI
  teleoperation delay-outlier paper
- Jetson thermal/Orin Nano: TechPowerUp, Frore Systems AirJet announcements; multi-model
  resource survey (dev.to)

*Written 2026-10-08. Web research only — no flight hardware was harmed or measured.*