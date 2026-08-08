# docs/ — design documents and competition material

Project overview and quick start live in the [root README](../README.md).
This directory holds the team's design documents and the material provided by
the competition.

## Our documents

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The whole design: live loop, safety ladder, motion library, DREAM-Chunk, and the report ↔ code traceability table |
| [DEMO.md](DEMO.md) | The tested command lines for the demo — one sentence each, verified on the Jetson, plus troubleshooting |
| [VERIFICATION.md](VERIFICATION.md) | How the cradle is verified without an infant: the layered protocol, and exactly what the mascot loopback does and does not prove |
| [DREAM-CHUNK.md](DREAM-CHUNK.md) | The DREAM-Chunk planner: why this paper, the estimator audit, and every measurement (kept and removed halves alike) |
| [IDEA.md](IDEA.md) | The personalisation idea the policy layer implements |

## The spec

| File | Role |
|---|---|
| `infant_robotic_cradle_evidence_report_ko.pdf` | **The motion spec** (Korean): the M01–M50 library, the kinematic envelope, and the §5 safety state machine. When motion behaviour is in question, it wins |
| `robotic_cradle_trajectory_parameter_guide_ko.pdf` | Trajectory parameter guide (Korean) |
| `motion-system.png` | The team's N01–N34 motion system: shapes × small-fast/large-slow × vibe × decay |
| `urdf_assembly.stl` | The rig's CAD assembly — `tools/make_urdf.py` regenerates `cad/sigma.urdf` from it |
| `Nubzuki.jpg` · `poses/` | The mascot reference sheet and the 17 rig renderings the vision system is graded against |

## Competition material

| Directory | Contents |
|---|---|
| `RH_Guide/` | Three papers, orientation material, P-Vector, phact, wiring |
| `RH_Guide_Jetson-SDK/` | The five official phorce SDK documents — **this generation is current** |
| `RH_Guide_Angel/` | Older-generation SDK docs + Studio/pcm manuals |

> `RH_Guide_Angel` and `RH_Guide_Jetson-SDK` are different generations and
> contradict each other. **Follow `RH_Guide_Jetson-SDK`.** (`robot.watch()`
> and `status.ethercat_operational`, which only the old docs mention, do not
> exist in the real SDK.)

## Historical note

The infant-vision stack (five-state watcher, face/emotion models, microphone
fusion, `serve.py --sense` in its original meaning) was removed on 2026-08-08;
`perception/nubzuki.py` — which reads the cradle *iPad*, never an infant — is
the only vision left, and `--sense` now names that camera path. The removal
and what it took with it are recorded in [VERIFICATION.md](VERIFICATION.md) and in git
history (`docs/face-recognition.md` described the deleted face pipeline).
