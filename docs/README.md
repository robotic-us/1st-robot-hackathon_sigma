# SIGMA — 제1회 로봇 해커톤 2026

로보틱어스(Roboticus) 주최 제1회 로봇 해커톤(2026. 8. 5.~8. 8., KAIST) 참가팀
**SIGMA**(서울대) 저장소입니다.

- 팀원: 백시은 · 김정환 · 이진명
- 대회: https://robotic-us.com

사람의 **표정과 목소리**를 보고 스스로 반응하는 돌봄 로봇입니다.

```
webcam ─┬─► YuNet → FER+ → 감정 확률 5종 ─┐
        │                                  ├─► distress 0..1 ─► 모션 슬롯 ─► play()
        └─► mic → RMS + 음성대역 비율 ──────┘
```

한 사람이 어디에 있는지가 **방향**(left/center/right)을, 얼마나 힘든 상태인지가
**크기**(small/medium/large)를 정하고, 그 두 좌표가 `slots.json`의 3×3 그리드에서
재생할 모션 하나를 고릅니다.

## 빠른 시작

```bash
python3 fetch_models.py                  # ONNX 모델 3종 내려받기 (최초 1회)
python3 listen.py                        # 마이크 확인 — 소리 미터
python3 sense.py                         # 카메라 확인 — 얼굴·감정·distress 창
python3 care.py --mock --no-window       # 전체 루프 (로봇 없이)
```

로봇/시뮬레이터와 함께:

```bash
python3 make_motions.py                  # slots.json → motions/motion_NN.csv
ros2 launch agx_bringup motion.launch.py motion_dir:=$PWD/motions   # 터미널 1
phorce list --target sim:demo            # 10개 슬롯이 보여야 정상
python3 care.py --target sim:demo        # 시뮬레이터로 재생
```

## 파일 지도

**메인 경로** — 이 5개만 읽으면 제품 전체입니다.

| 파일 | 역할 |
|---|---|
| `care.py` | ★ 진입점. 카메라 → 판단 → `play()` 루프 하나 |
| `sense.py` | 프레임 + 소리 → `Reading(x, distress, emotion, name)` |
| `listen.py` | 웹캠 내장 마이크. `arecord` 기반, 추가 설치 없음 |
| `sigma/` | 얼굴 검출(YuNet) · 인식(SFace) · 감정(FER+) |
| `slot_table.py` | `slots.json` 읽기·검증 |

**로봇 계층**

| 파일 | 역할 |
|---|---|
| `phorce_iface.py` | phorce/ROS 2를 아는 유일한 파일. `MockRobot` + `PhorceRobot` |
| `pvector.py` | P-Vector 파싱 + 5차 다항식 월드 모델 |
| `dream.py` | DREAM-Chunk — 후보 순위(`ChunkMatcher`), 이탈 감시(`DreamMonitor`) |

**도구**

| 파일 | 역할 |
|---|---|
| `make_motions.py` | `slots.json` → `motions/motion_NN.csv` (pcm·시뮬레이터가 읽는 포맷) |
| `enroll.py` | 얼굴 등록 → `faces/faces.npz` |
| `fetch_models.py` | ONNX 모델 내려받기 |

**`prototype/`** — 초기 프로토타입(색 원 자극). 지금 경로에서는 쓰지 않지만
DREAM-Chunk 전체 루프가 유일하게 다 도는 곳이라 회귀 테스트용으로 남겨 뒀습니다.

```bash
python3 prototype/stimulus.py --scenario demo --record prototype/demo.mp4
python3 prototype/main.py --mock --video prototype/demo.mp4 --once --dream
```

**`docs/`** — 대회 제공 자료(`RH_Guide*`)와 우리 설계 문서.

| 문서 | 내용 |
|---|---|
| [docs/dream-chunk.md](docs/dream-chunk.md) | DREAM-Chunk 설계와 **테스트 절차 5단계** |
| [docs/face-recognition.md](docs/face-recognition.md) | 얼굴 인식 + 5분류 감정 파이프라인 상세 |
| [docs/hackathon.md](docs/hackathon.md) | 대회 정보 · 지적재산권 |
| `docs/RH_Guide/` | 논문 3종 · OT 자료 · P-Vector · phact · 배선 |
| `docs/RH_Guide_Jetson-SDK/` | phorce SDK 공식 문서 5종 (**이쪽이 최신**) |
| `docs/RH_Guide_Angel/` | 구버전 SDK 문서 + Studio·pcm 매뉴얼 |

> `RH_Guide_Angel`과 `RH_Guide_Jetson-SDK`는 세대가 다르고 서로 어긋납니다.
> **`RH_Guide_Jetson-SDK` 쪽을 따르세요.** (구버전에만 있는 `robot.watch()`,
> `status.ethercat_operational`은 실제 SDK에 존재하지 않습니다.)

## 셀프테스트

전부 로봇 없이, ROS 없이 돕니다.

```bash
python3 listen.py --selftest     # 소리 특징 (톤/노이즈/무음)
python3 sense.py --selftest      # 감정→distress 가중, noisy-OR 융합
python3 pvector.py --selftest    # 5차 다항식 경계조건, CSV 파싱
python3 dream.py --selftest      # 매처 순위, 이탈 감지
python3 prototype/perception.py --selftest
```

## 알려진 제약

- **모션 슬롯이 아직 실물 로봇에 없습니다.** `phorce list`가 비어 있으면
  phorce Studio에서 교시하거나(①설정 영점 → ②교시), `make_motions.py`로 만든
  파일을 시뮬레이터에 물려 쓰세요.
- `slots.json`의 자세는 전부 `"placeholder": true`입니다. 교시 후 실제 각도로
  교체해야 `start_pose` 매칭이 의미를 갖습니다.
- 시뮬레이터는 **모션 계약만** 흉내 냅니다. `/phorce/feedback`(1 kHz)은 나오지
  않으므로 DREAM-Chunk의 이탈 감시는 sim에서 동작하지 않습니다 —
  `--mock` 또는 실물에서 확인하세요.
- `pvector.UNITS_PER_DEG`는 실제 피드백으로 보정이 필요합니다.

## 라이선스

MIT. [LICENSE](LICENSE) 참고. 본 프로젝트의 지적재산권은 SIGMA 팀 전원에게 있으며,
주최 측은 아카이브·홍보 목적으로만 활용합니다.
