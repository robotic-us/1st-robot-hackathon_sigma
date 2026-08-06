# SIGMA — 제1회 로봇 해커톤 2026

로보틱어스(Roboticus) 주최 제1회 로봇 해커톤(2026. 8. 5.~8. 8., KAIST) 참가팀
**SIGMA**(서울대) 저장소입니다.

- 팀원: 백시은 · 김정환 · 이진명
- 대회: https://robotic-us.com

영아의 **상태를 보고 스스로 반응하는 로봇 요람**입니다. 모션의 근거와 안전
규칙은 `docs/infant_robotic_cradle_evidence_report_ko.pdf`(M01–M50 모션
라이브러리 + 안전 사다리)를 따르고, 전체 구조는
[docs/ARCHITECTURE.md](ARCHITECTURE.md)에 있습니다.

```
상태 카드(태그) 또는 얼굴+울음 ─► CradleMachine(안전 사다리) ─► MotionEngine(M01-M50)
                                                                │
                                              웹 대시보드 · RViz · phorce 시뮬레이터
```

## 빠른 시작

```bash
python3 tests.py                         # 전체 셀프테스트 (10개 스위트, ~30초)
python3 serve.py --fake                  # 카메라 없이: 합성 상태 카드 시나리오
python3 serve.py                         # 웹캠 + 태그 상태 카드 (tag_0/1/2)
python3 serve.py --sense                 # 실전: 얼굴 감정 + 마이크가 상태를 판정
python3 tools/fetch_models.py            # --sense용 ONNX 모델 3종 (최초 1회)
```

로봇/시뮬레이터와 함께:

```bash
python3 tools/make_motions.py --library  # M01-M50 → motions_m50/motion_NN.csv
./sim.sh                                 # phorce 시뮬레이터 (50개 슬롯)
phorce play 12 --target sim:demo         # M12 = ML 0.5 Hz A10 재생
./cad/view.sh                            # RViz (터미널 분리)
```

## 파일 지도

| 경로 | 역할 |
|---|---|
| `serve.py` | ★ 진입점. 센싱 → 상태기계 → 모션 → 웹 대시보드 한 프로세스 |
| `tests.py` | 유일한 테스트 명령. 모든 스위트가 여기 있음 |
| `core/cradle.py` | 근거 보고서의 M01–M50 라이브러리 · MotionEngine · CradleMachine |
| `core/dream.py` · `core/pvector.py` | DREAM-Chunk 매처/모니터 + P-Vector 월드 모델 |
| `core/slot_table.py` · `core/phorce_iface.py` | 슬롯 표 · phorce/ROS 2 어댑터 |
| `perception/tag.py` | AprilTag **상태 카드** (tag_0 평온 / tag_1 칭얼 / tag_2 울음) |
| `perception/sense.py` · `perception/listen.py` | 얼굴+소리 → distress 0..1 |
| `perception/face/` | 얼굴 검출(YuNet) · 인식(SFace) · 감정(FER+) |
| `apps/` | demo(DREAM 시연) · care(구 진입점) · run(얼굴 데모) · animate(RViz) |
| `tools/` | make_motions(--library) · make_urdf · fetch_models · enroll |
| `motions_m50/` | M01–M50을 pcm 슬롯으로 컴파일한 것 (`./sim.sh`가 기본 사용) |
| `web/` · `cad/` | 대시보드 · URDF/메시(RViz) |

**`docs/`** — 대회 제공 자료(`RH_Guide*`)와 우리 설계 문서.

| 문서 | 내용 |
|---|---|
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | 전체 구조 · 근거 보고서 ↔ 코드 대응표 |
| `docs/infant_robotic_cradle_evidence_report_ko.pdf` | **모션 명세의 원전** |
| [docs/dream-chunk.md](dream-chunk.md) | DREAM-Chunk 설계 |
| [docs/face-recognition.md](face-recognition.md) | 얼굴 인식 + 5분류 감정 파이프라인 |
| `docs/RH_Guide/` | 논문 3종 · OT 자료 · P-Vector · phact · 배선 |
| `docs/RH_Guide_Jetson-SDK/` | phorce SDK 공식 문서 5종 (**이쪽이 최신**) |
| `docs/RH_Guide_Angel/` | 구버전 SDK 문서 + Studio·pcm 매뉴얼 |

> `RH_Guide_Angel`과 `RH_Guide_Jetson-SDK`는 세대가 다르고 서로 어긋납니다.
> **`RH_Guide_Jetson-SDK` 쪽을 따르세요.** (구버전에만 있는 `robot.watch()`,
> `status.ethercat_operational`은 실제 SDK에 존재하지 않습니다.)

## 셀프테스트

전부 로봇 없이, 카메라 없이, ROS 없이 돕니다.

```bash
python3 tests.py             # 전체 (listen sense models tag pvector dream cradle m50 demo serve)
python3 tests.py cradle m50  # 골라서
python3 tests.py --list      # 목록
```

## 알려진 제약

- **모션 슬롯이 아직 실물 로봇에 없습니다.** `phorce list`가 비어 있으면
  phorce Studio에서 교시하거나, `tools/make_motions.py --library`로 만든
  파일을 시뮬레이터에 물려 쓰세요.
- `slots.json`의 자세는 전부 `"placeholder": true`입니다(DREAM 데모용 10슬롯).
- 시뮬레이터는 **모션 계약만** 흉내 냅니다. `/phorce/feedback`(1 kHz)은 나오지
  않으므로 DREAM-Chunk의 이탈 감시는 sim에서 동작하지 않습니다.
- `pvector.UNITS_PER_DEG`는 실제 피드백으로 보정이 필요합니다.
- 이 기구는 수평 1자유도입니다 — ML/AP 모션이 같은 축에 실리고, Z 모드는
  재생할 자유도가 없습니다.

## 라이선스

MIT. [LICENSE](../LICENSE) 참고. 본 프로젝트의 지적재산권은 SIGMA 팀 전원에게 있으며,
주최 측은 아카이브·홍보 목적으로만 활용합니다.
