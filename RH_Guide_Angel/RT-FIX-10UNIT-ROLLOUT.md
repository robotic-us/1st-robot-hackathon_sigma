# 10대 일괄 복구 & 퀵스타트 진입 런북 (2026-08-04)

> 골든 이미지 10대가 전부 **순정 커널 부팅** 상태라는 전제에서( [1호기 검수 리포트](GOLDEN-IMAGE-INSPECTION-20260804.md) D1),
> 마스터 젯슨에서 만든 **USB 복구 페이로드(`agr-rt-fix/fix-unit-rt.sh`)** 로 유닛을 하나씩 복구하고,
> 각 유닛이 **참가자 퀵스타트로 자연스럽게 이어지는 지점**까지를 한 장으로 정리한다.
>
> 역할 분담: USB 페이로드의 생성·내용은 **마스터 젯슨 세션** 산출물 (커널 복구 + deb 정리,
> 이미 RT 인 보드에서는 커널을 건너뜀). 이 문서는 그것을 **10대에 돌리는 절차와 판정 기준**이다.

## 0. 준비 — USB 복제 (여러 개 병렬 작업용)

- 복구 USB 는 **부팅 USB 가 아니라 그냥 파일**이다. 복제는 아무 PC 에서 `agr-rt-fix/` 폴더를
  다른 USB 로 **복사만 하면 끝** (FAT32/exFAT 무관, 실행권한 무관 — `sudo bash` 로 실행하므로).
- USB 를 2~3개 만들면 보드 여러 대를 병렬로 돌릴 수 있다. 다 만들 필요는 없다 —
  보드당 USB 점유 시간은 복구 실행 몇 분뿐이고, 재부팅·확인 중에는 다음 보드로 옮기면 된다.

## 1. 유닛당 절차 (보드 1대 기준 약 10~15분)

### ① 복구 실행

1. USB 를 유닛에 꽂고 `phorce` 로 로그인
   - 첫 로그인이면 초기 비밀번호 입력 → **새 비밀번호 강제 변경**(정상, 출하 밀봉 동작) → 유닛 라벨에 기록
2. 터미널:
   ```bash
   cd /media/phorce/*/agr-rt-fix     # 자동마운트 안 되면: sudo mkdir -p /mnt/usb && sudo mount /dev/sda1 /mnt/usb && cd /mnt/usb/agr-rt-fix
   sudo bash fix-unit-rt.sh
   ```
3. 끝에 `[OK]` 줄들 확인 → `sudo reboot` (USB 는 재부팅 들어가면 뽑아서 다음 보드로)

> **재부팅은 생략 불가** — 커널은 부팅 때 한 번만 올라가므로, 실행 중 교체가 원리적으로
> 불가능하다 (kexec 류 꼼수는 이 장비/절차에서 금지). 대신 **재부팅당 1회로 끝**이고,
> 부팅을 기다릴 필요 없이 그 시간에 USB 를 다음 보드로 옮겨 겹치기 작업하면
> 전체 소요는 "보드 수 × 부팅시간"이 아니라 거의 부팅 1~2회분으로 흡수된다.
>
> **부팅이 안 끝나는 것 같으면 — 5분 규칙**: 이 이미지들은 첫부팅/네트워크 서비스
> 타임아웃(90~120초) 때문에 2~4분 걸릴 수 있다. 화면 로그가 가끔이라도 넘어가면 대기.
> **같은 화면에서 5분 이상 무변화**면 전원 강제 리셋 OK (커널 파일 쓰기는 재부팅 전에
> 끝났으므로 파일 손상 위험 낮음). 단 **리셋 후에도 매번 같은 지점에서 멈추면**
> 부팅 실패로 취급 — 반복 리셋하지 말고 마지막 화면 내용을 들고 보고할 것.

### ② 재부팅 후 커널 판정 (30초)

```bash
uname -r                    # 5.15.148-rt-tegra 나와야 함
cat /sys/kernel/realtime    # 1 나와야 함
```

둘 중 하나라도 아니면 **중단하고 그 보드 격리** — §3 문제 해결 참조.

### ③ SDK 판정 — 시뮬레이터판 퀵스타트 (로봇 불필요, 3분)

여기서부터가 "퀵스타트로 이어지는" 구간이다. 로봇이 없는 유닛은 퀵스타트의 시뮬레이터
버전으로 판정한다 (참가자 문서의 `--target sim:demo` 경로와 동일):

```bash
ros2 launch agx_bringup motion.launch.py &        # 가짜 로봇(sim) 켜기
phorce doctor --target sim:demo                   # 창구 3종 OK / FRESH 확인
phorce play 1 --target sim:demo && echo UNIT-PASS # SUCCEEDED + UNIT-PASS 나오면 판정 통과
kill %1                                           # sim 정리
```

- `phorce: command not found` → 재로그인 후 재시도. 그래도 없으면 보드 격리.
- doctor 의 "카탈로그 NOT LOADED" 경고는 **정상** (모션 라이브러리 미적재 — D5, 별도 트랙).
- `play 1` SUCCEEDED 가 나오면 이 유닛은 **참가자에게 줄 수 있는 상태**다.

### ④ 기록

| 유닛 | 라벨/위치 | ② 커널 | ③ sim PASS | 비고 |
|---|---|---|---|---|
| 1호기 (angel-robotics) | 검수 원본 | ✅ (복구 대상, 커널만) | ✅ 2026-08-04 검수 | 현장조치 이력 있음 — USB 스크립트도 한 번 돌릴 것 |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| 6 |  |  |  |  |
| 7 |  |  |  |  |
| 8 |  |  |  |  |
| 9 |  |  |  |  |
| 10 |  |  |  |  |

## 2. 로봇 벤치 유닛 — 실물 퀵스타트 진입 (선택, 로봇 연결된 보드만)

퀵스타트 ①~⑤가 실물에서 돌려면 **운영자 단계가 먼저** 필요하다 (실물 게이트웨이는
자동 기동이 없다 — 검수 리포트 I1). 이 두 프로세스는 참가자가 아니라 **운영진 몫**:

```bash
# 터미널 A — EtherCAT 마스터 (preflight 13/13 PASS 가 1차 판정)
ros2 run agx_phorce_bridge phorce_monitor --ros-args -p nic:=eno1 -p mode:=command -p axes:=2

# 터미널 B — 모션 창구 (경고 배너 "실물 로봇이 움직입니다" 정상)
ros2 run agx_motion_slot motion_action_server --ros-args -p backend:=ecat
```

그 다음이 참가자 퀵스타트 그대로: `phorce doctor` → `ros2 topic hz /phorce/feedback`(≈1000)
→ `phorce list`(pcm 적재 슬롯 — 이름 없이 "슬롯 N" 표기 가능) → **영점 버튼(1번) 0.6초 + 3초 대기**
→ 주변 확인 후 `phorce play 1`.

- 비상 정지는 **물리 E-Stop 버튼**뿐이다. play 전 항상 주변 확인.
- preflight 에서 RT 커널 외 항목이 FAIL 이면 그 보드 개별 진단 (NIC 링크·EEE·offload 등은
  케이블/포트 문제일 수 있음).

## 3. 문제 해결

| 증상 | 원인/조치 |
|---|---|
| ② 에서 여전히 순정 커널 | fix 스크립트가 멱등 마커 때문에 스킵했을 가능성 — 실행 로그에서 skip 문구 확인 후 마스터 세션 쪽에 보고 |
| monitor: `libagx_console_msgs...not found` | deb 정리 단계가 안 돌았음 — `ldconfig -p \| grep agx_console_msgs` 로 확인, 없으면 USB 스크립트 재실행 |
| monitor: `error while loading shared libraries: librclcpp.so` | ld.so.conf 미등록 — `sudo agr-register-ldpath ros-humble /opt/ros/humble/lib` |
| `phorce doctor` (기본 target) 전부 MISSING | 실물 스택 미기동 상태의 **정상 출력** — §2 의 2프로세스를 먼저 |
| play 가 `REJECT_NOT_READY_FOR_MOTION` | 영점 버튼 안 눌림 — 사람이 눌러야 하며 기다려도 안 풀림 |
| `phorce list` 에 이름이 안 나옴 | 모션 라이브러리 미적재(D5, 행사 전 별도 적재 절차) — 재생 가능 여부는 pcm 슬롯이 정본이므로 판정에는 지장 없음 |
| 부팅이 비정상적으로 오래 걸림 | 부팅 후 `systemd-analyze blame \| head` 로 범인 확인 — first-boot identity 실패(D4)나 네트워크 대기 서비스의 90~120초 타임아웃일 가능성. 결과를 이미지 재작업 트랙에 보고 (이미지에서 고치면 10대 전부 빨라짐) |

## 4. 관련 문서

- 결함 상세·증거: [GOLDEN-IMAGE-INSPECTION-20260804.md](GOLDEN-IMAGE-INSPECTION-20260804.md)
- 참가자 퀵스타트(최종 사용자 경험 기준): [participant-guide/01-quickstart.html](../participant-guide/01-quickstart.html)
- 운영 런북 원문: [HACKATHON-GUIDE.md](../HACKATHON-GUIDE.md)
- 이미지 **근본 수정**(재작업)은 별도 트랙 — 이 런북은 현장 복구용이며, 이미지 파이프라인이
  고쳐지면 다음 배포부터는 §1 이 불필요해져야 한다.
