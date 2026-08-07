# SIGMA — 제1회 로봇 해커톤 2026
로보틱어스(Roboticus)가 주최하는 제1회 로봇 해커톤(2026. 8. 5.~8. 8., KAIST) 참가팀 **SIGMA**(서울대)의 저장소입니다.

- 팀원: 백시은 · 김정환 · 이진명
- 대회: https://robotic-us.com

## 지적재산권

> 본 프로젝트의 지적재산권은 SIGMA 팀(팀원 전원)에게 있으며, 본 대회의 주최 측(로보틱어스)은 아카이브 및 홍보 목적으로만 본 저장소를 활용합니다.

본 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## iPad 가상 아기 + 모션센서

`feature/ipad-motion-sensor` 실험에서는 iPad가 가상 아기 얼굴을 표시하는
동시에 가속도계와 자이로스코프로 요람의 실제 움직임을 측정합니다.

```bash
python3 serve.py --baby --personality --policy reflex --no-ros
```

실행 후 같은 네트워크의 iPad에서 출력된 `/baby` 주소를 열고 **Start motion
sensor**를 누릅니다. 센서가 최근 1초 안에 연결되어 있으면 가상 아기는
MotionEngine의 명령값이 아니라 iPad가 측정한 움직임에 반응합니다. 연결이
없으면 기존 데스크톱 시뮬레이션으로 자동 복귀합니다.

iPadOS의 웹 모션센서는 HTTPS 보안 컨텍스트가 필요합니다. 신뢰할 수 있는
로컬 인증서와 키를 준비한 뒤 다음처럼 실행합니다. 현재 컴퓨터의 LAN IP가
`10.249.185.84`라면:

```bash
./tools/make_https_cert.sh 10.249.185.84
python3 serve.py --baby --personality --policy reflex --no-ros \
  --certfile .local-certs/server.crt --keyfile .local-certs/server.key
```

`.local-certs/sigma-ca.crt`만 iPad로 전송해 프로파일을 설치한 뒤, iPad의
`설정 → 일반 → 정보 → 인증서 신뢰 설정`에서 **SIGMA Local iPad CA**를
신뢰해야 합니다. `sigma-ca.key`와 `server.key`는 절대 전송하지 않습니다.
Jetson에서는 Jetson의 LAN IP로 인증서를 다시 만들어야 합니다. 센서 데이터는
`POST /motion-sensor`, 가상 아기 상태는 기존 `GET /events` SSE를 사용합니다.
