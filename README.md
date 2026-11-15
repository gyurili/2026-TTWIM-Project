<img width="1920" height="1080" alt="서비스 소개" src="https://github.com/user-attachments/assets/868d23d2-372c-400c-8124-86ec24d9e052" />

<br><br>

## 🏃 아자쓰! AZAS
> *발등 위 센서로 완성되는 실시간 러닝 자세 코칭, TTWIM!* <br>
> **AI 기반 실시간 러닝 패턴 분석을 통한 자세 교정 웨어러블 시스템**

<br><br>

### 📙 프로젝트 소개
본 프로젝트는 **발등 부착형 IMU 센서**로 러닝 중 착지 충격과 발 움직임 등 생체역학 데이터를 수집하고,
**SVM + Constrained DTW 이중 분석**으로 이상 주행 패턴을 실시간으로 감지하는 웨어러블 러닝 분석 시스템이다.
이상이 감지되면 **LLM 기반 음성(TTS) 코칭**을 즉시 제공하고, 운동 종료 후에는 **개인 맞춤형 러닝 리포트와 RAG 기반 AI 챗봇**으로
올바른 러닝 자세 형성을 지원한다.

<br><br>

### 🙌 개발 배경
국내 러닝 인구는 약 1,000만 명에 달하지만, 장거리 달리기의 연간 근골격계 부상 발생률은 **19.4~79.3%**, 전체 부상의 약 **80%가 하지**에서 발생한다.
특히 잘못된 착지 자세는 슬개대퇴증후군·아킬레스건염·족저근막염의 주요 원인이다.

그러나 기존 서비스는 한계가 뚜렷하다.
* **스마트워치·러닝 앱** (Garmin, NRC, Strava 등): GPS 기반 기록 측정 중심이며, 손목에서 측정하는 구조상 착지 충격·발 각도 분석이 어렵다.
* **3D 동작 분석 장비**: 정밀하지만 고비용·고정 설치 환경이 필요하다.
* **영상 기반 자세 분석**: 촬영 위치·각도 제약으로 야외 러닝에서 활용하기 어렵다.

자체 설문(2026.07, 30명)에서도 응답자의 **74.1%가 "올바른 자세를 몰라서"** 부상이 걱정된다고 답했고,
**93.3%가 러닝 중 실시간 자세 알림이 도움이 될 것**이라고 응답했다.
이에 **측정 지점을 손목에서 발등으로 옮기고**, 달리는 도중 즉시 교정받을 수 있는 예방 중심 코칭 시스템을 개발하였다.

<br><br>

### ✨ 주요 기능

#### 1. 실시간 러닝 데이터 수집
* Arduino Nano 33 BLE(LSM9DS1)의 6축 IMU(가속도 3축 + 자이로 3축)를 **50Hz**로 샘플링하고, 5개 샘플을 120바이트 패킷으로 묶어 BLE로 전송한다.
* 앱은 200포인트(4초) 윈도우 단위로 서버에 전송하며, 스마트폰 기압계 고도와 GPS 경로를 함께 기록한다.

#### 2. [1차 분석] SVM 기반 정상/비정상 자세 분류
* 4차 버터워스 저역통과필터(cutoff 20Hz)로 노이즈를 제거하고, 57개 통계 Feature 중 **상위 30개**를 추출해 RBF 커널 SVM으로 분류한다.
* 수직 가속도 표준편차가 0.5 미만인 윈도우는 정지 상태(신호 대기 등)로 보고 예측에서 제외한다.

#### 3. [2차 분석] Constrained DTW 기반 페이스 붕괴 감지
* 러닝 시작 후 20초 데이터로 **개인별 러닝 템플릿**을 만들고, 앱에서 Sakoe-Chiba band DTW로 현재 걸음과 비교해 `정상/주의/경고`를 판정한다.
* 템플릿과 이탈 점수·추세는 서버에 저장되어 LLM 코칭의 근거로 함께 쓰인다 (`/dtw`).

#### 4. [3차 분석] LLM 기반 유형 판정 및 실시간 음성 코칭
* 배치 내 비정상 윈도우가 과반이면 22개 유형 Feature, SVM 비정상 비율, DTW 이탈 정보, 기압계 경사(±1m) 정보를 LLM에 전달한다.
* **2단계 LLM 구조**: ① 충격 과다 / 발끌기 / 좁은 보폭 / 페이스 변화 4개 유형 판정(낮은 temperature) → ② 지식 베이스(`knowledge_base.json`) 범위 안에서 "분석 문장 + 교정 문장" 2문장 코칭 생성(temperature 0.8).
* LLM 호출 실패 시 유형별 고정 문구로 Fallback 하며, 앱은 수신 즉시 OS 내장 TTS로 송출한다.

#### 5. 러닝 리포트
* 세션 내 태그 비율, DTW 최대 이탈 지점, 실시간 코칭 이력을 종합해 **요약 · 잘한 점 · 개선점 · 다음 단계** 리포트를 생성한다.
* '안정 러닝' 비율이 70% 이상이면 격려 위주, 미만이면 상위 문제 태그 중심의 교정 위주로 방향을 정한다.
* 홈 화면 주간 기록과 월별 요약(거리·횟수·유형 분포) API를 함께 제공한다.

#### 6. RAG 기반 AI 챗봇
* 리포트 확인 후 멀티턴 후속 질문을 지원한다. 질문 재작성 → pgvector 벡터 검색 → LLM 리랭킹으로 논문·아티클 51편 중 근거를 선별하고,
  실제로 인용한 문서만 출처로 제공한다.

<br><br>

### 🧠 AI 모델

#### 데이터셋
* 피험자 **20명**이 정상 러닝과 4가지 비정상 러닝(충격 과다, 발끌기, 좁은 보폭, 페이스 변화)을 직접 수집·라벨링
* 200포인트 윈도우 총 **6,405개** (정상 3,381 / 비정상 3,024). 페이스 변화를 제외한 비정상 유형을 하나로 묶어 이진 분류로 학습

#### 모델 선정 (LOSO 교차검증)
처음 사용하는 러너에게도 동작하는지 확인하기 위해 Random Split 대신 **LOSO(Leave-One-Subject-Out)** 로 평가했다.
상호정보량 순위로 Feature 수를 10~57개까지 늘려가며 5개 모델을 비교한 결과는 다음과 같다.

| 모델 | Feature 수 | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Logistic Regression | 30 | 92.78% | 93.43% | 91.22% | **92.22%** |
| **SVM (RBF)** ✅ | 30 | 92.20% | 91.08% | **93.83%** | 91.98% |
| XGBoost | 20 | 84.16% | 83.56% | 88.40% | 84.29% |
| Random Forest | 45 | 82.01% | 82.88% | 88.14% | 82.69% |
| LightGBM | 20 | 82.54% | 86.10% | 80.76% | 81.27% |

비정상 자세의 **미탐을 줄이는 것**이 목적이므로, F1 차이가 1%p 이내인 모델 중 Recall이 가장 높은 **상위 30개 Feature의 RBF SVM**을 최종 채택했다
(충격 과다 러닝 탐지율 SVM 96.94% vs LR 85.71%). 학습은 Colab에서 진행하고 `app/data/gait_model.pkl`, `gait_scaler.pkl`, `gait_meta.json`으로 배포한다.

#### DTW 임곗값
같은 날 평소/느린/빠른 페이스로 달린 데이터로 이탈 점수를 검증했다 (정상 vs 페이스 변화 **AUC 0.949**).
주의 임곗값 **3**, 경고 임곗값 **5**에서 정상 러닝은 100% 정상, 느린 페이스는 91%가 경고로 판정되었고,
페이스를 바꾼 뒤 첫 판정 윈도우(6초 이내)에서 모든 참가자가 주의 판정을 받았다.
온디바이스 연산은 Galaxy S25 기준 회당 평균 15ms(3초 주기의 0.5%)다.

<br><br>

### 🧬 시스템 구조도
<img width="6984" height="3651" alt="최종 흐름도" src="https://github.com/user-attachments/assets/077e05cd-0df5-416a-8e4d-0787d1f380f6" />


<br><br>

### 📡 API

| Method | Endpoint | 설명 |
|---|---|---|
| POST | `/user/signup` | 회원가입 |
| POST | `/user/login` | 로그인 (JWT 발급) |
| GET / PATCH / DELETE | `/user/me` | 내 정보 조회 / 수정 / 탈퇴 |
| POST | `/imu/batch` | IMU 배치 업로드 |
| POST | `/imu/upload-db` | 앱 SQLite 버퍼 파일 업로드 |
| POST | `/predict` | SVM 추론 + 경사 판정 + LLM 실시간 코칭 |
| POST | `/dtw/` | 개인 러닝 템플릿·기준값 저장 |
| GET | `/dtw/me` | 내 러닝 템플릿 조회 |
| POST | `/running/complete` | 러닝 세션 종료 |
| POST / GET | `/running/{session_id}/feedback` | 최종 리포트 생성 / 조회 |
| GET | `/running/{session_id}/feedback/summary` | 세션 피드백 요약 |
| GET | `/running/home` | 홈 화면 주간 요약 |
| GET | `/running/monthly-summary` | 월별 요약 |
| POST / GET / DELETE | `/running/{session_id}/chat` | RAG 챗봇 질문 / 이력 조회 / 초기화 |

전체 스펙은 서버 실행 후 `/docs`(Swagger UI)에서 확인할 수 있다.

<br><br>

### 🔥 기술 스택

#### Back-end
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white"/> <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white"/> <img src="https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white"/>

#### AI / ML
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white"/> <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/> <img src="https://img.shields.io/badge/LangSmith-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>

#### Database
<img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/> <img src="https://img.shields.io/badge/pgvector-336791?style=for-the-badge&logo=postgresql&logoColor=white"/> <img src="https://img.shields.io/badge/amazonrds-527FFF?style=for-the-badge&logo=amazonrds&logoColor=white"/>

#### Infra / DevOps
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/> <img src="https://img.shields.io/badge/amazonec2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white"/> <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/> <img src="https://img.shields.io/badge/CodeRabbit-FF570A?style=for-the-badge&logo=coderabbit&logoColor=white"/>

#### Mobile / H/W (연관 저장소)
<img src="https://img.shields.io/badge/React%20Native-61DAFB?style=for-the-badge&logo=react&logoColor=black"/> <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white"/> <img src="https://img.shields.io/badge/Arduino-00878F?style=for-the-badge&logo=arduino&logoColor=white"/> <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white"/>

#### Collaboration
<img src="https://img.shields.io/badge/Github-black?style=for-the-badge&logo=Github&logoColor=white"/> <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=Discord&logoColor=white"/> <img src="https://img.shields.io/badge/Notion-black?style=for-the-badge&logo=Notion&logoColor=white"/> <img src="https://img.shields.io/badge/n8n-EA4B71?style=for-the-badge&logo=n8n&logoColor=white"/>

<br><br>

### 📊 타임라인

| 기간 | 주요 내용 |
|----------|-----------|
| 2026년 4월 | WBS·와이어프레임 작성, 요구사항·기능 명세서 및 기술 정의서 작성, 데이터 리서치 |
| 2026년 5월 | ERD·API 명세서 설계, FastAPI 초기 세팅 및 Docker 배포 환경 구성, 회원 API 선행 구현 |
| 2026년 6월 | IMU 하드웨어 제작·연동, 정상/비정상 러닝 데이터셋 구축(20명, 6,405 윈도우), AWS RDS·EC2 배포 |
| 2026년 7월 | LOSO 기반 모델 비교 및 SVM 채택, 사용자 설문(30명), 실시간 러닝 패턴 감지 기능 구축 |
| 2026년 8월 | Constrained DTW 개인 템플릿 연동, LLM 코칭·최종 리포트, pgvector RAG 챗봇 구현 |
| 2026년 9월 | 하드웨어·핵심 알고리즘 통합 테스트, 유형 판정 임곗값 통계 검증, 리포트 모델 응답 속도 개선 |
| 2026년 10월 | 실제 환경 사용성 검증 및 성능 최적화, 한이음 드림업 개발보고서 제출 (10.30) |

<br><br>

### 📁 디렉터리 구조

```
app/
├── api/            # 라우터 + 요청/응답 스키마 (user, imu, predict, dtw, running, chat)
├── services/       # 비즈니스 로직
│   ├── predict.py        # SVM 추론
│   ├── gait_features.py  # Feature 추출 (scipy 신호 처리)
│   ├── coaching.py       # 2단계 LLM 유형 판정 + 코칭 문구
│   ├── feedback.py       # 최종 러닝 리포트
│   ├── paper_rag.py      # pgvector 검색 + 리랭킹
│   └── chat.py           # 멀티턴 챗봇 (질문 재작성)
├── crud/, repositories/  # DB 접근 계층
├── models/         # SQLAlchemy ORM 모델
├── core/           # 설정, 보안(JWT), 예외
├── db/             # 엔진 / 세션
└── data/           # 모델(.pkl), 지식 베이스, RAG 코퍼스(paper/)
scripts/            # 논문 적재(ingest_papers.py), 검색 품질 평가(eval_retrieval.py)
tests/              # pytest API 테스트
```

<br><br>

### 🚀 실행 방법

```bash
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt     # 테스트까지 실행하려면 requirements-dev.txt
uvicorn app.main:app --reload
```
