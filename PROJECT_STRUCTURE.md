# 프로젝트 구조

```
study-multi-modal-civil-complaint-classifier/
│
├── notebooks/                          # Jupyter 노트북
│   └── simple_civil_complaint_classifier.ipynb
│
├── src/                                # 소스 코드
│   ├── __init__.py
│   ├── models/                         # 모델 정의
│   │   ├── __init__.py
│   │   └── clip_classifier.py         # CLIP 기반 분류기
│   ├── data/                           # 데이터 로더 및 전처리
│   │   └── __init__.py
│   ├── training/                       # 학습 모듈
│   │   └── __init__.py
│   ├── evaluation/                     # 평가 모듈
│   │   └── __init__.py
│   └── utils/                          # 유틸리티 함수
│       └── __init__.py
│
├── data/                               # 데이터 저장소
│   ├── raw/                            # 원본 데이터
│   │   ├── .gitkeep
│   │   ├── image.png
│   │   └── image2.png
│   ├── processed/                      # 전처리된 데이터
│   │   └── .gitkeep
│   └── external/                       # 외부 데이터
│       └── .gitkeep
│
├── models/                             # 학습된 모델 저장
│   └── .gitkeep
│
├── configs/                            # 설정 파일
│   └── default.yaml                    # 기본 설정
│
├── experiments/                        # 실험 결과
│   └── .gitkeep
│
├── scripts/                            # 실행 스크립트
│   ├── train.py                       # 학습 스크립트
│   └── evaluate.py                    # 평가 스크립트
│
├── logs/                               # 로그 파일
│   └── .gitkeep
│
├── .gitignore                          # Git 무시 파일
├── requirements.txt                    # 패키지 의존성
├── README.md                           # 프로젝트 설명
└── PROJECT_STRUCTURE.md                # 이 파일
```

## 디렉토리 설명

### notebooks/
- 실험 및 탐색적 데이터 분석용 Jupyter 노트북

### src/
- 재사용 가능한 Python 모듈
- `models/`: 모델 정의
- `data/`: 데이터 로더 및 전처리
- `training/`: 학습 로직
- `evaluation/`: 평가 메트릭 및 로직
- `utils/`: 공통 유틸리티 함수

### data/
- `raw/`: 원본 데이터 (Git에 커밋하지 않음)
- `processed/`: 전처리된 데이터
- `external/`: 외부에서 가져온 데이터

### models/
- 학습된 모델 체크포인트 저장

### configs/
- YAML 형식의 설정 파일

### experiments/
- 실험 결과 및 메트릭 저장

### scripts/
- 명령줄에서 실행 가능한 스크립트

### logs/
- 학습 및 실행 로그

