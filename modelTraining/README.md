# PRISM modelTraining

궤적 예측 모델(K-갈래 혼합 가우시안) 학습 시스템. `dataGen`이 `%PRISM_ROOT%\Logs\Dataset`에
생성한 데이터셋을 읽어 학습하고, 산출물을 `%PRISM_ROOT%\Logs\Models\<job_id>\`에 남긴다.
논문관련.txt **[5.1] (rev.3)** 의 입출력 계약을 구현한다.

## 실행

```bash
# GUI (기본): 브라우저가 자동으로 열림
python %PRISM_ROOT%\modelTraining\main.py

# 헤드리스 학습 (4090 러너 PC용)
python main.py train --train-run $PRISM_ROOT/Logs/Dataset/<run> --val-run $PRISM_ROOT/Logs/Dataset/<valid_run> --seed 0 --tag paper

# 체크포인트에서 재개 / 평가 리포트 / 자체 테스트 / 합성데이터 데모
python main.py resume --job <job_id> --checkpoint last.pt
python main.py evaluate --job <job_id> --run $PRISM_ROOT/Logs/Dataset/<test_run>
python main.py selftest
python main.py demo
```

GUI 포트: **8790** (dataGen GUI는 8770 — 동시 실행 가능).

## GUI 사용 순서

1. **학습 데이터** — Logs 아래 런이 자동 나열됨. 역할(train/val/사용 안 함) 지정
2. **학습 설정** — 기본값이 권장값. 시드 스윕(예: `0,1,2,3,4`) 입력 시 잡이 순차 큐로 돌아 논문용 평균±표준편차 확보
3. **학습 시작** — 진행률/ETA/처리량/GPU 메모리, 손실·검증 지표·calibration(coverage@90)·갈래 활용(mode collapse 감시) 차트, 하단 로그
4. **중단** — 현재 스텝 마무리 → 최종 검증 → best 가중치로 `final/model_final.pt` 패키징 후 종료. **강제 중단**은 검증 생략
5. **평가 리포트 생성** — 층화(기종/의도별) 지표 + reliability 표 + 추론 지연 → `eval_report.json`

## 산출물 구조

```
%PRISM_ROOT%\Logs\Models\<job_id>\
  job.json            상태 스냅샷
  config.json         사용된 설정 전체 (재현용)
  metrics.jsonl       모든 메트릭 포인트 (논문 그림 원자료)
  train.log           텍스트 로그
  index_train.npz     사용된 샘플 인덱스 (재현용)
  checkpoints\        ckpt_ep####.pt (주기), best.pt, last.pt
  final\model_final.pt      최종 모델 (가중치+설정 자족적)
  final\model_card.json     환경/버전/데이터/파라미터 수/지연/최고 성능
  eval_report.json    평가 리포트 (생성 시)
```

## 모델 (권장 기본값)

```
입력  [B, A=8, 20, 13]  관측층 특징만 (GT 금지) — data/dataset.py 모듈 docstring 참조
      대상 중심 좌표계: 원점=마지막 관측 위치, +x=관측 진행방향 (GT 누수 없음)
구조  시점 임베딩 → 시간 Transformer(2층) → 어텐션 풀링
      → 개체 간 Transformer(2층) → 트렁크 → 3개 헤드
출력  궤적 [B, K=3, 30, 5] (μx, μy, logσx, logσy, ρ) + 갈래확률 [B,3] + 타입 [B,7]
손실  WTA NLL + 갈래 CE + 타입 CE(0.2) + 후반부 혼합 NLL 램프(π 보정)
규모  ~0.8M 파라미터, RTX 3080에서 추론 3.3ms/호출 (배치 1)
```

## 논문 관점 메모

- **coverage@90 차트가 0.9 기준선에 붙는 것**이 [9.3] calibration의 실시간 프록시.
  최종 채점은 분기 롤아웃 참 분포로 별도 수행(그건 dataGen 쪽 산출물).
- mode usage 차트에서 한 갈래가 1.0을 독식하면 **mode collapse** — K를 줄이거나
  w_mixture 램프를 늦출 것.
- train/val 분할은 **에피소드 단위** (샘플 단위 분할은 누수 — [8.1]).
- `metrics.jsonl`은 지우지 말 것. 논문 학습곡선 그림의 원자료.
