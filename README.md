# 가우시안 프로세스 회귀 기반 실험 이론식 보정기

실험값과 기존 이론식의 잔차를 네 가지 GP 커널로 학습하고, 교차검증 성능을 비교해 최적 보정 모델을 선택하는 Streamlit 웹사이트입니다.

## 모델

- `r_i = y_exp,i - f_theory(x_i)`
- `y_corrected(x) = f_theory(x) + μ_GP(x)`

## 비교 커널

- RBF
- Matérn(ν=1.5)
- Rational Quadratic
- Linear(Dot Product)

모든 커널에 ConstantKernel과 WhiteKernel을 결합합니다.

## 검증 방식

- 모든 x가 서로 다르면 Leave-One-Out
- 동일한 x에서 반복 측정값이 있으면 Leave-One-X-Group-Out
- 주요 선택 지표: 교차검증 RMSE
- RMSE가 최저값의 3% 이내인 후보가 여러 개면 NLPD, 95% 구간 포함률, 모델 단순성 순으로 결정

## 설치와 실행

Python 3.10~3.12 환경을 권장합니다.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV 형식

```csv
x,y_exp
0.01,2.25
0.02,3.16
0.03,3.88
```

## 이론식 예시

```text
2*pi*sqrt(x/kappa)
```

`kappa`가 자동으로 감지되며 화면에서 값을 입력할 수 있습니다.

허용 함수: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, 상수 `pi`, `e`

## 제한

- 독립변수 하나인 `x`만 지원
- 최대 250개 데이터
- 이론식 매개변수 자체의 불확실성은 계산하지 않음
- 학습 범위 밖 외삽은 권장하지 않음
- 인터넷 공개 배포 시 수식 입력 보안 검토가 추가로 필요함
