# 다른 노트북에서 계획 데이터셋 나눠 돌리기

계획 데이터셋(체형 100개 × 사용자 유형 3개 = 300행)은 한 행에 약 2시간이 걸린다. 행끼리 서로 독립이라 **구간(사용자 유형 하나의 체형 20개)을 노트북마다 하나씩 맡아** 돌리고, 끝나면 파일을 합친다. 어떤 구간이 비어 있는지는 구간표 `docs/remote_plan_board.md`에 있다. 이 문서는 맡는 사람(팀원)과 합치는 사람(총괄)이 할 일을 순서대로 적은 것이다.

- GPU는 필요 없다. CPU만 쓴다.
- 도구는 `tools/remote/plan_share.py` 하나다. 가상환경(venv)을 따로 만들어 쓰므로 노트북에 이미 깔린 Python 패키지를 건드리지 않는다.
- 기준 노트북(총괄)과 **같은 코드·같은 설정**으로 만들어야 합칠 수 있다. 도구가 이것을 자동으로 맞추고 검사한다.

## 1. 필요한 것

| 항목 | 조건 |
|---|---|
| 운영체제 | Windows 10·11 (macOS·Linux도 같은 명령으로 된다. 확인은 Windows에서만 했다) |
| Python | **3.13 권장** (기준 노트북과 같음). 3.12는 될 가능성이 높지만 확인하지 않았다. 3.11 이하는 안 된다 |
| git | 저장소를 `git clone`으로 받아야 한다 (zip 내려받기는 안 된다) |
| CPU·메모리 | 코어 8개 이상, 메모리 16 GB 이상 권장 |
| 디스크 | 약 3 GB (패키지 포함) |
| 시간 | 구간 하나(20행)에 5~6시간(기준 노트북 속도). 그동안 노트북이 느려지고, 절전·종료하면 멈춘다 |

Python 버전 확인:

```bash
python --version
```

## 2. 설치 (약 10분)

저장소를 받는다.

```bash
git clone https://github.com/SKKU-DNA-Hero-AIRIS/project-AIRIS-v2.git
```

받은 폴더로 들어간다.

```bash
cd project-AIRIS-v2
```

설치와 점검을 한 번에 한다.

```bash
python tools/remote/plan_share.py setup
```

이 명령이 하는 일:

1. `.venv-remote` 가상환경을 만들고 필요한 패키지 8개만 설치한다 (`tools/remote/requirements-plan.txt`, 버전 고정).
2. 옆 폴더 `project-AIRIS-v2-run-4f9f310`에 **기준 노트북과 같은 커밋**의 코드를 꺼낸다. 실제 계산은 이 폴더의 코드로 돈다.
3. 점검한다: 물리 설정·분사구 설정의 식별값, 커밋, 체형 샘플이 기준 노트북과 같은지 보고, 아주 작은 실행 1건을 끝까지 돌려 본다.

마지막 줄이 `[통과] 이 노트북에서 돌릴 수 있다`면 된 것이다. `[실패]`가 나오면 화면 내용을 그대로 총괄에게 보낸다.

점검만 다시 하고 싶을 때:

```bash
python tools/remote/plan_share.py check
```

## 3. 속도 재 보기 (약 3분)

다른 프로그램을 닫고 실행한다.

```bash
python tools/remote/plan_share.py bench
```

"행 하나 약 ○분, 시간당 약 ○행"이 나온다. 이 숫자를 총괄에게 알려 주면 몇 행을 맡길지 정한다. 기준 노트북의 실측은 행 하나 122분, 시간당 3.9행(프로세스 8개)이다.

- 이 값은 어림값이다. 실제 속도는 `status`로 본다.
- 프로세스를 늘린다고 빨라지지 않는다(메모리 속도가 병목이다. 기준 노트북은 12개가 8개보다 3배 느렸다). 코어가 8개보다 적으면 `--processes 4`로도 재 본다.

```bash
python tools/remote/plan_share.py bench --processes 4
```

## 4. 돌리기

### 4.1 절전 끄기

돌리는 동안 노트북이 절전에 들어가거나 꺼지면 계산이 멈춘다. 시작 전에 직접 해 둔다.

- 전원 어댑터를 꽂는다.
- Windows: 설정 → 시스템 → 전원 → "화면 및 절전"에서 **전원 연결 시 절전 모드를 "안 함"**으로 바꾼다. 덮개를 닫아도 절전에 들어가지 않게 하려면 제어판 → 전원 옵션 → "덮개를 닫으면 수행할 작업"을 "아무 것도 안 함"으로 바꾼다.
- 끝나면 원래대로 돌려놓는다.

### 4.2 시작

`git pull`로 구간표(`docs/remote_plan_board.md`)를 최신으로 받고, 담당이 비어 있는 "대기" 구간 하나를 총괄에게 말한 뒤 시작한다.

```bash
python tools/remote/plan_share.py run --segment P4
```

- `--segment`: 구간표의 구간 이름(P1~P4 임산부, W1~W4 휠체어). 구간 목록은 `python tools/remote/plan_share.py segments`로도 볼 수 있다.
- `--processes`: 기본 8. 3절에서 더 빨랐던 값을 쓴다.
- 시작하면 40초 뒤 첫 줄(`작업 ○개 (건너뜀 ○)`)을 보여 준다. **명령 창을 닫아도 계속 돈다.**
- 한 노트북에서는 한 번에 구간 하나만 돈다.
- 다른 사람이 하다 만 구간을 이어받을 때는 받은 파일을 함께 준다: `run --segment P4 --seed-file <받은 파일>`.
- 구간표에 없는 범위를 총괄이 따로 정해 줄 때: `run --scenario pregnant --bodies 40-49`.

### 4.3 진행 확인

```bash
python tools/remote/plan_share.py status
```

도는 중인지, 저장된 행 수, 최근 속도, 예상 남은 시간이 나온다. 5행마다 파일에 저장한다. 첫 저장까지 몇 시간이 걸릴 수 있다(행 하나가 약 2시간이고 여러 행이 동시에 돈다).

### 4.4 멈추기, 이어 돌리기

```bash
python tools/remote/plan_share.py stop --yes
```

멈추면 저장 전이던 계산(최대 몇 시간 분량)은 사라지고, 저장된 행은 남는다. 노트북이 꺼졌거나 멈춘 뒤에는 같은 명령으로 이어 돌린다. `--seed-file`은 다시 줄 필요가 없다.

```bash
python tools/remote/plan_share.py run --segment P4
```

### 4.5 끝나면

`status`가 "끝났다"를 보여 주면 그 줄에 적힌 파일 하나를 총괄에게 보낸다.

```text
project-AIRIS-v2/outputs/remote_share/plan_dataset_n9_<구간>.parquet
```

보낸 뒤 구간표에서 다음 "대기" 구간을 받아 4.2부터 다시 한다. 다 끝나지 않았어도 중간에 보낼 수 있다(저장된 행까지만 들어 있다). 모든 일이 끝난 뒤 지워도 되는 것: `project-AIRIS-v2` 폴더와 옆의 `project-AIRIS-v2-run-4f9f310` 폴더 전체.

## 5. 총괄이 할 일 (기준 노트북)

### 5.1 구간표 관리

구간을 누가 맡았는지, 어디까지 받았는지는 `docs/remote_plan_board.md`에 적고 구간이 끝날 때마다 고친다. 받은 파일의 구간별 완료 행 수는 이렇게 본다.

```bash
python tools/remote/plan_share.py segments ../airis-v2-e4bound/data/datasets/plan_dataset_n9.parquet outputs/remote_share/plan_dataset_n9_P4.parquet
```

하다 만 구간을 다른 사람에게 넘길 때는 받은 파일을 그대로 `--seed-file`로 쓰게 한다. 기준 노트북의 데이터셋에서 특정 범위만 떼어 줄 때:

```bash
python tools/remote/plan_share.py export-seed --from ../airis-v2-e4bound/data/datasets/plan_dataset_n9.parquet --scenario wheelchair --bodies 19-39 --out outputs/remote_share/plan_seed_W1.parquet
```

### 5.2 기준 노트북의 범위

기준 노트북은 선 자세(D1)만 만든다(2026-10-09 10:55 전환). 임산부·휠체어 구간은 모두 다른 노트북 몫이라 겹치지 않는다. 겹친 행이 생겨도 합칠 때 하나만 남으므로 결과가 틀리지는 않고 시간만 버린다.

### 5.3 합치기

```bash
python tools/remote/plan_share.py merge --out outputs/remote_share/plan_dataset_n9_merged.parquet ../airis-v2-e4bound/data/datasets/plan_dataset_n9.parquet outputs/remote_share/plan_dataset_n9_P4.parquet outputs/remote_share/plan_dataset_n9_W4.parquet
```

- 설정 도장 열(물리·분사구 식별값, 단계 수, 평가 예산 등 21개)이 파일마다 같은지 검사하고, 다르면 합치지 않는다.
- 같은 (체형 번호, 유형) 행이 겹치면 **앞에 적은 파일**의 것을 남긴다.
- 유형별 행 수와 "세 유형이 모두 있는 체형 수"를 보여 준다. F의 학습은 세 유형이 모두 있는 체형만 쓴다.

합친 파일은 F에 넘기기 전에 통합이 `airis.model.plan_data.read_plan_dataset`으로 읽히는지 확인한다.

## 6. 알아 둘 것

- **다른 CPU에서는 같은 체형이라도 결과가 소수점 끝자리에서 조금 다를 수 있다.** 최적화가 긴 계산이라 작은 차이가 쌓인다. 각 행은 그 자체로 유효한 최적화 결과라 학습에는 문제가 없다. 다만 같은 행을 두 노트북에서 만들어 비교하면 값이 똑같지는 않다.
- **실행 코드는 커밋 `4f9f310`에 고정돼 있다.** 그 뒤 main에 들어간 후보 거리 계산 수정(#176)은 일부러 넣지 않았다. 한 데이터셋에 두 방식이 섞이지 않게 하기 위해서이고, 중복 후보는 읽는 쪽에서 거른다(`docs/interfaces.md` 계획 데이터셋 절).
- 점수는 시뮬레이션 안의 상대 비교값이다.

## 7. 문제가 생기면

| 증상 | 조치 |
|---|---|
| `Python 3.12 이상이 필요하다` | python.org에서 3.13을 설치하고 다시 `setup` |
| 패키지 설치 실패 | Python 버전·운영체제와 오류 화면을 총괄에게 보낸다. 3.13이 아니면 3.13으로 다시 해 본다 |
| `실행 폴더를 만들지 못했다` | zip이 아니라 `git clone`으로 받았는지 확인 |
| `기준 노트북과 설정이 다르다` | 설정 파일을 고치지 않았는지 확인하고, 화면 내용을 총괄에게 보낸다. 이 상태로는 돌리지 않는다 |
| `이미 실행 중이다` | `status`로 확인. 다시 시작하려면 `stop --yes` 뒤 `run` |
| `시작 직후 끝났다` | 화면의 오류 내용을 총괄에게 보낸다 |
| `status`가 "멈춰 있음" | 노트북이 절전·종료됐던 것이다. 4.4의 이어 돌리기 명령을 실행 |
| 노트북이 너무 느려 쓸 수 없다 | `stop --yes` 뒤 `--processes 4`로 다시 `run` |
