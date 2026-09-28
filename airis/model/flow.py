"""조건부 flow matching 자세 모델. 소유자: C. 제안: docs/proposals/flow_matching.md.

체형 5개 + 시나리오를 조건으로 p(자세 | 체형, 시나리오)를 학습한다. 최적 자세가 봉우리 여러 개(팔 만세 /
팔 내림, yaw 대칭)로 갈리므로 평균을 내는 회귀 대신 분포를 학습하고, 샘플 여러 개를 뽑아 시뮬레이터로
고른다(predict.py).

학습 (conditional flow matching, 직선 경로)

    x_0 ~ 시작 분포,  x_1 = 데이터 자세,  t ~ U(0, 1)
    x_t = (1 − t)·x_0 + t·x_1
    loss = Σ_d m_d·(v_θ(x_t, t, c) − (x_1 − x_0))_d² / Σ_d m_d

    m 은 시나리오가 고정하지 않은 자세 변수 마스크(휠체어 hip·knee = 0), c 는 표준화한 체형 + 시나리오 원핫.
    배치는 (체형, 시나리오) 묶음마다 합이 1 이 되는 가중치로 뽑는다. 후보를 여러 개 저장한 묶음이
    학습을 독차지하지 않게 하기 위해서다.

샘플: x_0 에서 t = 0 → 1 로 v_θ 를 중점법으로 적분하고 [-1, 1] 로 자른다.

자세 공간 (PoseSpace): 시나리오 공통 7차원. 변수마다 학습 시나리오들의 pose_bounds 합집합으로 [-1, 1] 에
정규화한다. torso_yaw 는 데이터셋이 0~90° 로 접어 두므로(e4.fold_yaw, interfaces.md 회귀 계약) 접은 범위
[0, min(90, max|bound|)] 로 정규화한다.

torch 는 이 모듈의 함수 안에서 import 한다 (패키지 전체가 torch 에 묶이지 않게).
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, PoseParams, Scenario

POSE_KEYS: list[str] = [f.name for f in fields(PoseParams)]
BODY_KEYS: list[str] = [f.name for f in fields(BodyParams)]
YAW_KEY = "torso_yaw"
ARTIFACT_VERSION = 1


def folded_yaw_bounds(lo: float, hi: float) -> tuple[float, float]:
    """yaw 범위 [lo, hi] 가 fold_yaw(90° − |90° − |θ||) 뒤에 차지하는 범위."""
    return 0.0, min(90.0, max(abs(float(lo)), abs(float(hi))))


class PoseSpace:
    """시나리오 공통 7차원 자세 ↔ [-1, 1] 변환. 순서는 PoseParams 필드 순서."""

    def __init__(self, bounds: dict[str, tuple[float, float]]) -> None:
        self.keys = list(POSE_KEYS)
        arr = np.array([bounds[k] for k in self.keys], dtype=np.float64)
        span = arr[:, 1] - arr[:, 0]
        if np.any(span <= 0):
            bad = [k for k, s in zip(self.keys, span) if s <= 0]
            raise ValueError(f"자세 공간 범위의 hi 가 lo 보다 크지 않다: {bad}")
        self._lo, self._span = arr[:, 0], span

    @classmethod
    def from_scenarios(cls, scenarios: Sequence[Scenario]) -> "PoseSpace":
        bounds: dict[str, tuple[float, float]] = {}
        for k in POSE_KEYS:
            los, his = [], []
            for s in scenarios:
                if k in s.fixed_pose:
                    continue
                lo, hi = s.pose_bounds[k]
                if k == YAW_KEY:
                    lo, hi = folded_yaw_bounds(lo, hi)
                los.append(float(lo))
                his.append(float(hi))
            if not los:        # 모든 시나리오가 고정한 변수: 마스크로 빠지므로 폭은 아무래도 된다
                v = float(scenarios[0].fixed_pose[k])
                los, his = [v - 1.0], [v + 1.0]
            bounds[k] = (min(los), max(his))
        return cls(bounds)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: [float(lo), float(lo + s)] for k, lo, s in zip(self.keys, self._lo, self._span)}

    @classmethod
    def from_dict(cls, d: dict) -> "PoseSpace":
        return cls({k: (float(v[0]), float(v[1])) for k, v in d.items()})

    def mask(self, scenario: Scenario) -> np.ndarray:
        """시나리오가 고정하지 않은 변수 1, 고정한 변수 0. (7,) float32."""
        return np.array([0.0 if k in scenario.fixed_pose else 1.0 for k in self.keys], dtype=np.float32)

    def encode(self, deg: np.ndarray) -> np.ndarray:
        """(N, 7) degree → (N, 7) [-1, 1]. 범위 밖은 자른다."""
        deg = np.atleast_2d(np.asarray(deg, dtype=np.float64))
        return np.clip(2.0 * (deg - self._lo) / self._span - 1.0, -1.0, 1.0)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """(N, 7) [-1, 1] → (N, 7) degree."""
        x = np.clip(np.atleast_2d(np.asarray(x, dtype=np.float64)), -1.0, 1.0)
        return self._lo + (x + 1.0) * self._span / 2.0

    def to_pose(self, x: np.ndarray, scenario: Scenario) -> PoseParams:
        """정규화 벡터 하나 → 시나리오 제약(범위 + fixed_pose) 안의 PoseParams (interfaces.md 회귀 계약)."""
        from airis.optimize.encoding import PoseEncoder

        deg = self.decode(x)[0]
        return PoseEncoder(scenario).clip_pose(PoseParams(**dict(zip(self.keys, map(float, deg)))))


# ---------- 학습 데이터 ----------

def _cand_cols() -> list[str]:
    return [f"cand_pose_{k}" for k in POSE_KEYS]


def _has_candidates(row) -> bool:
    v = row.get("cand_pose_shoulder_abduction") if hasattr(row, "get") else None
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and len(v) > 0


def body_from_row(row) -> BodyParams:
    """데이터셋 행의 body_* 열 → BodyParams."""
    return BodyParams(**{k: float(row[f"body_{k}"]) for k in BODY_KEYS})


def training_arrays(df, space: PoseSpace, scenarios: dict[str, Scenario]) -> dict[str, np.ndarray]:
    """데이터셋 DataFrame → 학습 배열.

    행에 cand_pose_* 목록(dataset.py n_candidates > 0)이 있으면 그 후보들을, 없으면 pose_* 최적 자세 하나를 쓴다.
    반환: body (N,5), scenario (N,) str, x (N,7), mask (N,7), weight (N,) (묶음마다 합 1), group (N,) int
    """
    body, scen, deg, group = [], [], [], []
    for g, (_, row) in enumerate(df.iterrows()):
        b = [float(row[f"body_{k}"]) for k in BODY_KEYS]
        if _has_candidates(row):
            cand = np.stack([np.asarray(row[c], dtype=np.float64) for c in _cand_cols()], axis=1)
        else:
            cand = np.array([[float(row[f"pose_{k}"]) for k in POSE_KEYS]])
        for d in cand:
            body.append(b)
            scen.append(str(row["scenario"]))
            deg.append(d)
            group.append(g)
    group_arr = np.array(group, dtype=np.int64)
    counts = np.bincount(group_arr)
    missing = sorted(set(scen) - set(scenarios))
    if missing:
        raise KeyError(f"시나리오 설정에 없는 이름: {missing}")
    return {
        "body": np.array(body, dtype=np.float32).reshape(-1, len(BODY_KEYS)),
        "scenario": np.array(scen),
        "x": space.encode(np.array(deg)).astype(np.float32),
        "mask": np.stack([space.mask(scenarios[s]) for s in scen]).astype(np.float32),
        "weight": (1.0 / counts[group_arr]).astype(np.float64),
        "group": group_arr,
    }


# ---------- 모델 ----------

@dataclass
class FlowConfig:
    hidden: int = 128
    layers: int = 3
    time_freqs: int = 8            # 시각 t 의 사인 임베딩 주파수 수
    steps: int = 4000              # 학습 스텝 (미니배치 수)
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    source: str = "gaussian"       # "gaussian": N(0, I) | "default_pose": 기본 자세(PoseParams()) + 노이즈
    source_sigma: float = 1.0
    ode_steps: int = 16            # 샘플링 적분 스텝 (중점법)
    seed: int = 0
    device: str = "cpu"            # 학습 장치 "cpu" | "cuda" | "auto". 샘플링은 항상 CPU


def _make_net(in_dim: int, out_dim: int, hidden: int, layers: int):
    import torch.nn as nn

    mods: list = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


@dataclass
class PoseFlow:
    space: PoseSpace
    scenario_names: list[str]
    cond_mean: np.ndarray            # (5,) 체형 표준화
    cond_std: np.ndarray             # (5,)
    source_mean: dict[str, list[float]]  # 시나리오별 시작 분포 중심 (gaussian 이면 0)
    cfg: FlowConfig
    net: object = None
    meta: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @property
    def in_dim(self) -> int:
        return len(POSE_KEYS) + 1 + 2 * self.cfg.time_freqs + len(BODY_KEYS) + len(self.scenario_names)

    def build_net(self):
        self.net = _make_net(self.in_dim, len(POSE_KEYS), self.cfg.hidden, self.cfg.layers)
        return self.net

    # --- 텐서 도우미 ---
    def _cond(self, body: np.ndarray, scen: Sequence[str]):
        import torch

        unknown = sorted(set(scen) - set(self.scenario_names))
        if unknown:
            raise KeyError(f"학습에 없던 시나리오: {unknown} (학습: {self.scenario_names})")
        b = (np.asarray(body, dtype=np.float32) - self.cond_mean) / self.cond_std
        onehot = np.zeros((len(scen), len(self.scenario_names)), dtype=np.float32)
        onehot[np.arange(len(scen)), [self.scenario_names.index(s) for s in scen]] = 1.0
        return torch.from_numpy(np.concatenate([b, onehot], axis=1).astype(np.float32))

    def _time_embed(self, t):
        import torch

        k = torch.arange(self.cfg.time_freqs, dtype=t.dtype, device=t.device)
        ang = t * (2.0 ** k) * math.pi
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=1)

    def velocity(self, x, t, c):
        import torch

        return self.net(torch.cat([x, self._time_embed(t), c], dim=1))

    def _source_center(self, scen: Sequence[str]):
        import torch

        return torch.tensor(np.array([self.source_mean[s] for s in scen], dtype=np.float32))

    # --- 샘플링 ---
    def sample(self, body: BodyParams, scenario_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        """체형 하나 × 시나리오 하나에서 정규화 자세 n 개. (n, 7), 같은 seed 면 같은 결과."""
        import torch

        bvec = np.array([[getattr(body, k) for k in BODY_KEYS]] * n, dtype=np.float32)
        scen = [scenario_name] * n
        c = self._cond(bvec, scen)
        g = torch.Generator().manual_seed(int(seed))
        x = self._source_center(scen) + self.cfg.source_sigma * torch.randn(n, len(POSE_KEYS), generator=g)
        steps = max(1, int(self.cfg.ode_steps))
        dt = 1.0 / steps
        self.net.eval()
        with torch.no_grad():
            for i in range(steps):
                t = torch.full((n, 1), i * dt)
                v1 = self.velocity(x, t, c)
                v2 = self.velocity(x + 0.5 * dt * v1, t + 0.5 * dt, c)
                x = x + dt * v2
        return x.clamp(-1.0, 1.0).numpy().astype(np.float64)

    def sample_poses(self, body: BodyParams, scenario: Scenario, n: int, *, seed: int = 0) -> list[PoseParams]:
        """sample → 시나리오 제약 안의 PoseParams 목록."""
        return [self.space.to_pose(x, scenario) for x in self.sample(body, scenario.name, n, seed=seed)]

    # --- 저장 ---
    def save(self, path: Path | str) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": ARTIFACT_VERSION,
            "config": asdict(self.cfg),
            "space": self.space.to_dict(),
            "scenario_names": list(self.scenario_names),
            "cond_mean": [float(v) for v in self.cond_mean],
            "cond_std": [float(v) for v in self.cond_std],
            "source_mean": {k: [float(v) for v in vs] for k, vs in self.source_mean.items()},
            "meta": json.loads(json.dumps(self.meta, default=str)),   # weights_only 로드가 되게 기본 타입만
            "history": json.loads(json.dumps(self.history, default=float)),
            "state_dict": self.net.state_dict(),
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PoseFlow":
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"자세 모델 산출물이 없다: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"산출물 버전 {ck.get('version')} (코드 {ARTIFACT_VERSION})")
        model = cls(
            space=PoseSpace.from_dict(ck["space"]),
            scenario_names=list(ck["scenario_names"]),
            cond_mean=np.array(ck["cond_mean"], dtype=np.float32),
            cond_std=np.array(ck["cond_std"], dtype=np.float32),
            source_mean={k: list(v) for k, v in ck["source_mean"].items()},
            cfg=FlowConfig(**ck["config"]),
            meta=dict(ck.get("meta", {})),
            history=list(ck.get("history", [])),
        )
        model.build_net().load_state_dict(ck["state_dict"])
        model.net.eval()
        return model


def _resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train_pose_flow(df, scenarios: dict[str, Scenario], cfg: FlowConfig = FlowConfig(), *,
                    meta: dict | None = None, log=None) -> PoseFlow:
    """데이터셋 DataFrame 으로 모델을 학습한다. 시나리오 목록은 df 의 scenario 열에서 정한다."""
    import torch

    if cfg.source not in ("gaussian", "default_pose"):
        raise ValueError(f"source 는 gaussian | default_pose: {cfg.source!r}")
    names = sorted(set(df["scenario"].astype(str)))
    space = PoseSpace.from_scenarios([scenarios[n] for n in names])
    arr = training_arrays(df, space, scenarios)
    if cfg.source == "default_pose":
        from airis.optimize.e4 import fold_pose
        from airis.optimize.encoding import PoseEncoder

        source_mean = {}
        for n in names:
            d = fold_pose(PoseEncoder(scenarios[n]).clip_pose(PoseParams()))
            source_mean[n] = space.encode(np.array([[d[k] for k in POSE_KEYS]]))[0].tolist()
    else:
        source_mean = {n: [0.0] * len(POSE_KEYS) for n in names}

    model = PoseFlow(
        space=space, scenario_names=names,
        cond_mean=arr["body"].mean(axis=0),
        cond_std=np.maximum(arr["body"].std(axis=0), 1e-6).astype(np.float32),
        source_mean=source_mean, cfg=cfg, meta=dict(meta or {}),
    )
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    net = model.build_net().to(device)

    x1_all = torch.from_numpy(arr["x"]).to(device)
    m_all = torch.from_numpy(arr["mask"]).to(device)
    c_all = model._cond(arr["body"], arr["scenario"]).to(device)
    x0c_all = model._source_center(arr["scenario"]).to(device)
    probs = torch.from_numpy(arr["weight"] / arr["weight"].sum()).float().to(device)
    g = torch.Generator(device=device).manual_seed(cfg.seed)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    net.train()
    running = 0.0
    for step in range(1, cfg.steps + 1):
        idx = torch.multinomial(probs, cfg.batch_size, replacement=True, generator=g)
        x1, m, c = x1_all[idx], m_all[idx], c_all[idx]
        x0 = x0c_all[idx] + cfg.source_sigma * torch.randn(x1.shape, generator=g, device=device)
        t = torch.rand((x1.shape[0], 1), generator=g, device=device)
        xt = (1.0 - t) * x0 + t * x1
        err = (model.velocity(xt, t, c) - (x1 - x0)) ** 2
        loss = ((err * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += float(loss)
        if step % 100 == 0 or step == cfg.steps:
            n = 100 if step % 100 == 0 else step % 100
            model.history.append({"step": step, "loss": running / n})
            if log is not None:
                log(f"[flow] step {step}/{cfg.steps}  loss {running / n:.4f}")
            running = 0.0

    model.net = net.to("cpu").eval()
    model.meta.setdefault("n_rows", int(len(df)))
    model.meta.setdefault("n_train_points", int(len(arr["x"])))
    return model


def flow_loss(model: PoseFlow, df, scenarios: dict[str, Scenario], *, n_t: int = 8, seed: int = 0) -> float:
    """검증용 flow matching 손실 (t 와 x_0 를 n_t 번 뽑아 평균). 학습 손실과 같은 식."""
    import torch

    arr = training_arrays(df, model.space, scenarios)
    g = torch.Generator().manual_seed(seed)
    x1 = torch.from_numpy(arr["x"])
    m = torch.from_numpy(arr["mask"])
    c = model._cond(arr["body"], arr["scenario"])
    x0c = model._source_center(arr["scenario"])
    w = torch.from_numpy(arr["weight"]).float()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_t):
            x0 = x0c + model.cfg.source_sigma * torch.randn(x1.shape, generator=g)
            t = torch.rand((x1.shape[0], 1), generator=g)
            err = (model.velocity((1 - t) * x0 + t * x1, t, c) - (x1 - x0)) ** 2
            per = (err * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            total += float((per * w).sum() / w.sum())
    return total / n_t
