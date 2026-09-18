"""MakeHuman 기본 메시 → 시뮬레이션용 sim 메시 (`data/meshes/makehuman/sim_mesh.npz`). 소유자: B.

오프라인 1회 실행한다 (결과 npz 를 저장소에 커밋, CC0 파생물). `docs/mesh_transition.md` 설계 2.

    python scripts/build_sim_mesh.py                 # 기본 목표 5,600 삼각형
    python scripts/build_sim_mesh.py --faces 6000

방법
- 원본 body 메시(13,380 정점, 26,756 삼각형)는 닫힌 다양체이고 정점이 y 거울 완전 대칭이다.
  이음선(y = 0) 정점 188개, 이음선을 가로지르는 면은 없다 (왼쪽·오른쪽 반쪽 13,378 면씩).
- 왼쪽 반쪽(y ≥ 0)만 QEM(Garland–Heckbert 이차 오차) half-edge collapse 로 줄이고 오른쪽은 거울 복제한다.
  → 정점·면이 구성상 정확히 거울 대칭이고, 원본 정점만 남으므로 뼈 가중치가 정확히 보존된다.
- 제약: 이음선 정점은 지우지 않는다 (거울 짝이 맞게). 정점은 같은 부위(bone_part) 이웃으로만 합친다
  (부위 경계가 흐려지지 않게). 합친 뒤 면이 뒤집히거나(법선 내적 < FLIP_DOT) 한 점으로 줄어들면 거부한다.
  링크 조건(공통 이웃 = 2)으로 비다양체를 막는다.

npz 키 (docs 와 D·A 공유)
    vertices (V,3) f64      휴지 자세, 우리 좌표 (x 앞, y 왼쪽, z 위), m
    faces (F,3) i32         바깥에서 반시계
    weight_bone (V,9) i16, weight_val (V,9) f64   원본 가중치를 자르지 않음 (정점당 최대 9, 빈칸 0)
    bone_names (B,) str, bone_parent (B,) i32, bone_head (B,3), bone_tail (B,3)
    face_part (F,) i8       PART_NAMES 인덱스 (몸통은 torso_front, 앞뒤는 런타임에 법선으로)
    mirror_vertex (V,) i32, mirror_face (F,) i32, mirror_face_perm (F,3) i8
                            mirror_face[f] 의 정점 k = mirror_vertex[faces[f, mirror_face_perm[f, k]]]
    obj_vertex (V,) i32     base.obj 전체 정점 번호 (모디파이어 타깃 적용용)
    source (str)            원본·생성 정보
"""
from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airis.sim.human_mesh import (  # noqa: E402
    MAKEHUMAN_DIR, _read_obj_body, bone_part, load_makehuman, mirror_bone_index,
)

SEAM_TOL = 1e-9          # 이음선 판정 |y| (원본은 정확히 0, 비이음선 최소 |y| 1.2 mm)
FLIP_DOT = 0.2           # 합친 뒤 면 법선과 원래 법선의 최소 내적
MAX_BONES = 9            # 정점당 가중치 뼈 최대 수 (원본 최대 9)
OUT = MAKEHUMAN_DIR / "sim_mesh.npz"


def _plane_quadrics(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """정점별 이차 오차 행렬 Q (V,4,4) = Σ 면적 · p pᵀ, p = (n, −n·x)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(area2, 1e-30)[:, None]
    p = np.concatenate([n, -(n * a).sum(1, keepdims=True)], axis=1)       # (F,4)
    K = 0.5 * area2[:, None, None] * p[:, :, None] * p[:, None, :]         # (F,4,4)
    Q = np.zeros((v.shape[0], 4, 4))
    for k in range(3):
        np.add.at(Q, f[:, k], K)
    return Q


def decimate_half(v: np.ndarray, faces: np.ndarray, locked: np.ndarray, part: np.ndarray,
                  target_faces: int) -> np.ndarray:
    """half-edge collapse QEM. 반환: 남은 면 (원래 정점 인덱스)."""
    Q = _plane_quadrics(v, faces)
    faces = [list(f) for f in faces]
    alive_face = [True] * len(faces)
    vf: dict[int, set[int]] = {}
    for fi, f in enumerate(faces):
        for x in f:
            vf.setdefault(x, set()).add(fi)

    def nbrs(x: int) -> set[int]:
        s = set()
        for fi in vf[x]:
            s.update(faces[fi])
        s.discard(x)
        return s

    ver = {x: 0 for x in vf}
    heap: list = []

    def cost(u: int, w: int) -> float:
        h = np.append(v[w], 1.0)
        return float(h @ (Q[u] + Q[w]) @ h)

    def push_around(x: int) -> None:
        for y in nbrs(x):
            for u, w in ((x, y), (y, x)):
                if not locked[u] and part[u] == part[w]:
                    heapq.heappush(heap, (cost(u, w), u, w, ver[u], ver[w]))

    for x in vf:
        push_around(x)

    n_faces = len(faces)
    while n_faces > target_faces and heap:
        _, u, w, vu, vw = heapq.heappop(heap)
        if ver.get(u) != vu or ver.get(w) != vw or u not in vf or w not in vf:
            continue
        shared = vf[u] & vf[w]
        if len(shared) != 2:                                     # 간선이 없거나 비다양체
            continue
        nu, nw = nbrs(u), nbrs(w)
        if len(nu & nw) != 2:                                    # 링크 조건
            continue
        # 이음선 두 점 사이에 새 간선이 생기면 거울 쪽과 합쳐 간선 하나에 면 4개가 된다 → 거부.
        # (세 점 모두 이음선인 면도 같은 이유로 생기면 안 된다.)
        if locked[w] and any(locked[y] and y not in nw for y in nu if y != w):
            continue
        ok = True
        for fi in vf[u] - shared:                                 # 뒤집힘·퇴화 검사
            f = faces[fi]
            a, b, c = (v[x] for x in f)
            n0 = np.cross(b - a, c - a)
            g = [w if x == u else x for x in f]
            a, b, c = (v[x] for x in g)
            n1 = np.cross(b - a, c - a)
            l0, l1 = np.linalg.norm(n0), np.linalg.norm(n1)
            if l1 < 1e-12 or (n0 @ n1) < FLIP_DOT * l0 * l1:
                ok = False
                break
        if not ok:
            continue
        for fi in shared:
            alive_face[fi] = False
            for x in faces[fi]:
                vf[x].discard(fi)
            n_faces -= 1
        for fi in list(vf[u]):
            faces[fi] = [w if x == u else x for x in faces[fi]]
            vf[w].add(fi)
        del vf[u]
        Q[w] += Q[u]
        ver[w] += 1
        for y in nbrs(w):
            ver[y] += 1
        push_around(w)
    return np.array([f for f, a in zip(faces, alive_face) if a], dtype=np.int64)


def build(target_faces: int) -> dict:
    mesh = load_makehuman()
    v, faces, W = mesh.vertices, mesh.faces, mesh.weights
    _, _, body_idx = _read_obj_body(MAKEHUMAN_DIR / "base.obj")

    # 정점 거울 짝 (원본은 정확히 대칭)
    from scipy.spatial import cKDTree
    d, mirror = cKDTree(v).query(v * [1.0, -1.0, 1.0])
    assert d.max() < 1e-9 and (mirror[mirror] == np.arange(len(v))).all(), "원본이 거울 대칭이 아니다"

    bone_parts = np.array([bone_part(n) if W[:, i].any() else -1 for i, n in enumerate(mesh.bone_names)])
    vpart = bone_parts[np.argmax(W, axis=1)]
    seam = np.abs(v[:, 1]) < SEAM_TOL
    fy = v[faces, 1]
    left = faces[(fy >= -SEAM_TOL).all(1)]
    assert len(left) * 2 == len(faces)

    half = decimate_half(v, left, seam, vpart, target_faces // 2)
    assert not seam[half].all(axis=1).any(), "세 정점 모두 이음선인 면이 남았다"
    right = mirror[half][:, [0, 2, 1]]                           # 거울 + 방향 뒤집기 (바깥 반시계 유지)
    allf = np.concatenate([half, right])
    used = np.unique(allf)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    F = remap[allf].astype(np.int32)
    V = v[used]
    mirror_v = remap[mirror[used]].astype(np.int32)
    assert (mirror_v >= 0).all()
    nh = len(half)
    mirror_f = np.concatenate([np.arange(nh) + nh, np.arange(nh)]).astype(np.int32)
    # mirror_face[f] 의 k 번째 정점 = mirror_vertex[faces[f, perm[f,k]]]. 오른쪽 = 왼쪽 [0,2,1] 이므로 대칭.
    perm = np.tile(np.array([0, 2, 1], dtype=np.int8), (len(F), 1))
    assert (F[mirror_f] == mirror_v[F[:, [0, 2, 1]]]).all()
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all(), "닫힌 다양체가 아니다"

    # 가중치 (자르지 않음)
    Wu = W[used]
    nnz = (Wu > 0).sum(1)
    assert nnz.max() <= MAX_BONES
    order = np.argsort(-Wu, axis=1)[:, :MAX_BONES]
    wval = np.take_along_axis(Wu, order, axis=1)
    wbone = np.where(wval > 0, order, 0).astype(np.int16)

    # 면 부위: 세 정점 가중치 합의 최댓값 뼈
    fw = Wu[F].sum(axis=1)                                        # (F,B)
    face_part = bone_parts[np.argmax(fw, axis=1)].astype(np.int8)
    assert (face_part >= 0).all()
    mb = mirror_bone_index(mesh.bone_names)
    assert (face_part[mirror_f] == face_part).all(), "면 부위가 거울 대칭이 아니다"
    assert (mb[np.argmax(fw, axis=1)][mirror_f] == np.argmax(fw, axis=1)).all()

    return dict(
        vertices=V, faces=F, weight_bone=wbone, weight_val=wval,
        bone_names=np.array(mesh.bone_names), bone_parent=mesh.bone_parent.astype(np.int32),
        bone_head=mesh.bone_head, bone_tail=mesh.bone_tail,
        face_part=face_part, mirror_vertex=mirror_v, mirror_face=mirror_f, mirror_face_perm=perm,
        obj_vertex=body_idx[used].astype(np.int32),
        source=np.array(f"MakeHuman hm08 base.obj body group (CC0), QEM half-edge collapse of left half "
                        f"to {nh} faces + mirror, scripts/build_sim_mesh.py, target {target_faces}"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MakeHuman sim 메시 생성 (B)")
    ap.add_argument("--faces", type=int, default=5600, help="목표 삼각형 수 (짝수, 양쪽 합)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    data = build(args.faces)
    np.savez_compressed(args.out, **data)
    print(f"{args.out}: 정점 {len(data['vertices'])}, 면 {len(data['faces'])}, "
          f"{time.perf_counter() - t0:.1f} s, {Path(args.out).stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
