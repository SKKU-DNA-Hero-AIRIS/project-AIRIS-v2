"""정성 검증 테스트. 소유자: D. 평가기가 구현되면 skip을 제거한다.

이 테스트들은 병합 게이트다: PR 전에 로컬에서 실행하고, 깨지면 병합하지 않는다.
"""
import pytest

pytestmark = pytest.mark.skip(reason="평가기 미구현 (D: 1주차 활성화)")


def test_raising_arms_increases_armpit_removal():
    """팔을 들면 겨드랑이/옆구리 제거율이 오른다."""


def test_facing_away_reduces_front_removal():
    """노즐을 등지면 정면 제거율이 떨어진다."""


def test_closer_to_nozzle_increases_removal():
    """노즐에 가까울수록 제거율이 오른다."""


def test_zero_strength_removes_nothing():
    """노즐 세기 0이면 제거율 0."""


def test_deterministic():
    """같은 입력 → 같은 점수."""
