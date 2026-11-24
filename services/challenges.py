from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import secrets

BASE_DIR = Path(__file__).resolve().parent.parent
CHALLENGES_DIR = BASE_DIR / "static" / "assets" / "challenges"


@dataclass(frozen=True)
class ChallengeItem:
    id: str
    code: str
    filename: str

    @property
    def file_path(self) -> Path:
        return CHALLENGES_DIR / self.filename

    @property
    def public_url(self) -> str:
        return f"/static/assets/challenges/{self.filename}"


CHALLENGE_POOL: tuple[ChallengeItem, ...] = (
    ChallengeItem(id="ch_01", code="CH-01", filename="challenge_01.jpg"),
    ChallengeItem(id="ch_02", code="CH-02", filename="challenge_02.jpg"),
    ChallengeItem(id="ch_03", code="CH-03", filename="challenge_03.jpg"),
    ChallengeItem(id="ch_04", code="CH-04", filename="challenge_04.jpg"),
    ChallengeItem(id="ch_05", code="CH-05", filename="challenge_05.jpg"),
    ChallengeItem(id="ch_06", code="CH-06", filename="challenge_06.jpg"),
    ChallengeItem(id="ch_07", code="CH-07", filename="challenge_07.jpg"),
    ChallengeItem(id="ch_08", code="CH-08", filename="challenge_08.jpg"),
    ChallengeItem(id="ch_09", code="CH-09", filename="challenge_09.jpg"),
    ChallengeItem(id="ch_10", code="CH-10", filename="challenge_10.jpg"),
)

CHALLENGE_BY_ID: dict[str, ChallengeItem] = {item.id: item for item in CHALLENGE_POOL}


def shuffle_challenge_ids() -> list[str]:
    ids = [item.id for item in CHALLENGE_POOL]
    rng = secrets.SystemRandom()
    rng.shuffle(ids)
    return ids


def get_challenge(challenge_id: str) -> ChallengeItem:
    if challenge_id not in CHALLENGE_BY_ID:
        raise KeyError(f"Unknown challenge id: {challenge_id}")
    return CHALLENGE_BY_ID[challenge_id]
