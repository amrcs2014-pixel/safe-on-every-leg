from dataclasses import dataclass


@dataclass(frozen=True)
class Vehicle:
    """Cargo space on two supports; x runs from the front wall to the rear door, lengths in m, masses in t."""

    length: float
    width: float
    height: float
    a_front: float
    a_rear: float
    tare_front: float
    tare_rear: float
    max_front: float
    max_rear: float
    track: float | None = None
    deck_height: float | None = None
    tare_cg_height: float | None = None
    lateral_accel: float = 0.5
    min_share: float = 0.0
    min_share_support: str = "rear"

    def __post_init__(self):
        if self.a_rear <= self.a_front:
            raise ValueError("rear support must lie behind the front support")
        if self.min_share_support not in ("front", "rear"):
            raise ValueError("min_share_support must be 'front' or 'rear'")

    @property
    def wheelbase(self) -> float:
        return self.a_rear - self.a_front

    @property
    def tare(self) -> float:
        return self.tare_front + self.tare_rear
