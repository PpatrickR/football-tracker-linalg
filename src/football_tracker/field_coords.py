"""Field coordinate system primitives.

All downstream football logic operates on field coords (yards), not pixels.
Once the field is registered, NFL broadcast All-22 and old college film are
interchangeable inputs to anything above this layer.

Origin: back of the left endzone at (0, 0).
  x: goal-to-goal axis, 0..120 yards (10yd endzone + 100yd field + 10yd endzone)
  y: sideline-to-sideline axis, 0..53.333 yards
"""
from dataclasses import dataclass

FIELD_LENGTH_YD = 120.0
FIELD_WIDTH_YD = 53.0 + 1.0 / 3.0  # 160 ft = 53'4" = 53.333 yd
ENDZONE_LENGTH_YD = 10.0
PLAYING_FIELD_X_MIN = ENDZONE_LENGTH_YD
PLAYING_FIELD_X_MAX = FIELD_LENGTH_YD - ENDZONE_LENGTH_YD


@dataclass(frozen=True)
class FieldPoint:
    x: float  # yards from back of left endzone
    y: float  # yards from y=0 sideline

    def in_field(self) -> bool:
        return 0.0 <= self.x <= FIELD_LENGTH_YD and 0.0 <= self.y <= FIELD_WIDTH_YD

    def in_play(self) -> bool:
        return (
            PLAYING_FIELD_X_MIN <= self.x <= PLAYING_FIELD_X_MAX
            and 0.0 <= self.y <= FIELD_WIDTH_YD
        )


def yard_lines() -> list[float]:
    """X-coords of every painted 5-yard line, including both goal lines."""
    return [PLAYING_FIELD_X_MIN + 5.0 * i for i in range(21)]


def yards_from_los(point: FieldPoint, los_x: float) -> float:
    """Signed yards along the goal-to-goal axis. Positive = past LOS."""
    return point.x - los_x


def yards_from_nearest_sideline(point: FieldPoint) -> float:
    return min(point.y, FIELD_WIDTH_YD - point.y)
