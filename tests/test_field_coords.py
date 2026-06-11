from football_tracker.field_coords import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    FieldPoint,
    yard_lines,
    yards_from_los,
    yards_from_nearest_sideline,
)


def test_in_field():
    assert FieldPoint(50, 26).in_field()
    assert FieldPoint(0, 0).in_field()
    assert FieldPoint(FIELD_LENGTH_YD, FIELD_WIDTH_YD).in_field()
    assert not FieldPoint(-1, 26).in_field()
    assert not FieldPoint(50, FIELD_WIDTH_YD + 1).in_field()


def test_in_play_excludes_endzones():
    assert FieldPoint(60, 26).in_play()
    assert not FieldPoint(5, 26).in_play()
    assert not FieldPoint(115, 26).in_play()


def test_yard_lines():
    lines = yard_lines()
    assert len(lines) == 21
    assert lines[0] == 10.0
    assert lines[-1] == 110.0
    diffs = [b - a for a, b in zip(lines, lines[1:])]
    assert all(abs(d - 5.0) < 1e-9 for d in diffs)


def test_yards_from_los_signed():
    los = 60.0
    assert yards_from_los(FieldPoint(65, 26), los) == 5
    assert yards_from_los(FieldPoint(55, 26), los) == -5


def test_yards_from_nearest_sideline():
    assert abs(yards_from_nearest_sideline(FieldPoint(50, 5)) - 5) < 1e-9
    assert (
        abs(yards_from_nearest_sideline(FieldPoint(50, FIELD_WIDTH_YD / 2))
            - FIELD_WIDTH_YD / 2)
        < 1e-9
    )
