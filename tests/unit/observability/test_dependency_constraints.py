from pathlib import Path


def test_requests_dependency_ranges_are_pinned() -> None:
    requirements = (Path(__file__).resolve().parents[3] / "requirements" / "app.txt").read_text(
        encoding="utf-8"
    )
    assert "urllib3>=2.2,<3" in requirements
    assert "charset-normalizer>=3.3,<4" in requirements
    assert "chardet>=5.2,<6" in requirements
