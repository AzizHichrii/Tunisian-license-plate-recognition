from src.ocr.metrics import cer, exact_match
from src.pipeline.validator import validate


def test_valid_tun_plate():
    v = validate("123T4567")
    assert v["status"] == "valid" and v["format"] == "TUN"
    assert v["display"] == "123 تونس 4567"


def test_valid_nt_plate():
    v = validate("103679N")
    assert v["status"] == "valid" and v["format"] == "NT"


def test_invalid_and_empty():
    assert validate("12T")["status"] == "invalid"        # missing right part
    assert validate("1234T567")["status"] == "invalid"   # left part too long
    assert validate("")["status"] == "empty"


def test_metrics():
    assert exact_match(["123T4567", "1T2"], ["123T4567", "1T3"]) == 0.5
    assert cer(["123T4568"], ["123T4567"]) == 1 / 8