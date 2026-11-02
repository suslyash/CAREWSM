import csv
from pathlib import Path
from src.audio.data.feature_audit import validate_artifact

def test_missing_empty_and_valid_csv(tmp_path):
    spec={"format":"csv","metadata":["turn_id"]}
    assert validate_artifact(tmp_path/"x.csv",spec)[0]=="MISSING"
    (tmp_path/"x.csv").write_text("turn_id,a\n")
    assert validate_artifact(tmp_path/"x.csv",spec)[0]=="EMPTY"
    (tmp_path/"x.csv").write_text("turn_id,a\n1,2\n")
    assert validate_artifact(tmp_path/"x.csv",spec)[0]=="VALID"

def test_bad_csv_statuses(tmp_path):
    spec={"format":"csv","metadata":["turn_id"]}; p=tmp_path/"x.csv"
    p.write_text("wrong,a\n1,2\n"); assert validate_artifact(p,spec)[0]=="SCHEMA_MISMATCH"
    p.write_text("turn_id,a\n1,nan\n"); assert validate_artifact(p,spec)[0]=="NONFINITE"
    p.write_text("turn_id,a\n1,abc\n"); assert validate_artifact(p,spec)[0]=="INVALID_DTYPE"
