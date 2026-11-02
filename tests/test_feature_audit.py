import sys
sys.path.insert(0, "src")
import csv
import struct
import zipfile
from pathlib import Path
from audio.data.feature_audit import validate_artifact, audit

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


def test_corrupt_shape_and_unreadable_npz(tmp_path):
    spec={"format":"csv","metadata":["frame_start_sec"]}; p=tmp_path/"x.csv"
    p.write_text("frame_start_sec,a\n2,1\n1,2\n"); assert validate_artifact(p,spec)[0]=="INVALID_SHAPE"
    q=tmp_path/"bad.npz"; q.write_bytes(b"not zip"); assert validate_artifact(q,{"format":"npz","metadata":[]})[0]=="UNREADABLE"

def test_empty_axis_and_wrong_schema(tmp_path):
    p=tmp_path/"x.csv"; p.write_text("turn_id,a\n1,\n"); assert validate_artifact(p,{"format":"csv","metadata":["turn_id"]})[0]=="INVALID_DTYPE"
    p.write_text("wrong,a\n1,2\n"); assert validate_artifact(p,{"format":"csv","metadata":["turn_id"]})[0]=="SCHEMA_MISMATCH"


def test_manifest_status_aggregation(tmp_path):
    root=tmp_path/"DATA"; (root/"splits").mkdir(parents=True); (root/"extracted"/"CONTROL"/"X"/"PROFILE_1"/"VIDEO_1").mkdir(parents=True)
    (root/"splits"/"train.csv").write_text("sample_id,label_id,profile_id,subgroup,video_id,video_dir,split,cv_fold\nS,CONTROL,1,X,1,,train,0\n")
    for n in ("dev","test"): (root/"splits"/f"{n}.csv").write_text("sample_id,label_id,profile_id,subgroup,video_id,video_dir,split,cv_fold\n")
    (root/"extracted"/"CONTROL"/"X"/"PROFILE_1"/"VIDEO_1"/"f.csv").write_text("turn_id,a\n1,2\n")
    reg=tmp_path/"features.json"; reg.write_text('{"sources":{"f":{"modality":"audio","file":"f.csv","format":"csv","metadata":["turn_id"]}}}')
    out=tmp_path/"out"; result=audit({"DATA":root},reg,out)
    assert result["rows"]==1 and result["statuses"]["DATA|f|VALID"]==1


def test_npz_nonfinite_values_are_rejected(tmp_path):
    raw=b"\x93NUMPY\x01\x00"; header=b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }"; header += b" "*((16-(10+len(header))%16)%16); header += b"\n"
    q=tmp_path/"nan.npz"
    with zipfile.ZipFile(q,"w") as z: z.writestr("embed.npy",raw+struct.pack("<H",len(header))+header+struct.pack("<ff",1.0,float("nan")))
    assert validate_artifact(q,{"format":"npz","metadata":[]})[0]=="NONFINITE"

def test_split_contract_is_scoped_per_corpus(tmp_path):
    reg=tmp_path/"features.json"; reg.write_text('{"sources":{}}')
    roots={}
    for name, sample, profile in (("A","a","pa"),("B","b","pb")):
        root=tmp_path/name; (root/"splits"/"folds").mkdir(parents=True)
        head="sample_id,profile_id,split,cv_fold\n"; (root/"splits"/"train.csv").write_text(head+f"{sample},{profile},train,0\n")
        (root/"splits"/"dev.csv").write_text(head); (root/"splits"/"test.csv").write_text(head)
        for k in range(5):
            (root/"splits"/"folds"/f"fold_{k}_train.csv").write_text(head if k else head+f"{sample},{profile},train,0\n")
            (root/"splits"/"folds"/f"fold_{k}_val.csv").write_text(head)
        roots[name]=root
    result=audit(roots,reg,tmp_path/"out")
    assert result["split_contract"]["A"]["train"]["rows"]==1
    assert result["split_contract"]["B"]["train"]["rows"]==1
    assert result["split_contract"]["A"]["train"]["profiles"]==1
