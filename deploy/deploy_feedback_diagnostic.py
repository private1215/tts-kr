"""以有权限账号执行；仅热更新反馈题库和UI，不重启，不改语音处理。"""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request

BUNDLE = pathlib.Path("/home/zheng/feedback_20261013b_diag")
TONE = pathlib.Path("/mnt/tts/data/tone-ab")
UI = pathlib.Path("/mnt/tts/projects/qwen3-tts/app/ui/feedback.html")
OLD_ITEMS = "a65500a0dcd2fa19cd30c6397ec4b359974bbb8440e71d824d743b9aa9b7427e"
OLD_UI = "8eb8df7541e198726cc8654f9999d9126b7b6b52441fc7d653be43f901b56eb9"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(src, dst):
    tmp = dst.with_name(dst.name + ".20261013b_diag.tmp")
    with tmp.open("xb") as f:
        f.write(src.read_bytes())
        f.flush()
        os.fsync(f.fileno())
    tmp.chmod(0o644)
    os.replace(tmp, dst)


def main():
    assert os.access(TONE, os.W_OK) and os.access(UI.parent, os.W_OK), "无部署权限；未修改生产"
    assert sha(TONE / "items.json") == OLD_ITEMS, "线上题库已变化，必须重新审核"
    assert sha(UI) == OLD_UI, "线上UI已变化，必须重新审核"
    subprocess.run([sys.executable, "/home/zheng/mk_feedback_diagnostic.py", "--out", str(BUNDLE),
                    "--validate-only"], check=True)
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    names = [a["file"] for a in manifest["audio"]] + ["items_20261013b_diag.json", "manifest_20261013b_diag.json"]
    assert all(not (TONE / name).exists() for name in names), "版本已存在，不覆盖"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = pathlib.Path("/mnt/tts/backups") / ("feedback_" + stamp)
    backup.mkdir(mode=0o700)
    before = {p.name: sha(p) for p in TONE.iterdir() if p.is_file()}
    shutil.copytree(TONE, backup / "tone-ab")
    shutil.copy2(UI, backup / "feedback.html")
    pid = subprocess.check_output(["systemctl", "show", "qwen-tts", "-p", "MainPID"], text=True)
    (backup / "before.json").write_text(json.dumps(before, indent=2))
    for a in manifest["audio"]:
        atomic(BUNDLE / a["file"], TONE / a["file"])
    atomic(BUNDLE / "manifest.json", TONE / "manifest_20261013b_diag.json")
    atomic(BUNDLE / "items_20261013b_diag.json", TONE / "items_20261013b_diag.json")
    try:
        atomic(BUNDLE / "feedback.html", UI)
        atomic(BUNDLE / "items_20261013b_diag.json", TONE / "items.json")
        with urllib.request.urlopen("http://127.0.0.1:8000/api/fb/items", timeout=15) as r:
            assert "no-store" in r.headers.get("Cache-Control", "")
            assert json.load(r)["items"] == json.loads((BUNDLE / "items_20261013b_diag.json").read_text())["items"]
        with urllib.request.urlopen("http://127.0.0.1:8000/feedback", timeout=15) as r:
            assert r.read() == UI.read_bytes()
        for a in manifest["audio"]:
            req = urllib.request.Request("http://127.0.0.1:8000/fb-audio/" + a["file"], headers={"Range": "bytes=0-1023"})
            with urllib.request.urlopen(req, timeout=15) as r:
                assert r.status == 206, "播放器寻址依赖Range支持"
        for name, digest in before.items():
            if name not in ("items.json", "votes.jsonl"):
                assert sha(TONE / name) == digest
        old_votes = (backup / "tone-ab/votes.jsonl").read_bytes()
        assert (TONE / "votes.jsonl").read_bytes().startswith(old_votes)
        assert pid == subprocess.check_output(["systemctl", "show", "qwen-tts", "-p", "MainPID"], text=True)
    except Exception:
        atomic(backup / "tone-ab/items.json", TONE / "items.json")
        atomic(backup / "feedback.html", UI)
        raise
    report = {"deployed": True, "backup": str(backup), "pid": pid.strip(),
              "votes_preserved": True, "old_audio_preserved": True, "restart": False}
    (backup / "deployment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
