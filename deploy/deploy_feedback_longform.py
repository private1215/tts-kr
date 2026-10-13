"""长稿多方案盲听题库热部署 20261013c。

沿用已验证的 20261013b 部署流程：只热更新反馈题库与 UI，不重启服务、不改语音处理、
不改服务环境变量。任一步校验失败立即回滚 items.json 与 UI。
"""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request

BUNDLE = pathlib.Path("/home/zheng/feedback_20261013c_longform")
TONE = pathlib.Path("/mnt/tts/data/tone-ab")
UI = pathlib.Path("/mnt/tts/projects/qwen3-tts/app/ui/feedback.html")
BASE = "http://127.0.0.1:8000"

AUDIO = (["c1_ref.mp3"] + ["c1_%d.mp3" % i for i in range(1, 6)]
         + ["c2_ref.mp3"] + ["c2_%d.mp3" % i for i in range(1, 6)])
MANIFEST = "manifest_20261013c_longform.json"
ITEMS_SRC = "items_20261013c.json"
NEW_FILES = AUDIO + [MANIFEST]

# 部署前线上基线，防并发变更
OLD_UI = "2ae0c61be4918442ca864ea3e4c67fa2b4d724f60823e3644491f3bf12c1a767"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def atomic(src, dst):
    tmp = dst.with_name(dst.name + ".20261013c_longform.tmp")
    with tmp.open("xb") as f:
        f.write(pathlib.Path(src).read_bytes())
        f.flush()
        os.fsync(f.fileno())
    tmp.chmod(0o644)
    os.replace(tmp, dst)


def main():
    assert os.access(TONE, os.W_OK) and os.access(UI.parent, os.W_OK), "无部署权限；未修改生产"
    assert sha(UI) == OLD_UI, "线上UI已变化，必须重新审核"

    items_text = (BUNDLE / ITEMS_SRC).read_text()
    assert MANIFEST not in items_text and "manifest" not in items_text.lower(), \
        "items 内不得引用后台 manifest"
    new_items = json.loads(items_text)["items"]

    clash = [n for n in NEW_FILES if (TONE / n).exists()]
    assert not clash, "新文件名与线上现有文件冲突，不覆盖：%s" % clash

    # ---- 备份（记录备份前每个文件 sha256）----
    before = {p.name: sha(p) for p in TONE.iterdir() if p.is_file()}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = pathlib.Path("/mnt/tts/backups") / ("feedback_" + stamp)
    backup.mkdir(mode=0o700)
    shutil.copytree(TONE, backup / "tone-ab")
    shutil.copy2(UI, backup / "feedback.html")
    pid = subprocess.check_output(["systemctl", "show", "qwen-tts", "-p", "MainPID"], text=True)
    (backup / "before.json").write_text(json.dumps(before, indent=2))
    assert sha(backup / "tone-ab/items.json") == before["items.json"], "备份与线上不一致"

    steps = []
    try:
        # ---- 写入新音频与后台 manifest ----
        for name in NEW_FILES:
            atomic(BUNDLE / name, TONE / name)
            assert sha(TONE / name) == sha(BUNDLE / name), "写入后校验失败：" + name
        steps.append("wrote_%d_new_files" % len(NEW_FILES))

        # ---- 替换前再次校验线上 items.json 未发生并发变更 ----
        assert sha(TONE / "items.json") == sha(backup / "tone-ab/items.json"), \
            "线上题库在备份后被改动，中止以防并发覆盖"

        atomic(BUNDLE / "feedback.html", UI)
        atomic(BUNDLE / ITEMS_SRC, TONE / "items.json")
        steps.append("items_and_ui_replaced")

        # ---- 部署后验证 ----
        with urllib.request.urlopen(BASE + "/api/fb/items", timeout=20) as r:
            assert "no-store" in r.headers.get("Cache-Control", ""), "items 缺少 no-store"
            assert json.load(r)["items"] == new_items, "线上题库未生效"
        steps.append("api_items_ok")

        with urllib.request.urlopen(BASE + "/feedback", timeout=20) as r:
            assert r.status == 200 and r.read() == UI.read_bytes(), "feedback 页面未生效"
        steps.append("feedback_page_ok")

        for name in AUDIO:
            req = urllib.request.Request(BASE + "/fb-audio/" + name,
                                         headers={"Range": "bytes=0-1023"})
            with urllib.request.urlopen(req, timeout=20) as r:
                assert r.status == 206, "Range 不支持（播放器寻址依赖）：" + name
        steps.append("range_206_ok_%d" % len(AUDIO))

        assert (TONE / "votes.jsonl").read_bytes().startswith(
            (backup / "tone-ab/votes.jsonl").read_bytes()), "votes.jsonl 被破坏"
        steps.append("votes_prefix_ok")

        assert pid == subprocess.check_output(
            ["systemctl", "show", "qwen-tts", "-p", "MainPID"], text=True), "服务被重启"
        steps.append("pid_unchanged")

        changed = [n for n, d in before.items() if sha(TONE / n) != d]
        assert set(changed) <= {"items.json", "votes.jsonl"}, \
            "旧资产被改动：%s" % [c for c in changed if c not in ("items.json", "votes.jsonl")]
        steps.append("old_assets_intact")

    except Exception as e:
        atomic(backup / "tone-ab/items.json", TONE / "items.json")
        atomic(backup / "feedback.html", UI)
        report = {"deployed": False, "rolled_back": True, "error": "%s: %s" % (type(e).__name__, e),
                  "steps_done": steps, "backup": str(backup)}
        (backup / "deployment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
        return 1

    report = {
        "deployed": True,
        "rollback": False,
        "backup": str(backup),
        "pid": pid.strip(),
        "steps": steps,
        "items_sha256_before": before["items.json"],
        "items_sha256_after": sha(TONE / "items.json"),
        "ui_sha256_before": OLD_UI,
        "ui_sha256_after": sha(UI),
        "new_files": {n: sha(TONE / n) for n in NEW_FILES},
        "old_files_checked": len(before),
        "restart": False,
        "votes_preserved": True,
        "old_audio_preserved": True,
    }
    (backup / "deployment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
