"""生成单音频定位题；完整转码、不切片、不作跨版本语义对齐声明。"""
import argparse
import hashlib
import json
import pathlib
import subprocess

VERSION = "20261013b_diag"
SOURCES = {
    "f": "638e629a5b2dab0257f38ec40bb3f090f021dfeb1a7868286201e5a848640ded",
    "m": "37c6fa93fc23fb475fefa8a75fb11b0a315c8a61493fc699b03b21196def17b8",
}
DECISIONS = {
    "A": "本范围未发现问题，仅作定位阴性样本，不批准交付",
    "B": "按时间点复现断句或卡顿，列入修复队列",
    "C": "没听清，不计质量偏好，安排重听",
    "D": "先排查播放器缓冲或网络，不能归因语音引擎",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path)], text=True))


def dump(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate(out):
    manifest = json.loads((out / "manifest.json").read_text())
    config = json.loads((out / "items_20261013b_diag.json").read_text())
    assert sha(out / "items_20261013b_diag.json") == manifest["items_sha256"]
    assert sha(out / "feedback.html") == manifest["ui_sha256"]
    assert len(config["items"]) == 6
    assert len({i["id"] for i in config["items"]}) == 6
    assert {(i["gender"], i["region"]) for i in config["items"]} == {
        (g, r) for g in SOURCES for r in ("head", "mid", "tail")}
    for a in manifest["audio"]:
        assert sha(pathlib.Path(a["source"])) == a["source_sha256"] == SOURCES[a["gender"]]
        assert sha(out / a["file"]) == a["sha256"]
        assert abs(duration(out / a["file"]) - a["duration_seconds"]) < .15
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(out / a["file"]),
                        "-f", "null", "-"], check=True)
    for i in config["items"]:
        assert i["mode"] == "diagnostic", "未验证候选禁止进入此定位题库"
        assert i["version"] == VERSION and i["id"].endswith(VERSION)
        a = next(a for a in manifest["audio"] if a["gender"] == i["gender"])
        assert i["source_sha256"] == a["source_sha256"]
        assert i["ref"]["file"] == a["file"]
        assert 0 <= i["focus"]["start_seconds"] < i["focus"]["end_seconds"] <= a["duration_seconds"]
        assert {o["key"] for o in i["options"]} == set(DECISIONS)
        assert all(o["file"] is None for o in i["options"])
        assert i["decision_by_key"] == DECISIONS
    return {"ok": True, "items": 6, "audio_full_decode": 2,
            "semantic_alignment": "不适用：单源完整音频，未切片，建议范围不是句界",
            "ab_published": False, "delivery_approved": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--source-dir", type=pathlib.Path, default=pathlib.Path("/mnt/tts/projects/gpt-sovits/out"))
    ap.add_argument("--ui", type=pathlib.Path)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if args.validate_only:
        print(json.dumps(validate(args.out), ensure_ascii=False))
        return
    args.out.mkdir(exist_ok=False)
    assert args.ui and args.ui.is_file()
    (args.out / "feedback.html").write_bytes(args.ui.read_bytes())
    config = {"version": VERSION, "items": []}
    manifest = {
        "version": VERSION, "mode": "diagnostic_only", "audio": [],
        "latest_feedback": {"ts_utc": "2026-10-13 02:55:07", "voice_f_m": "B",
                            "smooth_f_m": "B", "ship_f_m": "B=勉强", "free": "断句有点瑕疵，卡顿问题还在"},
        "delivery_approved": False,
        "provenance": "沿用 mk_feedback_clips_d.py 的 B 源路径，生成时锁定实测 SHA256；历史投票未记录源 hash，无法追溯证明历史时刻字节未变化",
        "excluded": ["PBA90秒男声实验不是全链路男女回归", "未经验证的修复候选", "时长比例语义对齐"],
        "ab_release_gate": {"required": ["男女候选分别验证完成", "对应句转写一致且起止句界分别验证", "源与候选hash及验证报告", "非全长比例切片"],
                            "independent_options": {"C": "差不多", "D": "都不行", "E": "没听清"},
                            "default": None},
        "processing": "完整音频 MP3 转码，无切片、增益、归一化或停顿处理；不把寻址点称为语义边界",
    }
    labels = [("A", "本范围没发现明显断句或卡顿（不代表可交付）"),
              ("B", "有断句或卡顿，请记录时间点"), ("C", "没听清／拿不准，需重听"),
              ("D", "播放器转圈或加载中断，暂不评价声音")]
    for gender in SOURCES:
        src = args.source_dir / ("cosy_full_" + gender + ".wav")
        assert sha(src) == SOURCES[gender], "源音频已变化，停止生成"
        dur = duration(src)
        filename = "diag_" + VERSION + "_" + gender + "_" + SOURCES[gender][:12] + ".mp3"
        command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(src),
                   "-map", "0:a:0", "-codec:a", "libmp3lame", "-b:a", "192k", str(args.out / filename)]
        subprocess.run(command, check=True)
        manifest["audio"].append({"gender": gender, "source": str(src),
                                  "source_sha256": sha(src), "duration_seconds": dur,
                                  "file": filename, "sha256": sha(args.out / filename), "command": command})
        # 固定绝对时间寻址，不用于两个版本对齐；尾部从结束前45秒听至完整结尾。
        for region, name, start, end in [("head", "头段", 0, 45), ("mid", "中段", 190, 235),
                                         ("tail", "尾段", round(dur - 45, 3), dur)]:
            title = ("女声" if gender == "f" else "男声") + name
            config["items"].append({
                "id": "locate_" + gender + "_" + region + "_" + VERSION,
                "version": VERSION, "mode": "diagnostic", "gender": gender, "region": region,
                "dim": "断句问题定位", "title": title + "：哪里停得不对或卡住了？",
                "desc": "这是已选裸推理声音的单音频问题定位，不是跨版本同句比较，不重新评价音色。\n"
                        + "建议听原音频 %.3f–%.3f 秒；范围不是句界，播放器保留全文，请向前后拖动把句子听完整。\n" % (start, end)
                        + "听出问题请暂停，点记录时间点，补充‘句中停住／重复／断掉’；重听同一点判断是否仍出现。\n"
                        + "业务用途：本范围的问题按时间点进入复现与修复队列；没发现问题也不代表整体可交付。",
                "ref": {"file": filename, "label": title + "待定位原音频（完整上下文）",
                        "note": "点击跳到建议起点；播放器时间就是原音频时间。无需从头听完整条。"},
                "focus": {"start_seconds": start, "end_seconds": end},
                "source_sha256": SOURCES[gender],
                "options": [{"key": k, "label": v, "file": None} for k, v in labels],
                "decision_by_key": DECISIONS})
    dump(args.out / "items_20261013b_diag.json", config)
    manifest["items_sha256"] = sha(args.out / "items_20261013b_diag.json")
    manifest["ui_sha256"] = sha(args.out / "feedback.html")
    dump(args.out / "manifest.json", manifest)
    report = validate(args.out)
    dump(args.out / "validation.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
