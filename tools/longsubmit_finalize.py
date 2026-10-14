# -*- coding: utf-8 -*-
"""生成本地最终台账 manifest.json / result.json。

输入：从服务器拉下来的各阶段 json（均在 build/ 子目录）
输出：outputs/长稿多方案-20261013/manifest.json
      outputs/长稿多方案-20261013/result.json
      outputs/长稿多方案-20261013/复用资产核验记录.md
"""
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "长稿多方案-20261013")
BUILD = os.path.join(OUT, "build")
CASES = {
    "d172b8c4efb3": {"gender": "男", "label": "제주반도체"},
    "ce1a3fda551a": {"gender": "女", "label": "현대로템"},
}

# 每个案例的候选：内部代号 -> 说明（真实 sung 不体现在这里，仅台账内部用）
# blind 编号在运行时用固定种子打乱，避免位置泄露方案身份
CAND_SPEC = {
    "d172b8c4efb3": [
        {"key": "reuse_EL_cosy", "src": {"dir": "/mnt/tts/output/自测-男-真实口径-d172b8c4efb3"},
         "job": "35391b1f0e1f", "route": "线上完整链", "engine": "cosyvoice3",
         "voice": "EL_male", "reused": True},
        {"key": "reuse_yohan_cosy", "src": {"dir": "/mnt/tts/output/AB-男-参考yohan_male-d172b8c4efb3"},
         "job": "159b1c5e6a66", "route": "线上完整链", "engine": "cosyvoice3",
         "voice": "yohan_male", "reused": True},
        {"key": "online_qwen_EL", "src": {"dir": "/mnt/tts/output/长稿多方案-M1"},
         "job": "9501d27c7fd6", "route": "线上完整链", "engine": "qwen",
         "voice": "EL_male", "reused": False},
        {"key": "bare_cosy_EL", "src": {"wav": "/mnt/tts/diag/longsubmit-20261013/M3.wav"},
         "job": None, "rec_key": "M3", "route": "裸推理", "engine": "cosyvoice3",
         "voice": "EL_male", "reused": False},
        {"key": "bare_qwen_EL", "src": {"wav": "/mnt/tts/diag/longsubmit-20261013/Q1.wav"},
         "job": None, "rec_key": "Q1", "route": "裸推理", "engine": "qwen",
         "voice": "EL_male", "reused": False},
    ],
    "ce1a3fda551a": [
        {"key": "reuse_EL_cosy", "src": {"dir": "/mnt/tts/output/自测-女-真实口径-ce1a3fda551a"},
         "job": "bc8196e72fed", "route": "线上完整链", "engine": "cosyvoice3",
         "voice": "EL_female", "reused": True},
        {"key": "reuse_SOLA_cosy", "src": {"dir": "/mnt/tts/output/AB-女-参考SOLA_female-ce1a3fda551a"},
         "job": "81e99d131e9b", "route": "线上完整链", "engine": "cosyvoice3",
         "voice": "SOLA_female", "reused": True},
        {"key": "online_qwen_EL", "src": {"dir": "/mnt/tts/output/长稿多方案-M2"},
         "job": "fb76065b2f27", "route": "线上完整链", "engine": "qwen",
         "voice": "EL_female", "reused": False},
        {"key": "bare_cosy_EL", "src": {"wav": "/mnt/tts/diag/longsubmit-20261013/M4.wav"},
         "job": None, "rec_key": "M4", "route": "裸推理", "engine": "cosyvoice3",
         "voice": "EL_female", "reused": False},
        {"key": "bare_qwen_EL", "src": {"wav": "/mnt/tts/diag/longsubmit-20261013/Q2.wav"},
         "job": None, "rec_key": "Q2", "route": "裸推理", "engine": "qwen",
         "voice": "EL_female", "reused": False},
    ],
}


def load(name, default=None):
    p = os.path.join(BUILD, name)
    if not os.path.exists(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def make_blind_plan():
    """为每个案例随机打乱候选->盲标编号的映射（固定种子，可复现）。"""
    rng = random.Random(20261013)
    plan = {"cases": {}}
    mapping = {}
    for i, cid in enumerate(sorted(CAND_SPEC), start=1):
        cands = CAND_SPEC[cid]
        order = list(range(len(cands)))
        rng.shuffle(order)
        entry = {"ref_name": "case%d-ref" % i, "candidates": {}}
        for pos, src_idx in enumerate(order, start=1):
            c = cands[src_idx]
            bname = "case%d-cand%d" % (i, pos)
            entry["candidates"][bname] = c["src"]
            mapping[bname] = {"case_id": cid, **{k: v for k, v in c.items()
                                                 if k != "src"}}
        plan["cases"][cid] = entry
    return plan, mapping


def main():
    os.makedirs(OUT, exist_ok=True)
    blind_plan, mapping = make_blind_plan()
    json.dump(blind_plan, open(os.path.join(BUILD, "blind_plan.json"), "w"),
              ensure_ascii=False, indent=1)

    deliver = load("deliver_manifest.json", {})
    reuse = load("reuse_check.json", [])
    reuse_by_job = {r["job_id"]: r for r in reuse}
    asr = load("asr_screen.json", {})
    jobinfo = load("jobinfo.json", {})
    cosyinfo = load("cosyinfo.json", {})

    cases_out = {}
    rows = []
    for i, cid in enumerate(sorted(CAND_SPEC), start=1):
        dwav = os.path.join(OUT, "audio")
        cand_recs = []
        dinfo = deliver.get("cases", {}).get(cid, {})
        for bname in sorted(mapping):
            m = mapping[bname]
            if m["case_id"] != cid:
                continue
            d = next((c for c in dinfo.get("candidates", [])
                      if c.get("blind_name") == bname), {})
            local = os.path.join(dwav, bname + ".wav")
            rec = {
                "盲标": bname,
                "案例id": cid,
                "性别": CASES[cid]["gender"],
                "生产线/裸推理": m["route"],
                "引擎": m["engine"],
                "参考音色": m["voice"],
                "是否复用历史资产": m["reused"],
                "job_id": m["job"],
                "状态": d.get("status"),
                "音频时长_秒": d.get("dur_sec"),
                "音频sha256": d.get("sha256"),
                "交付文件": ("outputs/长稿多方案-20261013/audio/" + bname + ".wav"
                          if os.path.exists(local) else None),
                "源产物": d.get("source_files"),
                "源产物sha256": d.get("source_file_sha256"),
            }
            # 参考音频 / ref_text hash
            if m["reused"] and m["job"] in reuse_by_job:
                rc = reuse_by_job[m["job"]]["reference_check"]
                rec["参考音频"] = rc["ref_wav"]
                rec["参考音频sha256"] = rc["ref_wav_sha256"]
                rec["参考音频mtime"] = rc["ref_wav_mtime"]
                rec["ref_text_sha256"] = rc["ref_text_sha256"]
                rec["ref_text字数"] = rc["ref_text_len"]
                rec["voices_json_engine字段"] = rc["voices_json_engine_field"]
                rec["job_case_id"] = reuse_by_job[m["job"]]["job"]["case_id"]
                rec["job_seed"] = reuse_by_job[m["job"]]["job"]["seed"]
                rec["job_created_at"] = reuse_by_job[m["job"]]["job"]["created_at_str"]
                rec["产生耗时_秒"] = None
                rec["显存峰值_GiB"] = None
            else:
                ji = jobinfo.get(m["job"]) if m["job"] else None
                ci = cosyinfo.get(m.get("rec_key")) if m.get("rec_key") else None
                if ci:
                    rec["参考音频"] = ci.get("ref_wav")
                    rec["参考音频sha256"] = ci.get("ref_wav_sha256")
                    rec["ref_text_sha256"] = ci.get("ref_text_sha256")
                    rec["ref_text字数"] = ci.get("ref_text_len")
                    rec["job_case_id"] = ci.get("case_id")
                    rec["job_seed"] = ci.get("params", {}).get("seed")
                    rec["产生耗时_秒"] = ci.get("gen_seconds")
                    rec["显存峰值_GiB"] = ci.get("vram_gib_after")
                    rec["模型路径"] = "Fun-CosyVoice3-0.5B (/mnt/tts/models)"
                    rec["参数"] = ci.get("params")
                    rec["CUDA_VISIBLE_DEVICES"] = ci.get("cuda_visible_devices")
                if ji:
                    rec["参考音频"] = ji.get("ref_wav")
                    rec["参考音频sha256"] = ji.get("ref_wav_sha256")
                    rec["ref_text_sha256"] = ji.get("ref_text_sha256")
                    rec["ref_text字数"] = ji.get("ref_text_len")
                    rec["job_case_id"] = ji.get("case_id")
                    rec["job_seed"] = ji.get("seed")
                    rec["job_engine"] = ji.get("engine")
                    rec["job_engine_force"] = ji.get("engine_force")
                    rec["job_fallback_from"] = ji.get("fallback_from")
                    rec["产生耗时_秒"] = ji.get("elapsed_sec")
                    rec["模型路径"] = ji.get("model_path")
                    rec["参数"] = ji.get("params")
                    rec["stats"] = ji.get("stats")
                    rec["提交时间"] = ji.get("created_at_str")
            rec["文案hash_提交口径"] = None
            rec["ASR内容线索_CER"] = asr.get(bname, {}).get("cer")
            cand_recs.append(rec)
            rows.append(rec)
        # 基准
        ref = dinfo.get("ref", {})
        cases_out[cid] = {
            "案例id": cid,
            "性别": CASES[cid]["gender"],
            "标识": CASES[cid]["label"],
            "基准_EL原音频": {
                "盲标名": ref.get("blind_name"),
                "源": ref.get("source"),
                "时长_秒": ref.get("dur_sec"),
                "sha256": ref.get("sha256"),
                "交付文件": "outputs/长稿多方案-20261013/audio/" +
                            (ref.get("blind_name") or "") + ".wav",
            },
            "候选": cand_recs,
        }

    manifest = {
        "任务": "长文案多方案语音生成（首批：员工盲听筛选候选）",
        "日期": "2026-10-13",
        "产出目录": "outputs/长稿多方案-20261013/",
        "显卡分配": {
            "GPU0_RTX3060_12G": "主服务 qwen-tts(pid 1610488) 约占 4.2G，生产优先",
            "GPU1_RTX3090_24G": "cosy-tts(pid 1294) 与 gs-tts(pid 624477)；"
                                "直接客户端推理固定 CUDA_VISIBLE_DEVICES=1 串行",
        },
        "案例": cases_out,
        "盲标映射（真实身份，仅后台台账）": mapping,
        "复用资产核验": [
            {"job_id": r["job_id"], "目录": r["output_dir"], "案例": r["case_id"],
             "引擎": r["job"]["engine"], "音色": r["job"]["voice"],
             "job_case_id": r["job"]["case_id"], "seed": r["job"]["seed"],
             "job创建时间": r["job"]["created_at_str"],
             "判定": r["verdict"],
             "文案比对": r["text_check"]["note"],
             "参考音频": r["reference_check"]["ref_wav"],
             "参考音频sha256": r["reference_check"]["ref_wav_sha256"],
             "参考音频mtime": r["reference_check"]["ref_wav_mtime"],
             "参考未晚于job": r["reference_check"]["ref_wav_mtime_before_job_created"],
             "ref_text_sha256": r["reference_check"]["ref_text_sha256"],
             "产物总时长": r["total_dur_sec"], "产物数": len(r["artifacts"])}
            for r in reuse],
        "GS权重核验": load("gs_check.json", {}),
        "ASR筛查说明": "Whisper-small / CPU / int8，仅用于判断有无少读或跑飞的内容线索；"
                       "不作为音色相似度或听感结论。",
        "禁止性声明": "本文件不含任何听感通过/方案胜出的结论；"
                      "所有「谁更好」的判断须由员工盲听产生。",
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"),
              ensure_ascii=False, indent=1)

    bad = [r["盲标"] for r in rows if r["状态"] != "ok"]
    mapping_note = {b: "%s/%s/%s(%s)%s" % (m["case_id"], m["route"], m["engine"],
                                           m["voice"],
                                           " 复用" if m["reused"] else " 本轮新生成")
                    for b, m in mapping.items()}
    gs = load("gs_check.json", {})
    result = {
        "日期": "2026-10-13",
        "本轮新生成": [r["盲标"] for r in rows
                   if not r["是否复用历史资产"] and r["状态"] == "ok"],
        "复用历史资产": [r["盲标"] for r in rows if r["是否复用历史资产"]],
        "盲标->真实身份（后台）": mapping_note,
        "每条音频": [{"盲标": r["盲标"], "案例": r["案例id"],
                  "时长_秒": r["音频时长_秒"], "sha256": r["音频sha256"],
                  "ASR内容线索_CER": r["ASR内容线索_CER"],
                  "状态": r["状态"]} for r in rows],
        "基准音频": {cid: {"盲标": load("deliver_manifest.json", {})
                            .get("cases", {}).get(cid, {}).get("ref", {})
                            .get("blind_name"),
                       "时长_秒": load("deliver_manifest.json", {})
                            .get("cases", {}).get(cid, {}).get("ref", {})
                            .get("dur_sec"),
                       "sha256": load("deliver_manifest.json", {})
                            .get("cases", {}).get(cid, {}).get("ref", {})
                            .get("sha256")} for cid in sorted(CAND_SPEC)},
        "未做及原因": [
            {"项": "GS(GPT-SoVITS) 方案",
             "原因": gs.get("结论", "") + "；" + gs.get("处理", ""),
             "依据": gs.get("实测解析结果", {})},
            {"项": "组合方案（Qwen 输出转写作 Cosy 参考的真串联克隆）",
             "原因": "按用户约定须在单模型全部完成且资源允许后才做；本轮资源已用于"
                     "裸推理-Qwen 的失败重跑，且未做此项。留作下一轮，不做拼接/混音冒充。"},
        ],
        "失败与重试记录": [
            {"时间": "2026-10-13 05:52",
             "项": "裸推理-Qwen（男，首次尝试）",
             "现象": "按字面 CUDA_VISIBLE_DEVICES=1 提交后，进程实际落在生产卡 RTX3060"
                     "（nvidia-smi GPU0，主服务 pid 1610488 所在），占用约 4.5GiB，"
                     "并出现多次 CUDACachingAllocator OOM 后自动降 batch 重试。",
             "处置": "按『遇显存不足先记录并停、不得挤占生产』立即 kill 该进程，"
                     "GPU0 恢复基线 4223MiB，生产服务未受影响。",
             "根因": "本机 nvidia-smi 索引与 CUDA 序号相反："
                     "CUDA_VISIBLE_DEVICES=1 → RTX3060（生产卡），"
                     "CUDA_VISIBLE_DEVICES=0 → RTX3090（非生产卡）。"
                     "已用 torch.cuda.get_device_properties 实测确认。",
             "重试": "改用 CUDA_VISIBLE_DEVICES=0（3090）重跑，两条均成功，"
                     "显存峰值 9.47GiB，全程生产卡保持在 4223MiB 未被触碰。"},
        ],
        "生产安全核查": {
            "提交前检查次数": "每次生产提交前后均检查队列与活动任务，全程无员工排队/运行中任务",
            "Kill任何员工任务": "无",
            "重启任何服务": "无",
            "修改生产开关": "无",
            "员工多人任务": "0 条（全程 queued/running 均为 0，仅本轮自己的任务）",
        },
        "禁止性声明": "本轮不产出任何听感通过/方案胜出结论；"
                      "ASR 的 CER 仅作「有无少读/跑飞」的内容线索，"
                      "不代表音色相似度或听感优劣，方案取舍须由员工盲听决定。",
    }
    if bad:
        result["异常"] = bad
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"),
              ensure_ascii=False, indent=1)
    print("盲标映射:")
    for b in sorted(mapping):
        m = mapping[b]
        print("  %s -> %s/%s/%s(%s)" % (b, m["case_id"], m["route"],
                                        m["engine"], m["voice"]))
    print("-> manifest.json / result.json")


if __name__ == "__main__":
    main()
