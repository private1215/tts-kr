# -*- coding: utf-8 -*-
"""只读核查服务器，准备但不执行缺少原始快照的后处理回放。"""
import ast
import collections
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import symtable
import textwrap

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/mnt/tts/projects/qwen3-tts"


def ssh_python(code):
    result = subprocess.run(
        ["ssh", "yuyin", "python3 -c " + shlex.quote(code)],
        check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    remote = ssh_python(r'''
import pathlib, hashlib, json, subprocess, collections, wave
root = pathlib.Path('/mnt/tts/projects/qwen3-tts')
files = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                 'source': p.read_text(errors='replace') if not p.name.startswith('._') else None}
         for p in (root/'app').glob('*.py')}
rows = list(json.loads(pathlib.Path('/mnt/tts/data/jobs.json').read_text()).values())
assets = []
base = pathlib.Path('/mnt/tts/projects/gpt-sovits/out')
for name in ['cosy_full_m.wav', 'cosy_full_f.wav', 'cosy_full_m_elmale.wav',
             'full_1a6c20a60481.txt', 'full_00b6a45ad6aa.txt']:
    p = base/name
    item = {'path': str(p), 'exists': p.is_file()}
    if p.is_file():
        item.update(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if p.suffix == '.wav':
            with wave.open(str(p)) as w:
                item.update(seconds=w.getnframes()/w.getframerate(), sr=w.getframerate(), channels=w.getnchannels())
        else:
            item['chars'] = len(p.read_text().strip())
    assets.append(item)
residual = []
for rootdir in ['/tmp', '/mnt/tts/output', '/mnt/tts/projects/qwen3-tts/tools']:
    for pattern in ['**/.asr_*.wav', '**/.enc_*.wav', '**/*replay*', '**/*regress*']:
        residual.extend(str(p) for p in pathlib.Path(rootdir).glob(pattern) if p.is_file())
legacy = []
for pattern in ['*pba*', '*selftest*', '*gap*', '*proso*', '*score_full*']:
    for p in pathlib.Path('/tmp').glob(pattern):
        legacy.append({'path': str(p), 'size': p.stat().st_size, 'directory': p.is_dir()})
experiment = json.loads(pathlib.Path('/tmp/pba_edge_experiment.json').read_text())
processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,comm,args'], text=True)
# 不把当前只读探针自身的命令源码误报为遗留任务。
processes = [l for l in processes.splitlines() if any(k in l for k in
             ['engine_score_all.py', 'server.py', 'serve.py', 'regress', 'replay'])
             and 'python3 -c' not in l and 'bash -c' not in l]
print(json.dumps({'files': files, 'assets': assets, 'residual': sorted(set(residual)),
    'legacy': legacy, 'experiment': {k: v for k,v in experiment.items() if k != 'rows'},
    'processes': processes, 'jobs': {'count': len(rows),
        'status_counts': dict(collections.Counter(r.get('status') for r in rows)),
        'latest_keys': sorted(rows[-1]),
        'internal_snapshot_keys_present': sorted({k for r in rows for k in r if k.startswith('_')})}}))
''')
    local = {p.name: p for p in (ROOT / "app").glob("*.py")}
    comparisons = []
    for name in sorted(local.keys() | remote["files"].keys()):
        lp = local.get(name)
        rp = remote["files"].get(name, {})
        lh = sha(lp.read_bytes()) if lp else None
        rh = rp.get("sha256")
        try:
            head = subprocess.check_output(
                ["git", "show", "HEAD:app/" + name], cwd=ROOT, stderr=subprocess.DEVNULL)
            hh = sha(head)
        except subprocess.CalledProcessError:
            hh = None
        category = "相同" if lh == rh else "待核实差异"
        if name.startswith("._") and lp is None:
            category = "服务端独有的 AppleDouble 元数据，非应用模块"
        if name == "pba.py" and lp:
            tree = ast.parse(lp.read_text())
            tree.body = [n for n in tree.body if not (
                isinstance(n, ast.FunctionDef) and n.name == "_fit_seg_preserve_edges")]
            equivalent = ast.dump(tree, include_attributes=False) == ast.dump(
                ast.parse(rp["source"]), include_attributes=False)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
                     and n.id == "_fit_seg_preserve_edges"]
            if equivalent and not calls:
                category = "仅本地未接入的 PBA 保边实验；去除候选函数后 AST 与生产一致"
        comparisons.append(dict(name=name, local_sha256=lh, remote_sha256=rh,
                                head_sha256=hh, equal=lh == rh, category=category,
                                deployed_but_uncommitted=bool(lh == rh and hh and lh != hh)))

    source = remote["files"]["server.py"]["source"]
    start_marker = '        if "_gain_off" not in job:'
    end_marker = '        # 变速后实际时长 = 原时长 / speed'
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError("生产后处理边界改变，拒绝提取")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    block = textwrap.dedent(source[start:end])
    wrapper = 'def replay_chunk():\n' + textwrap.indent(block, '    ')
    compile(wrapper, '<prepared-production-postprocess>', 'exec')
    table = symtable.symtable(wrapper, '<prepared-production-postprocess>', 'exec').get_children()[0]
    dependencies = sorted(s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced())
    prep = {
        "status": "prepared_not_executed", "source_sha256": sha(source.encode()),
        "source_start_line": source[:start].count('\n') + 1,
        "source_end_line": source[:end].count('\n'),
        "compile_passed": True, "external_dependencies": dependencies,
        "production_code": block,
        "safety": ["只解析源码，不导入 server，不触发初始化、推理、接口或持久化",
                   "未执行提取段；真正执行前必须注入可核验的配置、目标快照和原始逐块音频",
                   "输出及中间文件必须使用新建隔离目录；不得直接调用 run_job",
                   "生产 PROSO=0，禁止启动已取消的 engine_score_all 评分"],
        "required_snapshot": ["逐块原始音频及其哈希、对应规范化 chunk_text、块序号",
                              "job 的 seed/speed/_cosy_speed_applied 和性别、目标缓存",
                              "生产进程环境及模块常量、案例目标数据的版本和哈希",
                              "隔离 folder_path、仅内存 job、只读目标函数、错误阶段记录"],
        "scope": "性别及目标获取到 PBA、包络、slowdyn、dyn、变速、条件 PROSO、呼吸、响度、首尾、EQ/空气感、瞬态、限幅、底噪、峰值、MP3、响度闭环；不包括模型推理及 ASR",
    }
    prep_path = ROOT / 'outputs' / 'offline_postprocess_prepare.json'
    prep_path.write_text(json.dumps(prep, ensure_ascii=False, indent=2) + '\n')
    for item in remote['files'].values():
        item.pop('source', None)
    report = {
        'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'comparison': comparisons,
        'summary': {'local_modules': len(local),
                    'same_name_equal': sum(c['equal'] for c in comparisons if c['name'] in local),
                    'same_name_different': [c['name'] for c in comparisons if c['name'] in local and not c['equal']]},
        'remote_evidence': remote,
        'replay': {'prepared_manifest': str(prep_path.relative_to(ROOT)),
                   'full_chain_completed': False, 'male_completed': False, 'female_completed': False,
                   'audio_execution_performed': False,
                   'blockers': ['所检查目录未找到线上后处理前逐块 WAV，任务 wav 字段为空，临时文件已清理',
                                '持久化 jobs 未保存内部目标参数及原始分块文本快照',
                                '既有男女 cosy_full 是整篇裸推理，不能替代线上逐块同源输入',
                                '仅完成生产后处理代码提取、语法编译及依赖清单，不是可直接执行的完整回放器'],
                   'pba_90s_is_full_chain': False},
        'boundaries': {'production_modified': False, 'service_restarted': False,
                       'online_switch_changed': False, 'cancelled_scoring_started': False,
                       'git_commit_created': False, 'git_push_performed': False,
                       'ui_questions_memory_modified': False},
        'prior_agent_task_list': [
            {'id': '2', 'status': 'in_progress', 'subject': '核查完整后处理链与离线素材'},
            {'id': '3', 'status': 'in_progress', 'subject': '建立跨文案验收与反馈闭环'},
            {'id': '6', 'status': 'pending', 'subject': '取得生产目录权限后备份并部署定位题库',
             'note': '历史待办状态；用户说明已部署。本次不重新部署，不修改历史任务状态'}],
        'limitations': ['文件哈希证明磁盘同步，不能独立证明在运行进程内已加载的字节完全相同',
                        '遗留搜索覆盖 /tmp、/mnt/tts/output、项目 tools；非全盘穷举',
                        '未验证主观听感、ASR、音色相似度；未得出候选优于生产的结论'],
    }
    destination = ROOT / 'outputs' / 'server_sync_audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary'], ensure_ascii=False))
    print('准备结果已写入；完整回放未执行。')


if __name__ == '__main__':
    main()
