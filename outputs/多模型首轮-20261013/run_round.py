import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-b67eee2c-c672-6915-910d-617c62abebee'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys, json, time, hashlib, subprocess, threading, re, random, importlib.metadata
from pathlib import Path
ROOT = Path('/mnt/tts/diag/multimodel_first_20261013')
DATA = Path('/mnt/tts/data')
ENGINE = sys.argv[1]

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def active():
    j = json.loads((DATA/'jobs.json').read_text())
    return [x.get('id') for x in (j.values() if isinstance(j, dict) else j) if x.get('status') in ('running', 'queued')]

def resources():
    return {k: command(v) for k,v in {'gpu':['nvidia-smi'], 'memory':['free','-h'], 'disk':['df','-h','/mnt/tts'], 'processes':['ps','-eo','pid,etimes,pcpu,pmem,args']}.items()}

if ENGINE == 'prepare':
    assert not active(), '生产队列非空，暂停实验'
    voices = json.loads((DATA/'voices/voices.json').read_text())
    index = json.loads((DATA/'cases.json').read_text())
    cases = []
    for gender,cid,voice in [('male','d172b8c4efb3','EL_male'), ('female','ce1a3fda551a','EL_female')]:
        v = voices[voice]
        assert index[cid]['gender'] == gender and v['gender'] == gender
        assert cid not in [v.get('bench_case'), v.get('src_case')]
        source = DATA/'cases'/cid/'full.txt'
        full = source.read_text()
        # 逐字保留连续原文，以完整句号或问号结尾。
        ends = [m.end() for m in re.finditer(r'[.!?](?=\s|$)', full)]
        end = next(e for e in ends if 150 <= e <= 250)
        text = full[:end]
        assert text not in v['ref_text']
        cases.append({'gender':gender,'case_id':cid,'speaker':index[cid]['name'],'gender_evidence':'/mnt/tts/data/cases.json','source_path':str(source),'source_sha256':sha(source),'source_char_range':[0,end], 'text':text,'chars':len(text),'hangul_syllables':len(re.findall('[가-힣]',text)), 'voice':voice,'reference':{**{k:v.get(k) for k in ['path','ref_text','gender','bench_case','dur','sr']},'sha256':sha(v['path']),'ref_text_sha256':hashlib.sha256(v['ref_text'].encode()).hexdigest()},'reference_text_validation':'固定为现役 voices.json 原文；未进行新ASR或人工听验'})
    weights = Path('/mnt/tts/projects/gpt-sovits/repo/SoVITS_weights_v2ProPlus')
    inventory = [{'path':str(p),'bytes':p.stat().st_size} for p in weights.rglob('*.pth')]
    meta = {'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'cases':cases,'resources_before':resources(),'active_jobs_before':active(),'gs':{'status':'skipped_missing_active_speaker_weights','reason':'现役pick_weight目录中yohan/SOLA均缺失；_poisoned_1012及_bak_e40存在旧权重，不启用未预检/隔离版本，hh不是本轮目标说话人','inventory':inventory},'combination':{'status':'not_run','reason':'单模型尚未通过内容及人工听感验收，不投入串联'},'proso':'未启动任何评分或后处理','postprocess':'无；只将模型输出编码为WAV','production_changes':False}
    dump(ROOT/'manifest.json', meta)
    for c in cases: print(c['gender'],c['chars'],c['text'],flush=True)
    sys.exit(0)

manifest = json.loads((ROOT/'manifest.json').read_text())
assert not active(), '生产队列非空，暂停实验'
# 仅终止本进程；从不终止或暂停员工/生产任务。
def watch_production():
    while True:
        time.sleep(2)
        try:
            jobs = active()
            if jobs:
                dump(ROOT/(ENGINE+'_interrupted.json'), {'status':'interrupted_for_production','active_jobs':jobs,'at':time.time()})
                os._exit(75)
        except Exception as e:
            dump(ROOT/(ENGINE+'_guard_error.json'), {'error':repr(e)})
            os._exit(76)
threading.Thread(target=watch_production, daemon=True).start()
import numpy as np
import soundfile as sf
import torch
versions = {}
for name in ['torch','torchaudio','transformers','qwen-tts','numpy','soundfile']:
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: pass
load_start = time.time()
if ENGINE == 'qwen':
    from qwen_tts import Qwen3TTSModel
    model_path = '/mnt/tts/models/Qwen3-TTS-12Hz-1.7B-Base'
    model = Qwen3TTSModel.from_pretrained(model_path, device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    params = {'language':'Korean','non_streaming_mode':True,'x_vector_only_mode':False,'do_sample':True,'temperature':0.9,'top_k':50,'top_p':1.0,'repetition_penalty':1.1,'subtalker_dosample':True,'subtalker_temperature':0.9,'subtalker_top_k':50,'subtalker_top_p':1.0,'seed':42,'batch_size':1,'dtype':'bfloat16','attention':'sdpa','lora':False}
else:
    repo = '/mnt/tts/projects/cosy/repo'
    sys.path[:0] = [repo, repo+'/third_party/Matcha-TTS']
    from cosyvoice.cli.cosyvoice import AutoModel
    model_path = '/mnt/tts/models/Fun-CosyVoice3-0.5B'
    model = AutoModel(model_dir=model_path, fp16=False)
    params = {'seed':42,'fp16':False,'stream':False,'speed':1.0,'text_frontend':False,'prompt_prefix':'You are a helpful assistant.<|endofprompt|>','method':'inference_zero_shot','segmentation':'完整短段一次输入；不改写原文','sft':False}
load_seconds = time.time()-load_start
model_inventory = [{'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for p in Path(model_path).rglob('*') if p.is_file()]
config_hashes = {str(p):sha(p) for p in Path(model_path).glob('*') if p.suffix in ('.json','.yaml') and p.is_file()}
code_paths = ['/mnt/tts/projects/qwen3-tts/app/tts_engine.py','/mnt/tts/projects/qwen3-tts/app/cosy_server.py','/mnt/tts/projects/gpt-sovits/serve.py']
code_hashes = {p:sha(p) for p in code_paths}
for c in manifest['cases']:
    assert not active(), '生产队列非空，暂停实验'
    gender = c['gender']
    blind = {('qwen','male'):'试听-男-B7',('cosy','male'):'试听-男-A3',('qwen','female'):'试听-女-C4',('cosy','female'):'试听-女-D8'}[(ENGINE,gender)]
    dst = ROOT/'audio'/(blind+'.wav')
    assert not dst.exists(), '不覆盖已有试听音频'
    record = {'engine':'Qwen3-TTS Base' if ENGINE=='qwen' else 'CosyVoice3','backend':'独立离线裸推理进程，非生产/api/jobs','gender':gender,'case_id':c['case_id'],'blind_id':blind,'text':c['text'],'reference':c['reference'],'model_path':model_path,'model_inventory':model_inventory,'model_config_sha256':config_hashes,'source_code_sha256':code_hashes,'versions':versions,'params':dict(params),'load_seconds':load_seconds,'gpu':'RTX3090 GPU-b67eee2c-c672-6915-910d-617c62abebee','status':'running','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'resources_before':resources(),'output':str(dst),'human_listening':'not_performed','asr':'not_performed'}
    recpath = ROOT/(ENGINE+'_'+gender+'.json')
    dump(recpath, record)
    start = time.time()
    try:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        if ENGINE == 'qwen':
            cap = int(min(2048,max(64,len(c['text'])/7*12.5*2+32)))
            record['params']['max_new_tokens'] = cap
            kw = {k:v for k,v in params.items() if k not in ['seed','batch_size','dtype','attention','lora']}
            wavs,sr = model.generate_voice_clone(text=c['text'],ref_audio=c['reference']['path'],ref_text=c['reference']['ref_text'],max_new_tokens=cap,**kw)
            audio = np.asarray(wavs[0],dtype=np.float32).reshape(-1)
        else:
            outs = list(model.inference_zero_shot(c['text'],params['prompt_prefix']+c['reference']['ref_text'],c['reference']['path'],stream=False,speed=1.0,text_frontend=False))
            assert len(outs) == 1, '完整短段非流式预期一个输出；不进行拼接'
            audio = outs[0]['tts_speech'].detach().cpu().numpy().reshape(-1)
            sr = model.sample_rate
        assert len(audio) and np.isfinite(audio).all()
        sf.write(dst,audio,sr,subtype='PCM_16')
        back,rs = sf.read(dst,dtype='float32')
        duration = len(back)/rs
        peak = float(np.max(np.abs(back)))
        rms = float(np.sqrt(np.mean(back**2)))
        record.update(status='generated_valid_container',sha256=sha(dst),bytes=dst.stat().st_size,sample_rate=rs,channels=1,audio_seconds=duration,peak=peak,rms=rms,clipped_fraction=float(np.mean(np.abs(back)>=0.999)),finite=bool(np.isfinite(back).all()),chars_per_second=len(c['text'])/duration,valid_audio=bool(duration>1 and rms>0.0001),content_pass=None,listening_pass=None)
    except Exception as e:
        record.update(status='failed',error=repr(e))
    record.update(elapsed_seconds=time.time()-start,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,resources_after=resources())
    dump(recpath,record)
    print(json.dumps({k:record.get(k) for k in ['engine','gender','status','audio_seconds','elapsed_seconds','error']},ensure_ascii=False),flush=True)
