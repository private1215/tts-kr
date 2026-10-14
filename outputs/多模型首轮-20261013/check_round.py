import os
os.environ['HF_HOME']='/mnt/tts/models/asr'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,'/mnt/tts/projects/qwen3-tts/app')
import asr_check
root=Path('/mnt/tts/diag/multimodel_first_20261013')
def save(p,j): p.write_text(json.dumps(j,ensure_ascii=False,indent=2))
def inspect(path,text):
    t=time.time(); hyp=asr_check.transcribe(str(path),'ko')
    a,b=asr_check.normalize(text),asr_check.normalize(hyp)
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1): row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return {'status':'completed' if hyp else 'failed_or_unavailable','model':asr_check.MODEL_ID,'device':'cpu','compute_type':'int8','local_files_only':True,'hypothesis':hyp,'normalized_reference':a,'normalized_hypothesis':b,'levenshtein_cer':previous[-1]/max(1,len(a)) if hyp else None,'existing_approx_cer':asr_check.cer(text,hyp) if hyp else None,'elapsed_seconds':time.time()-t,'note':'ASR是粗筛，非人工听验；数字格式和专有名词可能造成识别差异'}
manifest=json.loads((root/'manifest.json').read_text())
for c in manifest['cases']:
    c['reference']['asr_check']=inspect(c['reference']['path'],c['reference']['ref_text'])
records=[]
for engine in ['qwen','cosy']:
    for gender in ['male','female']:
        p=root/(engine+'_'+gender+'.json')
        if not p.exists(): continue
        r=json.loads(p.read_text())
        if r['status']=='generated_valid_container': r['asr']=inspect(r['output'],r['text'])
        save(p,r); records.append(r)
manifest['results']=records
manifest['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
manifest['resources_after']={name:subprocess.run(args,capture_output=True,text=True).stdout for name,args in {'gpu':['nvidia-smi'],'memory':['free','-h'],'disk':['df','-h','/mnt/tts'],'services':['systemctl','show','qwen-tts','cosy-tts','gs-tts','-p','ActiveState','-p','ExecMainStartTimestamp']}.items()}
j=json.loads(Path('/mnt/tts/data/jobs.json').read_text())
manifest['active_jobs_after']=[x.get('id') for x in (j.values() if isinstance(j,dict) else j) if x.get('status') in ['running','queued']]
manifest['limitations']=['仅男女各一个150/170字符短段、单seed，不代表全量效果','参考转写固定为现役配置；ASR只提供复核线索，不改写参考','未人工试听，不声明听感、音色或内容合格；不部署候选','GS对应现役权重缺失，备份与隔离权重未启用','未执行串联组合、PROSO评分或复杂后处理','实验进程每2秒检查持久化生产队列；不能排除检查间隙及未持久化直接API请求','音频目录为盲标；manifest及日志包含引擎信息，仅供后台，不能一并发给员工']
save(root/'manifest.json',manifest)
result={'status':'first_round_generated_pending_listening','success':len([r for r in records if r['status']=='generated_valid_container']),'failed':len([r for r in records if r['status']=='failed']),'items':[{k:r.get(k) for k in ['engine','gender','blind_id','status','audio_seconds','sample_rate','elapsed_seconds','peak_allocated_gib','sha256','valid_audio']}|{'asr_levenshtein_cer':r.get('asr',{}).get('levenshtein_cer')} for r in records],'gs':manifest['gs']['status'],'combination':'not_run','proso':'not_run','human_listening':'not_performed','deployed':False,'limitations':manifest['limitations']}
save(root/'result.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
