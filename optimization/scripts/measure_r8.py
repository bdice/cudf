import sys,pathlib,subprocess,os,time,signal,json,csv,statistics,random,argparse
ROOT=pathlib.Path('/home/coder/cudf');BASE=ROOT/'work/string-prefix-optimizations';BUILD=ROOT/'cpp/build/conda/cuda-13.3/latest'
sys.path.insert(0,str(ROOT/'work/string-prefix-pr24498'))
from profile_common import read_results
DEADLINE=1791535200
def environment(c,stable=False,descending=False,schedule=0):
 e={k:v for k,v in os.environ.items()if not k.startswith(('LIBCUDF_STRING_SORT_','LIBCUDF_SEGMENTED_STRING_SORT_','LIBCUDF_RADIX_LRB_','CUDF_STRING_SORT_'))}
 e.update(CUDA_VISIBLE_DEVICES='GPU-bb252097-6a31-65cc-6d08-8d5e5c148bf2',LIBCUDF_STRING_SORT_ALGORITHM='3',LIBCUDF_RADIX_LRB_STRING_SORT_EXPERIMENT=str(c.get('experiment',0)),LIBCUDF_RADIX_LRB_STRING_SORT_SCHEDULE=str(schedule),CUDF_STRING_SORT_BENCH_STABLE=str(int(stable)),CUDF_STRING_SORT_BENCH_DESCENDING=str(int(descending)))
 if c.get('baseline'):e['LD_PRELOAD']=str(BUILD/'libcudftest_default_stream.so')+':'+str(BASE/'baseline-libcudf.so')
 return e
def idle():
 p=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not p,p
def run(args,e,path,timeout=180):
 assert time.time()<DEADLINE-180,'Reservation deadline';idle();start=time.time()
 with path.open('w')as f:
  child=subprocess.Popen(args,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=child.wait(timeout=min(timeout,DEADLINE-time.time()-120))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise RuntimeError('Timed out: '+str(path))
 assert rc==0,(path,rc)
 return time.time()-start
def benchmark(axes,path,samples=20):
 return [str(BUILD/'benchmarks/SORT_NVBENCH'),'--devices','0','--cold-warmup-runs','5','--min-samples',str(samples),'--stopping-criterion','sample-count','--target-samples',str(samples),'--timeout','120','--no-batch','--json',str(path)]+axes
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',default='comparison');parser.add_argument('--experiments',default='0,1,2,3');parser.add_argument('--rounds',type=int,default=3);args=parser.parse_args()
 while not(BASE/(args.phase+'-build.status')).exists():
  assert time.time()<DEADLINE-300;time.sleep(5)
 assert(BASE/(args.phase+'-build.status')).read_text().strip()=='0'
 OUT=BASE/args.phase;OUT.mkdir(exist_ok=True)
 configs=[dict(name='baseline',baseline=True,experiment=0)]+[dict(name='experiment-'+str(i),experiment=i)for i in map(int,args.experiments.split(','))]
 for c in configs:
  if c.get('baseline'):continue
  p=OUT/(c['name']+'-sort-tests.log');run([str(BUILD/'gtests/SORT_TEST')],environment(c),p,timeout=180)
  p=OUT/(c['name']+'-graph-tests.log');run([str(BUILD/'gtests/SORT_TEST'),'--gtest_filter=StringRadixLrbSort.*:StringSort.*'],environment(c,schedule=2),p,timeout=180)
 old=json.loads((ROOT/'work/string-prefix-pr24498/confirmation/manifest.json').read_text())
 groups=old['groups']
 if args.phase=='threshold':groups={k:v for k,v in groups.items()if k in ['target','segments','pr-diagnostics']}
 jobs=[(r,c,g,axes,False,False)for r in range(args.rounds)for c in configs for g,axes in groups.items()]
 warp_axes=['-b','sorted_order_strings_segments','-a','num_rows=[32768,2097152]','-a','shared_suffix=[0,64]','-a','segment_profile=[pairs,tiny4,tiny8,tiny16,tiny32]']
 for descending in [False,True]:
  jobs +=[(r,c,'stable-warps-'+str(int(descending)),warp_axes,True,descending)for r in range(args.rounds)for c in configs]
 random.Random(245183).shuffle(jobs)
 manifest=dict(phase=args.phase,configs=configs,groups=groups,jobs=jobs,rounds=args.rounds,samples=20,warmups=5,baseline_commit=json.loads((BASE/'baseline.json').read_text())['commit'])
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');raw=[]
 for r,c,g,axes,stable,descending in jobs:
  stem=OUT/f'{c["name"]}-{g}-r{r}';p=stem.with_suffix('.json');cmd=benchmark(axes,p)
  elapsed=run(cmd,environment(c,stable,descending),stem.with_suffix('.log'),timeout=300)
  results=read_results(p);assert results
  for key,v in results.items():
   assert v['samples']==20,(stem,key,v)
   raw.append(dict(config=c['name'],group=g,benchmark=key[0],state=key[1],stable=stable,descending=descending,round=r,gpu_ms=v['time']*1000,noise_pct=v['noise']*100,peak_MiB=v['memory']/2**20,axes=json.dumps(v['axes'],sort_keys=True)))
  with(OUT/'raw.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
  print('DONE',len(raw),'means',c['name'],g,'round',r,flush=True)
 rows=[];by={}
 for x in raw:by.setdefault(tuple(x[k]for k in ['config','group','benchmark','state','stable','descending']),[]).append(x)
 for k,xs in by.items():
  assert len(xs)==args.rounds
  rows.append(dict(zip(['config','group','benchmark','state','stable','descending'],k),gpu_ms=statistics.median(x['gpu_ms']for x in xs),round_min_ms=min(x['gpu_ms']for x in xs),round_max_ms=max(x['gpu_ms']for x in xs),noise_max_pct=max(x['noise_pct']for x in xs),peak_MiB=max(x['peak_MiB']for x in xs),axes=xs[0]['axes']))
 with(OUT/'timings.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'complete').write_text('complete\n');print('COMPLETE',len(raw),'means',flush=True)
if __name__=='__main__':main()
