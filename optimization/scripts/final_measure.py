import sys,pathlib,json,csv,random,statistics,collections,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from measure_r8 import BASE,ROOT,BUILD,environment,run,benchmark,read_results
OUT=BASE/'final';OUT.mkdir(exist_ok=True)
configs=[dict(name='baseline',baseline=True),dict(name='optimized'),dict(name='prefix',algorithm=0),dict(name='pr1-p32-c512',algorithm=1,cutoff=512),dict(name='pr2-p32-c4096',algorithm=2,cutoff=4096)]
def env(c,stable=False,descending=False,schedule=0):
 e=environment(c,stable,descending,schedule)
 e.update(LIBCUDF_STRING_SORT_ALGORITHM=str(c.get('algorithm',3)),LIBCUDF_SEGMENTED_STRING_SORT_LEXIC_PRECISION='32',LIBCUDF_SEGMENTED_STRING_SORT_RADIX_RUN_MIN=str(c.get('cutoff',512)))
 return e
for c in configs:
 if c.get('baseline'):continue
 run([str(BUILD/'gtests/SORT_TEST')],env(c),OUT/(c['name']+'-sort-tests.log'),600)
run([str(BUILD/'gtests/SORT_TEST'),'--gtest_filter=StringRadixLrbSort.*:StringSort.*'],env(dict(name='optimized'),schedule=2),OUT/'optimized-graph-tests.log',600)
groups=json.loads((ROOT/'work/string-prefix-pr24498/confirmation/manifest.json').read_text())['groups']
warp=['-b','sorted_order_strings_segments','-a','num_rows=[32768,2097152]','-a','shared_suffix=[0,64]','-a','segment_profile=[pairs,tiny4,tiny8,tiny16,tiny32]']
jobs=[(r,c,g,axes,False,False)for r in range(3)for c in configs for g,axes in groups.items()]
# The supplement compares the optimized and saved R8 implementations, separate from the PR ranking.
for descending in [False,True]:jobs.extend((r,c,'stable-warps-'+str(int(descending)),warp,True,descending)for r in range(3)for c in configs[:2])
random.Random(245189).shuffle(jobs)
(OUT/'manifest.json').write_text(json.dumps(dict(configs=configs,groups=groups,jobs=jobs,rounds=3,samples=20,warmups=5),indent=2)+'\n')
raw=[]
for r,c,g,axes,stable,descending in jobs:
 stem=OUT/f'{c["name"]}-{g}-r{r}';p=stem.with_suffix('.json')
 run(benchmark(axes,p),env(c,stable,descending),stem.with_suffix('.log'),600)
 for key,v in read_results(p).items():
  assert v['samples']==20
  raw.append(dict(config=c['name'],group=g,benchmark=key[0],state=key[1],stable=stable,descending=descending,round=r,gpu_ms=v['time']*1000,noise_pct=v['noise']*100,peak_MiB=v['memory']/2**20,axes=json.dumps(v['axes'],sort_keys=True)))
 with(OUT/'raw.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 print('DONE',len(raw),'means',c['name'],g,'round',r,flush=True)
by=collections.defaultdict(list)
for x in raw:by[tuple(x[k]for k in ['config','group','benchmark','state','stable','descending'])].append(x)
rows=[]
for k,xs in by.items():
 assert len(xs)==3
 rows.append(dict(zip(['config','group','benchmark','state','stable','descending'],k),gpu_ms=statistics.median(x['gpu_ms']for x in xs),round_min_ms=min(x['gpu_ms']for x in xs),round_max_ms=max(x['gpu_ms']for x in xs),noise_max_pct=max(x['noise_pct']for x in xs),peak_MiB=max(x['peak_MiB']for x in xs),axes=xs[0]['axes']))
with(OUT/'timings.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'complete').write_text('complete\n')
