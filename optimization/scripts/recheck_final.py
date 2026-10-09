import sys,pathlib,csv,json,statistics,collections,random
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from measure_r8 import BASE,BUILD,environment,benchmark,run,read_results
OUT=BASE/'final-recheck';OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((BASE/'final/timings.csv').open()));tables=collections.defaultdict(dict)
for r in rows:tables[r['config']][(r['group'],r['benchmark'],r['state'],r['stable'],r['descending'])]=r
cases=[]
for k,x in tables['optimized'].items():
 ref=tables['baseline'][k];ratio=float(x['gpu_ms'])/float(ref['gpu_ms'])
 if ratio>1.02:cases.append(dict(key=k,axes=json.loads(x['axes']),ratio=ratio))
# The 262K cardinality case was a confirmed flag in the prototype; always recheck final source.
for k,x in tables['optimized'].items():
 if x['benchmark']=='sorted_order_strings_cardinality' and json.loads(x['axes'])==dict(num_rows='262144',max_width='32',cardinality='64') and not any(c['key']==k for c in cases):cases.append(dict(key=k,axes=json.loads(x['axes']),ratio=float(x['gpu_ms'])/float(tables['baseline'][k]['gpu_ms'])))
configs=[dict(name='baseline',baseline=True),dict(name='optimized')]
jobs=[(r,i,c)for r in range(5)for i in range(len(cases))for c in configs];random.Random(2451851).shuffle(jobs)
(OUT/'manifest.json').write_text(json.dumps(dict(cases=cases,jobs=jobs,samples=50,rounds=5,warmups=5),indent=2)+'\n');raw=[]
for r,i,c in jobs:
 case=cases[i];k=case['key'];axes=['-b',k[1]]
 for a,v in case['axes'].items():axes+=['-a',a+'='+v]
 stem=OUT/f'{i}-{c["name"]}-r{r}';p=stem.with_suffix('.json');run(benchmark(axes,p,50),environment(c,k[3]=='True',k[4]=='True'),stem.with_suffix('.log'),300)
 for key,v in read_results(p).items():raw.append(dict(config=c['name'],group=k[0],benchmark=k[1],state=k[2],stable=k[3],descending=k[4],round=r,gpu_ms=v['time']*1000,noise_pct=v['noise']*100,axes=json.dumps(v['axes'],sort_keys=True)))
 with(OUT/'raw.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 print('DONE',len(raw),flush=True)
(OUT/'complete').write_text('complete\n')
# Stop publication/validation of a candidate that fails its five-round regression gate.
import subprocess
subprocess.run([sys.executable,str(BASE/'statistical_gate.py'),str(OUT/'raw.csv')],check=True)
failed=[r for r in csv.DictReader((OUT/'statistical-gate.csv').open()) if r['config']=='optimized'and r['accepted']!='True']
assert not failed,failed
