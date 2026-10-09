"""Clustered process-level uncertainty; NVBench sample noise is not a confidence interval."""
import csv,json,statistics,math,pathlib,collections,sys
path=pathlib.Path(sys.argv[1]);raw=list(csv.DictReader(path.open()));by=collections.defaultdict(dict)
for r in raw:by[(r['config'],r['group'],r['state'],r.get('stable','False'),r.get('descending','False'))][int(r['round'])]=r
rows=[]
for k,data in by.items():
 if k[0]=='baseline':continue
 ref=by.get(('baseline',)+k[1:],{});rounds=sorted(set(data)&set(ref))
 if len(rounds)<3:continue
 logs=[math.log(float(data[i]['gpu_ms'])/float(ref[i]['gpu_ms']))for i in rounds]
 n=len(logs);t={3:4.303,4:3.182,5:2.776}.get(n,1.96)
 uncertainty=t*statistics.stdev(logs)/math.sqrt(n)
 ratio=statistics.median(float(data[i]['gpu_ms'])for i in rounds)/statistics.median(float(ref[i]['gpu_ms'])for i in rounds)
 center=statistics.mean(logs);lo,hi=math.exp(center-uncertainty),math.exp(center+uncertainty)
 rows.append(dict(config=k[0],group=k[1],state=k[2],stable=k[3],descending=k[4],rounds=n,ratio=ratio,ratio_ci_low=lo,ratio_ci_high=hi,accepted=ratio<=1.02 or lo<=1<=hi,axes=data[rounds[0]]['axes']))
for c in sorted({r['config']for r in rows}):
 xs=[r for r in rows if r['config']==c];failed=[r for r in xs if not r['accepted']]
 print(c,'states',len(xs),'speedup',statistics.geometric_mean(1/r['ratio']for r in xs),'failed',len(failed))
 print(json.dumps(failed,indent=2))
with path.with_name('statistical-gate.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
