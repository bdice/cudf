import sys,pathlib,sqlite3,csv,statistics,json,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from measure_r8 import BASE,BUILD,run,environment,benchmark,read_results
OUT=BASE/'final-profiles';OUT.mkdir(exist_ok=True)
configs=[dict(name='baseline',baseline=True),dict(name='optimized',experiment=0)]
cases={k:(['-b','sorted_order_strings_segments','-a','num_rows=2097152','-a','shared_suffix='+str(shared),'-a','segment_profile='+profile],stable)for k,profile,shared,stable in [
 ('giant-long','one_segment',64,False),('giant-short','one_segment',0,False),
 ('hot90-long','hot90',64,False),('few-giants-long','few_giants',64,False),
 ('stable-tiny32-long','tiny32',64,True),('logarithmic-long','logarithmic',64,False)]}
cases['duplicates12']=(['-b','sorted_order_strings_segmented_diagnostics','-a','num_rows=2097152','-a','profile=duplicates_12'],False)
records=[];kernels=[]
for c in configs:
 for case,(axes,stable)in cases.items():
  stem=OUT/(c['name']+'-'+case)
  cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(stem)]+benchmark(axes,stem.with_suffix('.json'),7)
  if not all(stem.with_suffix(ext).exists()for ext in ['.json','.nsys-rep','.sqlite']):
   run(cmd,environment(c,stable),stem.with_suffix('.log'),300)
   subprocess.run(['nsys','export','--type','sqlite','--force-overwrite=true','--output',str(stem.with_suffix('.sqlite')),str(stem.with_suffix('.nsys-rep'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  checked=read_results(stem.with_suffix('.json'));assert len(checked)==1 and next(iter(checked.values()))['samples']==7
  db=sqlite3.connect(stem.with_suffix('.sqlite'))
  ranges=db.execute("select e.start,e.end,e.globalTid from NVTX_EVENTS e left join StringIds s on e.textId=s.id where coalesce(e.text,s.value) in ('sorted_order','stable_sorted_order') and e.end is not null order by e.start").fetchall();assert len(ranges)>=12
  allk=db.execute('select s.value,k.start,k.end,k.correlationId,k.registersPerThread,k.staticSharedMemory,k.gridX,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id').fetchall()
  for i,(start,end,tid)in enumerate(ranges[-7:]):
   apis=db.execute('select s.value,a.correlationId,a.start,a.end from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on a.nameId=s.id where a.globalTid=? and a.start>=? and a.start<?',(tid,start,end)).fetchall();correlations={x[1]for x in apis};calls=[x for x in allk if x[3]in correlations]
   records.append(dict(config=c['name'],case=case,sample=i,kernel_count=len(calls),stream_waits=sum('cudaStreamSynchronize'in x[0]for x in apis),cpu_range_ms=(end-start)/1e6,kernel_ms=sum(x[2]-x[1]for x in calls)/1e6))
   for n,b,e,correlation,registers,shared,grid,block in calls:kernels.append(dict(config=c['name'],case=case,sample=i,kernel=n,gpu_ms=(e-b)/1e6,registers=registers,shared_bytes=shared,grid_x=grid,block_x=block))
  db.close()
for name,data in [('calls.csv',records),('kernels.csv',kernels)]:
 with(OUT/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary=[]
for c,case in sorted({(x['config'],x['case'])for x in records}):
 xs=[x for x in records if x['config']==c and x['case']==case];summary.append(dict(config=c,case=case,**{k:statistics.median(x[k]for x in xs)for k in xs[0]if k not in ['config','case','sample']}))
with(OUT/'summary.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(OUT/'manifest.json').write_text(json.dumps(dict(configs=configs,cases=cases,profile_timings_excluded_from_rankings=True),indent=2)+'\n')
(OUT/'complete').write_text('complete\n')
