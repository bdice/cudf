import sys,pathlib,csv,json,re
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from measure_r8 import BASE,BUILD,run,environment
OUT=BASE/'final-validation';OUT.mkdir(exist_ok=True);results=[]
for schedule,pattern in [(0,'StringSort.RadixLrbLargeRunProofAndSuffixRadix'),(2,'StringRadixLrbSort.*')]:
 for tool in ['memcheck','synccheck','racecheck']:
  name=f'schedule{schedule}-{tool}';p=OUT/(name+'.log')
  cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','1',str(BUILD/'gtests/SORT_TEST'),'--gtest_filter='+pattern]
  seconds=run(cmd,environment(dict(name='optimized'),schedule=schedule),p,600)
  text=p.read_text();assert 'ERROR SUMMARY: 0 errors'in text or 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'in text,(name,text[-3000:])
  results.append(dict(name=name,schedule=schedule,tool=tool,elapsed_s=seconds,errors=0))
  print('PASSED',name,flush=True)
(OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n');(OUT/'complete').write_text('complete\n')
