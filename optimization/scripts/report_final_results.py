import pathlib,json,csv,statistics,collections,math
ROOT=pathlib.Path('/home/coder/cudf');BASE=ROOT/'work/string-prefix-optimizations';OUT=BASE/'deliverables';OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((BASE/'final/timings.csv').open()));T=collections.defaultdict(dict)
for r in rows:T[r['config']][(r['group'],r['benchmark'],r['state'],r['stable'],r['descending'])]=r
optimized=T['optimized'];baseline=T['baseline']
primary=[k for k in optimized if k[3:]==('False','False')];stable=[k for k in optimized if k[3]=='True']
old=list(csv.DictReader((ROOT/'work/string-prefix-pr24498/confirmation/timings.csv').open()))
old_tables=collections.defaultdict(dict)
for r in old:
 if r['phase']=='main':old_tables[r['config']][(r['group'],r['benchmark'],r['state'])]=r
old_prefix={(r['group'],r['benchmark'],r['state']):r for r in old if r['phase']=='main' and r['config']=='prefix'}
active=[k for k in primary if k[0]in ['main','target'] and float(old_prefix[k[:3]]['peak_MiB'])*2**20>int(json.loads(optimized[k]['axes'])['num_rows'])*4+1024]
segments=[k for k in primary if k[0]=='segments'];pr=[k for k in primary if k[0].startswith('pr-')];end=[k for k in primary if k[0]=='end-to-end']
pools={'original-active':active,'segments':segments,'pr-new':pr,'combined':active+segments+pr,'end-to-end':end,'all-primary':primary,'stable-warps':stable}
assert len(active)==68 and len(active+segments+pr)==206 and len(end)==16
summary=[];wins=[]
for c in ['baseline','prefix','pr1-p32-c512','pr2-p32-c4096']:
 for pool,ks in pools.items():
  if not set(ks)<=set(T[c]):continue
  ratios=[float(T[c][k]['gpu_ms'])/float(optimized[k]['gpu_ms'])for k in ks]
  summary.append(dict(config=c,pool=pool,states=len(ks),time_over_optimized=statistics.geometric_mean(ratios),optimized_faster_over_2=sum(v>1.02 for v in ratios),other_faster_over_2=sum(v<.98 for v in ratios),within_2=sum(.98<=v<=1.02 for v in ratios)))
 if c=='baseline':continue
 for k in pools['combined']:
  old_win=float(T[c][k]['gpu_ms'])>1.02*float(baseline[k]['gpu_ms'])
  new_win=float(T[c][k]['gpu_ms'])>float(optimized[k]['gpu_ms'])
  published_win=float(old_tables[c][k[:3]]['gpu_ms'])>1.02*float(old_tables['radix-lrb'][k[:3]]['gpu_ms'])
  wins.append(dict(comparison=c,group=k[0],benchmark=k[1],state=k[2],baseline_win_over_2=old_win,published_win_over_2=published_win,preserved=new_win,baseline_ms=baseline[k]['gpu_ms'],optimized_ms=optimized[k]['gpu_ms'],other_ms=T[c][k]['gpu_ms'],axes=optimized[k]['axes']))
for fn,data in [('summary.csv',summary),('win-preservation.csv',wins)]:
 with(OUT/fn).open('w')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
result=dict(pools={k:len(v)for k,v in pools.items()},lost_wins=[r for r in wins if r['baseline_win_over_2'] and not r['preserved']],lost_published_wins=[r for r in wins if r['published_win_over_2'] and not r['preserved']],summary=summary)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=['#687887','#C98925','#BF4F69','#267B55']
labels={'baseline':'Previous R8 LRB (74dd068)','prefix':'Prefix merge (0)','pr1-p32-c512':'PR segmented (1; p32, cutoff512)','pr2-p32-c4096':'PR segmented + duplicates (2; p32, cutoff4096)'}
plot_pools=['original-active','segments','pr-new','combined','end-to-end'];fig,ax=plt.subplots(figsize=(11.5,5))
for j,c in enumerate(labels):
 v=[next(float(x['time_over_optimized'])for x in summary if x['config']==c and x['pool']==p)for p in plot_pools]
 ax.bar([i+(j-1.5)*.18 for i in range(len(v))],v,.18,color=colors[j],label=labels[c])
ax.axhline(1,color='black',lw=.8,label='Optimized R8 = 1');ax.set_xticks(range(len(plot_pools)),['Original active\n68 states','Controlled runs\n48 states','PR input families\n90 states','Combined permutation\n206 states','End-to-end\n16 states']);ax.set_ylabel('Geometric mean time / optimized R8 (lower is faster)');ax.set_title('H100: fixed R8 recipe versus saved R8 and unchanged PR backends');ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'optimization-ranking.png',dpi=170);fig.savefig(OUT/'optimization-ranking.svg');plt.close(fig)
cases=[('Giant, short','one_segment','0',False),('Giant, long','one_segment','64',False),('Hot90, long','hot90','64',False),('Few giants, long','few_giants','64',False),('Stable tiny32, long','tiny32','64',True)]
fig,ax=plt.subplots(figsize=(11.5,4.8))
for j,c in enumerate(['baseline','optimized']):
 values=[]
 for title,profile,shared,isstable in cases:
  match=[r for k,r in T[c].items()if k[0]==('stable-warps-0'if isstable else'segments') and json.loads(r['axes']).get('num_rows')=='2097152'and json.loads(r['axes']).get('segment_profile')==profile and json.loads(r['axes']).get('shared_suffix')==shared]
  assert len(match)==1,(c,profile,len(match));values.append(float(match[0]['gpu_ms']))
 ax.bar([i+(j-.5)*.32 for i in range(len(values))],values,.32,label='Previous R8'if j==0 else'Optimized R8',color=['#687887','#267B55'][j])
 for i,v in enumerate(values):ax.text(i+(j-.5)*.32,v*1.03,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
ax.set_yscale('log');ax.set_ylim(.4,30);ax.set_xticks(range(len(cases)),[x[0]for x in cases]);ax.set_ylabel('GPU-event mean, ms (log scale)');ax.set_title('2,097,152 rows: repeated unprofiled measurements');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'optimization-cases.png',dpi=170);fig.savefig(OUT/'optimization-cases.svg');plt.close(fig)
