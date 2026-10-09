import json
from pathlib import Path
BUILD=Path("/home/coder/cudf/cpp/build/conda/cuda-13.3/latest")
NAMES={1:"P",2:"A",3:"B",4:"C",5:"D",6:"E"}
def read_results(path):
    data=json.loads(path.read_text()); rows={}
    for b in data['benchmarks']:
        for s in b['states']:
            values={}
            for summary in s.get('summaries',[]):
                for d in summary.get('data',[]):
                    if d['name']=='value':
                        try: values[summary['tag']]=float(d['value'])
                        except (ValueError,TypeError): pass
            value=values.get('nv/cold/time/gpu/mean')
            if value is not None:
                key=(b['name'],s['name'])
                axes={a['name']:a['value'] for a in s.get('axis_values') or []}
                rows[key]={'time':value,'noise':values.get('nv/cold/time/gpu/stdev/relative',0),
                           'memory':values.get('peak_memory_usage',0),'axes':axes,'samples':values.get('nv/cold/sample_size',0)}
            elif not s.get('is_skipped',False):
                raise RuntimeError(f'Missing GPU timing: {path.name}, {b["name"]}, {s["name"]}')
    return rows
