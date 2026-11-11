from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,sys,time
from urllib.request import urlopen
import av,numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vlm_psychophysics import catalog,release
parser=argparse.ArgumentParser(description='Verify pinned metadata and one decoded video/static example per task and domain.')
parser.add_argument('--data-root',type=Path,default=Path('dataset'))
parser.add_argument('--report',type=Path,default=Path('outputs/release-check.json'))
args=parser.parse_args()
root=args.data_root;root.mkdir(parents=True,exist_ok=True)
rows=catalog.public_plan(root,download_missing=True)
chosen=[]
for scope in catalog.SCOPES:
 sr=[r for r in rows if r['scope']==scope and r['input']=='video']
 v=next((r for r in sr if r['task']=='T5' and len(r['parameters'].get('events',[]))==2),sr[0])
 s=next(r for r in rows if r['scope']==scope and r['input']=='static' and r['sample_id']==v['sample_id'])
 chosen.append((v,s))
media={m['path']:m for v,s in chosen for r in (v,s) for m in r['media']}
def fetch(entry):
 path,m=entry;p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  raw=p.read_bytes()
 else:
  with urlopen(f'https://huggingface.co/datasets/{release.REPO_ID}/resolve/{release.REVISION}/{path}',timeout=60) as response: raw=response.read()
 assert hashlib.sha256(raw).hexdigest()==m['sha256'],path
 if not p.exists(): p.write_bytes(raw)
 return len(raw)
t=time.monotonic()
with ThreadPoolExecutor(max_workers=4) as pool: sizes=list(pool.map(fetch,media.items()))
samples=[]
for v,s in chosen:
 path=v['source_video']['path'];expected={m['frame_index']:m for m in s['media'] if m.get('frame_index') is not None};count=0;matches=0
 with av.open(str(root/path)) as container:
  for i,frame in enumerate(container.decode(video=0)):
   assert frame.width==768 and frame.height==512
   assert abs(float(frame.pts*frame.time_base)-i/60)<1e-6
   if i in expected:
    image=np.asarray(Image.open(root/expected[i]['path']).convert('RGB')); assert np.array_equal(frame.to_ndarray(format='rgb24'),image),(v['sample_id'],i);matches+=1
   count+=1
 assert count==180 and matches==len(expected)
 samples.append({'scope':v['scope'],'sample_id':v['sample_id'],'frames':count,'decoded_static_frames_match':matches,'truth':v['truth'],'reference_first':v['task']!='T5' or s['media'][0].get('role')=='reference'})
receipt={'repo':release.REPO_ID,'revision':release.REVISION,'anonymous_media_files':len(media),'verified_media_bytes':sum(sizes),'download_and_decode_seconds':round(time.monotonic()-t,2),'decoded_video_frames':180*len(chosen),'samples':samples}
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'report':str(args.report),'samples':len(samples),'media_files':len(media),'decoded_frames':receipt['decoded_video_frames']}))
