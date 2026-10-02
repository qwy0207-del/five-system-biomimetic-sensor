from pathlib import Path
import csv,math,random
root=Path(__file__).resolve().parents[1]
out=root/'demo'
(out/'raw').mkdir(parents=True,exist_ok=True)
rng=random.Random(42)
channels=['neural','metabolic','circulatory','immune','developmental']
manifest=[]
for i in range(12):
 tu=2*i/11
 sid=f'demo_{i+1:02d}'
 with (out/'raw'/f'{sid}.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time_s']+channels)
  for k in range(901):
   t=k/10
   v=[]
   for j in range(5):
    base=80+12*j+0.5*i
    response=(1-math.exp(-max(0,t-65)/(2+j)))*(4+tu*(8+2*j))
    v.append(round(base+((-1)**j)*response+0.08*math.sin(t*(0.7+0.1*j))+rng.gauss(0,0.02),8))
   w.writerow([t]+v)
 manifest.append([sid,round(tu,8),f'{sid}.csv',True])
with (out/'sample_manifest.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['sample_id','tu','csv_file','include']);w.writerows(manifest)
with (out/'prediction_manifest.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['sample_id','csv_file']);w.writerows([[x[0],x[2]] for x in manifest[:3]])
print('Created 12 simulated recordings and manifests')
