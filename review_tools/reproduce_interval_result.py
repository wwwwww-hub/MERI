"""Train the DS4 five-fold model and evaluate its out-of-fold intervals."""
from pathlib import Path
import argparse
import gzip
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def prepare_ds4(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for group in ['features','tracks']:
        for source in sorted((ROOT/'data/DS4_intervals'/group).glob('*.csv.gz')):
            destination=output/source.stem
            if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
                temporary=destination.with_suffix('.partial')
                with gzip.open(source,'rb') as src,temporary.open('wb') as dst:
                    shutil.copyfileobj(src,dst,8*1024*1024)
                temporary.replace(destination)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=ROOT/'working_data/DS4_intervals/features')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/intervals')
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    if a.threads<1:p.error('--threads must be positive')
    features=prepare_ds4(a.features.resolve())
    output=a.output.resolve()
    env=dict(os.environ)
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        env[name]=str(a.threads)
    annotations=ROOT/'data/DS4_intervals/annotations/confirm_milking.md'
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/train_nursing_interval_model.py'),
        'train','--features',str(features),'--annotations',str(annotations),
        '--output',str(output),'--folds','5','--bin-sec','1'],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/evaluate_nursing_interval_hybrid.py'),
        '--annotations',str(annotations),'--predictions',str(output/'oof_intervals.json'),
        '--output',str(output/'evaluation')],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
