"""Automatic ZIP discovery, EMF construction and train-only normalization.

Archive convention:
  excel/<subject>/<character>.xlsx
  excel/<subject>/<character>_trial001.xlsx
  excel/<subject>/<character>-02.xls

The character split is read from imu_split_config.json. No CSV manifest is used.
First 3 channels: gravity-removed linear acceleration; last 3: gyroscope.
"""
from __future__ import annotations
import hashlib, io, json, logging, re, zipfile, zlib
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)
EPS = 1e-8


class _DirectoryReader:
    """Small read-only adapter so directory files share the ZIP reader interface."""
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, member):
        path = (self.root / Path(member)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Path escapes data directory: {member}")
        return path.read_bytes()


@contextmanager
def _open_source(source):
    """Open either an extracted data directory or a ZIP archive."""
    path = Path(source)
    if path.is_dir():
        yield _DirectoryReader(path)
    else:
        with zipfile.ZipFile(path) as zf:
            yield zf


def load_split_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    required = ('train_characters', 'val_characters', 'test_characters')
    if not all(k in cfg and isinstance(cfg[k], list) for k in required):
        raise ValueError(f'{path} must contain {required}')
    groups = {k: set(cfg[k]) for k in required}
    if any(not x for x in groups.values()):
        raise ValueError('Each train/val/test character list must be non-empty')
    if groups['train_characters'] & groups['val_characters'] or groups['train_characters'] & groups['test_characters'] or groups['val_characters'] & groups['test_characters']:
        raise ValueError('Character lists in split config overlap')
    return cfg


def _identify_character(stem: str, known_chars):
    """Match exact character or a trial suffix. Recommended suffix: _trial001."""
    for char in sorted(known_chars, key=len, reverse=True):
        if stem == char:
            return char
        if stem.startswith(char):
            suffix = stem[len(char):]
            # Require a delimiter or digit so a different character name is not
            # silently interpreted as a repetition of this character.
            if re.match(r'^(?:[_\-.\s（(第]|\d)', suffix):
                return char
    return None


def discover_records(archive, split_config):
    """Discover xls/xlsx recordings from an extracted directory or ZIP archive."""
    cfg = load_split_config(split_config) if isinstance(split_config, (str, Path)) else split_config
    split_by_char = {}
    for split, key in [('train','train_characters'), ('val','val_characters'), ('test','test_characters')]:
        for c in cfg[key]:
            split_by_char[c] = split
    known = set(split_by_char)
    records, ignored, entries = [], [], []
    source = Path(archive)

    if source.is_dir():
        # The supplied directory is the data root, e.g. /root/autodl-tmp/excel.
        # Expected layout: <data_root>/<subject>/<character>[_trialNNN].xlsx
        for file_path in sorted(source.rglob('*')):
            if not file_path.is_file() or file_path.suffix.lower() not in ('.xlsx', '.xls'):
                continue
            rel_path = file_path.relative_to(source).as_posix()
            parts = rel_path.split('/')
            if len(parts) < 2:
                ignored.append({'path': rel_path, 'reason': 'not under <data_root>/<subject>/'})
                continue
            raw = file_path.read_bytes()
            entries.append((rel_path, Path(rel_path), parts[0], len(raw), zlib.crc32(raw) & 0xffffffff))
    else:
        with zipfile.ZipFile(source) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                normalized = info.filename.replace('\\', '/')
                p = Path(normalized)
                if p.suffix.lower() not in ('.xlsx', '.xls'):
                    continue
                parts = normalized.split('/')
                if len(parts) < 3 or parts[0] != 'excel':
                    ignored.append({'path': normalized, 'reason': 'not under excel/<subject>/'})
                    continue
                entries.append((normalized, p, parts[1], info.file_size, info.CRC))

    for archive_path, p, subject, size, crc in entries:
        char = _identify_character(p.stem, known)
        if char is None:
            ignored.append({'path': archive_path, 'reason': 'filename does not match a character in split config'})
            continue
        records.append({'subject': subject, 'character': char, 'split': split_by_char[char],
                        'archive_path': archive_path, 'format': p.suffix.lower().lstrip('.'),
                        'size': size, 'crc': crc})

    if not records:
        raise ValueError('No records discovered. For a directory use <data_root>/<subject>/<character>[_trialNNN].xlsx; for ZIP use excel/<subject>/<character>[_trialNNN].xlsx')
    records.sort(key=lambda r: (r['subject'], r['character'], r['archive_path']))
    paths = [r['archive_path'] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate data paths detected')
    subjects = sorted({r['subject'] for r in records})
    for subject in subjects:
        have = {r['split'] for r in records if r['subject'] == subject}
        missing = {'train', 'val', 'test'} - have
        if missing:
            raise ValueError(f'Subject {subject} has no records in splits {sorted(missing)}')
    return records, ignored, subjects

def _numeric_table(frame):
    a=frame.apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
    if a.size==0: raise ValueError('Empty numeric spreadsheet')
    a=a[~np.isnan(a).all(axis=1)]
    keep=np.isfinite(a).sum(axis=0)>=max(2,int(.8*len(a)))
    a=a[:,keep]
    a=a[np.isfinite(a).all(axis=1)]
    if a.shape[1]<3 or len(a)<2: raise ValueError(f'Insufficient numeric cells: {a.shape}')
    return a


def _unit_to_seconds(t, sample_rate):
    d=np.diff(t); positive=d[np.isfinite(d)&(d>0)]
    if not len(positive): return np.arange(len(t),dtype=np.float64)/sample_rate
    med=np.median(positive); scales=(1.,1e-3,1e-6,1e-9)
    scale=min(scales,key=lambda v:abs(np.log(max(med*v*sample_rate,EPS))))
    return (t-t[0])*scale


def _extract_time_signal(arr, sample_rate):
    if arr.shape[1]<4:
        return np.arange(len(arr),dtype=np.float64)/sample_rate, arr[:,:3]
    t=arr[:,0]
    if np.mean(np.diff(t)>0)<.85:
        t=arr[:,-1]
        if np.mean(np.diff(t)>0)<.85: raise ValueError('Cannot recognize timestamp column')
        sig=arr[:,:3]
    else: sig=arr[:,1:4]
    return _unit_to_seconds(t,sample_rate),sig


def read_record(zf, archive_path, sample_rate):
    raw=zf.read(archive_path); suffix=Path(archive_path).suffix.lower()
    if suffix=='.xlsx':
        arr=_numeric_table(pd.read_excel(io.BytesIO(raw),header=None,engine='openpyxl'))
        if arr.shape[1]<6: raise ValueError(f'Expected six columns: {archive_path}, got {arr.shape}')
        x=arr[:,:6]; t=np.arange(len(x),dtype=np.float64)/sample_rate
    elif suffix=='.xls':
        book=pd.ExcelFile(io.BytesIO(raw),engine='xlrd'); names=book.sheet_names
        if len(names)<2: raise ValueError('Legacy xls requires separate accelerometer and gyroscope sheets')
        def score(name, words): return max([int(w in name.lower()) for w in words] or [0])
        ai=next((i for i,n in enumerate(names) if score(n,('acc','加速','线性'))),0)
        gi=next((i for i,n in enumerate(names) if score(n,('gyro','角速','陀螺'))),1 if ai==0 else 0)
        if ai==gi: raise ValueError(f'Cannot identify accel/gyro sheets: {names}')
        ta,xa=_extract_time_signal(_numeric_table(pd.read_excel(book,sheet_name=ai,header=None)),sample_rate)
        tg,xg=_extract_time_signal(_numeric_table(pd.read_excel(book,sheet_name=gi,header=None)),sample_rate)
        ia=np.unique(ta,return_index=True)[1]; ig=np.unique(tg,return_index=True)[1]
        ta,xa=ta[np.sort(ia)],xa[np.sort(ia)]; tg,xg=tg[np.sort(ig)],xg[np.sort(ig)]
        if not(np.all(np.diff(ta)>0) and np.all(np.diff(tg)>0)): raise ValueError('Non-monotonic timestamps')
        start,end=max(ta[0],tg[0]),min(ta[-1],tg[-1]); mask=(ta>=start)&(ta<=end); t=ta[mask]
        if len(t)<2: raise ValueError('Accel/gyro timestamps do not overlap')
        a=xa[mask]; g=np.column_stack([np.interp(t,tg,xg[:,j]) for j in range(3)])
        x=np.concatenate([a,g],axis=1); t=t-t[0]
    else: raise ValueError(f'Unsupported format {suffix}')
    if not np.isfinite(x).all(): raise ValueError(f'Non-finite values in {archive_path}')
    return x.astype(np.float32),t.astype(np.float64)


def _integrate(a,t,detrend=False):
    aa=a-a.mean(0,keepdims=True) if detrend else a
    dt=np.diff(t).clip(min=1e-5)[:,None]
    inc=.5*(aa[1:]+aa[:-1])*dt
    return np.concatenate([np.zeros((1,aa.shape[1])),np.cumsum(inc,axis=0)],axis=0)


def _differentiate(a,t):
    return np.gradient(a,t,axis=0) if len(a)>=3 else np.zeros_like(a)


def _spearman(x,y):
    if np.std(x)<1e-8 or np.std(y)<1e-8: return 0.
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])


def _global_motion(x,t,input_mode):
    blocks=[x[:,:3]]
    if input_mode=='accgyro6': blocks.append(x[:,3:6])
    result=[]; fs=(len(t)-1)/max(t[-1]-t[0],EPS)
    bands=(0,2,5,15,50)
    for block in blocks:
        for j in range(3):
            sig=block[:,j]; power=np.abs(np.fft.rfft(sig-sig.mean()))**2
            freq=np.fft.rfftfreq(len(sig),d=1/max(fs,EPS)); total=max(power.sum(),EPS)
            result.append(float(np.dot(freq,power)/total))
            for lo,hi in zip(bands[:-1],bands[1:]):
                mask=(freq>=lo)&((freq<hi) if hi<bands[-1] else (freq<=hi))
                result.append(float(power[mask].sum()/total))
        cov=np.cov(block.T,bias=True); result.extend(cov[np.triu_indices(3)].tolist())
        result.extend([_spearman(block[:,i],block[:,j]) for i,j in ((0,1),(0,2),(1,2))])
    if input_mode=='accgyro6': result.extend(np.cov(x[:,:6].T,bias=True)[:3,3:6].ravel().tolist())
    result.append(float(np.log1p(max(t[-1]-t[0],0))))
    return np.asarray(result,dtype=np.float32)


def build_features(x,t,input_mode='accgyro6',seq_len=256,velocity_detrend=False):
    if len(x)<4 or len(np.unique(t))<4: raise ValueError('Record too short for EMF')
    a,w=x[:,:3].astype(np.float64),x[:,3:6].astype(np.float64)
    v=_integrate(a,t,velocity_detrend); j=_differentiate(a,t)
    stacks=[a,v,j,np.linalg.norm(a,axis=1,keepdims=True),np.linalg.norm(v,axis=1,keepdims=True),np.linalg.norm(j,axis=1,keepdims=True)]
    if input_mode=='accgyro6':
        aw=_differentiate(w,t)
        stacks += [w,aw,np.linalg.norm(w,axis=1,keepdims=True),np.linalg.norm(aw,axis=1,keepdims=True)]
    seq=np.concatenate(stacks,axis=1)
    speed=float(np.linalg.norm(v,axis=1).mean())
    dst=np.linspace(t[0],t[-1],seq_len)
    seq=np.stack([np.interp(dst,t,seq[:,j]) for j in range(seq.shape[1])],axis=1).astype(np.float32)
    glob=_global_motion(x,t,input_mode)
    if not(np.isfinite(seq).all() and np.isfinite(glob).all() and np.isfinite(speed)): raise ValueError('Non-finite EMF')
    return seq,glob,speed


def fit_stats(items):
    seq=np.stack([x['seq'] for x in items]); glob=np.stack([x['global'] for x in items])
    return {'seq_mean':seq.mean((0,1)).reshape(1,-1),'seq_std':np.maximum(seq.std((0,1)).reshape(1,-1),1e-4),
            'global_mean':glob.mean(0),'global_std':np.maximum(glob.std(0),1e-4)}


class FeatureDataset(Dataset):
    def __init__(self,items,stats,augment=False,jitter_std=0.): self.items,self.stats,self.augment,self.jitter_std=items,stats,augment,jitter_std
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        s=self.items[i]; seq=(s['seq']-self.stats['seq_mean'])/self.stats['seq_std']; glob=(s['global']-self.stats['global_mean'])/self.stats['global_std']
        if self.augment and self.jitter_std: seq=seq+np.random.normal(0,self.jitter_std,seq.shape).astype(np.float32)
        return {'seq':torch.tensor(seq,dtype=torch.float32),'global':torch.tensor(glob,dtype=torch.float32),
                'identity':torch.tensor(s['identity'],dtype=torch.long),'velocity':torch.tensor(s['velocity'],dtype=torch.long),
                'path':s['path'],'character':s['character'],'subject':s['subject']}


def prepare_data(archive, split_config, cache_dir, *, input_mode='accgyro6',sample_rate=100.,seq_len=256,
                 min_length=16,velocity_detrend=False,exclude_subjects=(),rebuild_cache=False):
    cfg=load_split_config(split_config); records,ignored,subjects=discover_records(archive,cfg)
    records=[r for r in records if r['subject'] not in set(exclude_subjects)]
    subjects=sorted({r['subject'] for r in records})
    settings={'input_mode':input_mode,'sample_rate':sample_rate,'seq_len':seq_len,'min_length':min_length,
              'velocity_detrend':velocity_detrend,'exclude_subjects':sorted(exclude_subjects)}
    # Include archive member CRCs and split-config contents so adding trials invalidates cache automatically.
    fingerprint={'records':[(r['archive_path'],r['size'],r['crc'],r['split']) for r in records],
                 'config':{k:cfg[k] for k in ('train_characters','val_characters','test_characters')},'settings':settings}
    key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True); cache=cache_dir/f'features_{key}.pt'
    if cache.exists() and not rebuild_cache: return torch.load(cache,map_location='cpu',weights_only=False)
    ids={s:i for i,s in enumerate(subjects)}; data={'train':[],'val':[],'test':[]}; skipped=[]
    with _open_source(archive) as zf:
        for r in records:
            try:
                x,t=read_record(zf,r['archive_path'],sample_rate)
                if len(x)<min_length: raise ValueError(f'too_short: {len(x)} < {min_length}')
                seq,glob,speed=build_features(x,t,input_mode,seq_len,velocity_detrend)
                data[r['split']].append({'seq':seq,'global':glob,'speed':speed,'identity':ids[r['subject']],
                    'subject':r['subject'],'character':r['character'],'path':r['archive_path']})
            except Exception as e:
                LOG.warning('Skipping %s: %s',r['archive_path'],e); skipped.append({'path':r['archive_path'],'reason':str(e)})
    if any(not data[s] for s in data): raise ValueError(f'Empty split after QC: { {k:len(v) for k,v in data.items()} }')
    # Every enrolled identity must have training examples. Validation/test may be
    # incomplete during ongoing collection, but their reported class support is explicit.
    found_train={x['subject'] for x in data['train']}
    missing=set(subjects)-found_train
    if missing: raise ValueError(f'Train split missing subjects: {sorted(missing)}')
    speeds=np.asarray([x['speed'] for x in data['train']]); q1,q2=np.quantile(speeds,[1/3,2/3]).tolist()
    for items in data.values():
        for x in items: x['velocity']=int(np.searchsorted([q1,q2],x['speed'],side='right'))
    stats=fit_stats(data['train'])
    payload={'data':data,'stats':stats,'subjects':subjects,'velocity_thresholds':[q1,q2],
             'settings':settings,'records_discovered':len(records),'ignored':ignored,'skipped':skipped,
             'split_counts':{k:len(v) for k,v in data.items()},'character_split':{k:cfg[k] for k in ('train_characters','val_characters','test_characters')}}
    torch.save(payload,cache)
    report={'cache':str(cache),'records_discovered':len(records),'kept':payload['split_counts'],
            'skipped':skipped,'ignored':ignored,'settings':settings}
    (cache_dir/f'qc_{key}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload
