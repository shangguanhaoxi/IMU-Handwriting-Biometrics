import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omc.data import FeatureDataset,prepare_data
from omc.engine import get_device,inference,metrics,write_report
from omc.model import OMC

def main():
 p=argparse.ArgumentParser(description='Evaluate OMC checkpoint; auto-discovers records inside ZIP, no CSV manifest.')
 p.add_argument('--checkpoint',required=True); p.add_argument('--archive',default='../excel.zip'); p.add_argument('--split_config',default='imu_split_config.json')
 p.add_argument('--cache_dir',default='cache'); p.add_argument('--output_dir',default=''); p.add_argument('--split',choices=['train','val','test'],default='test')
 p.add_argument('--exclude_subject',action='append',default=[]); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--device',default='auto'); a=p.parse_args(); device=get_device(a.device)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); ds_cfg=ck['data_settings']
 ds=prepare_data(str(Path(a.archive).resolve()),str(Path(a.split_config).resolve()),a.cache_dir,input_mode=ds_cfg['input_mode'],sample_rate=ds_cfg['sample_rate'],seq_len=ds_cfg['seq_len'],min_length=ds_cfg['min_length'],velocity_detrend=ds_cfg['velocity_detrend'],exclude_subjects=a.exclude_subject)
 if ck.get('character_split') != ds['character_split']: raise ValueError('Character split differs from checkpoint. Use the same split config as training.')
 if ck['subjects']!=ds['subjects']: raise ValueError(f"Subject list differs from checkpoint. checkpoint={ck['subjects']} current={ds['subjects']}. Retrain after adding/removing identities.")
 q1,q2=ck['velocity_thresholds']
 for items in ds['data'].values():
  for x in items: x['velocity']=int(x['speed']>=q1)+int(x['speed']>=q2)
 loader=DataLoader(FeatureDataset(ds['data'][a.split],ck['stats']),batch_size=a.batch_size,shuffle=False)
 model=OMC(**ck['architecture']).to(device); model.arcface.margin=ck['arc_margin']; model.arcface.scale=ck['arc_scale']; model.load_state_dict(ck['model_state'])
 y,pr,rows=inference(model,loader,device); met,cm=metrics(y,pr,len(ck['subjects']))
 met.update({'split':a.split,'checkpoint':str(Path(a.checkpoint).resolve()),'epoch':ck['epoch'],'subjects':ck['subjects'],'records_discovered':ds['records_discovered'],'split_counts':ds['split_counts']})
 out=Path(a.output_dir) if a.output_dir else Path(a.checkpoint).parent/f'evaluation_{a.split}'; write_report(out,met,cm,rows,ck['subjects'])
 print(json.dumps(met,ensure_ascii=False,indent=2)); print('Report:',out.resolve())
if __name__=='__main__': main()
