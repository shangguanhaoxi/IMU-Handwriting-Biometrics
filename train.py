import argparse,json,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from omc.data import FeatureDataset,prepare_data
from omc.engine import fix_seed,get_device,train_epoch,inference,metrics,write_report
from omc.model import OMC,BoundedUncertaintyWeighting

def main():
 p=argparse.ArgumentParser(description='Train OMC from an extracted IMU data directory; no CSV manifest.')
 p.add_argument('--data_path',default='../excel',help='Extracted IMU data directory, e.g. ../excel'); p.add_argument('--split_config',default='imu_split_config.json')
 p.add_argument('--gravity_config',default='imu_gravity_config.json'); p.add_argument('--cache_dir',default='cache'); p.add_argument('--output_dir',default='runs/omc_6axis')
 p.add_argument('--input_mode',choices=['acc3','accgyro6'],default='accgyro6'); p.add_argument('--sample_rate',type=float,default=100.)
 p.add_argument('--seq_len',type=int,default=256); p.add_argument('--min_length',type=int,default=16); p.add_argument('--velocity_detrend',action='store_true')
 p.add_argument('--rebuild_cache',action='store_true'); p.add_argument('--exclude_subject',action='append',default=[]); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=20)
 p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight_decay',type=float,default=1e-4); p.add_argument('--dim',type=int,default=64); p.add_argument('--embed_dim',type=int,default=128)
 p.add_argument('--depth',type=int,default=6); p.add_argument('--heads',type=int,default=4); p.add_argument('--ff_mult',type=int,default=2); p.add_argument('--conv_kernel',type=int,default=15); p.add_argument('--dropout',type=float,default=.1)
 p.add_argument('--ort_lambda',type=float,default=.1); p.add_argument('--arc_scale',type=float,default=30.); p.add_argument('--arc_margin',type=float,default=.35)
 p.add_argument('--no_emf',action='store_true'); p.add_argument('--no_vat',action='store_true'); p.add_argument('--no_arcface',action='store_true'); p.add_argument('--jitter_std',type=float,default=0.)
 p.add_argument('--seed',type=int,default=42); p.add_argument('--num_workers',type=int,default=0); p.add_argument('--device',default='auto')
 a=p.parse_args(); fix_seed(a.seed); device=get_device(a.device)
 if a.rebuild_cache: shutil.rmtree(a.cache_dir,ignore_errors=True)
 ds=prepare_data(str(Path(a.data_path).resolve()),str(Path(a.split_config).resolve()),a.cache_dir,input_mode=a.input_mode,
   sample_rate=a.sample_rate,seq_len=a.seq_len,min_length=a.min_length,velocity_detrend=a.velocity_detrend,exclude_subjects=a.exclude_subject,rebuild_cache=a.rebuild_cache,gravity_config=a.gravity_config)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 tr=FeatureDataset(ds['data']['train'],ds['stats'],True,a.jitter_std); va=FeatureDataset(ds['data']['val'],ds['stats'])
 tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=device.type=='cuda')
 vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
 first=ds['data']['train'][0]
 arch={'seq_dim':first['seq'].shape[-1],'global_dim':len(first['global']),'num_classes':len(ds['subjects']),'dim':a.dim,'depth':a.depth,'heads':a.heads,'embed_dim':a.embed_dim,'ff_mult':a.ff_mult,'conv_kernel':a.conv_kernel,'dropout':a.dropout,'use_emf':not a.no_emf,'use_vat':not a.no_vat,'input_mode':a.input_mode}
 model=OMC(**arch).to(device); model.arcface.scale=a.arc_scale; model.arcface.margin=a.arc_margin
 awl=BoundedUncertaintyWeighting().to(device); opt=torch.optim.AdamW(list(model.parameters())+list(awl.parameters()),lr=a.lr,weight_decay=a.weight_decay)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(a.epochs,1),eta_min=a.lr*.05)
 (out/'run_config.json').write_text(json.dumps(vars(a),ensure_ascii=False,indent=2),encoding='utf8')
 print(f'device={device}; discovered={ds["records_discovered"]}; kept={ds["split_counts"]}; skipped={len(ds["skipped"])}; subjects={ds["subjects"]}')
 hist=[]; best=-np.inf; stale=0; best_epoch=0
 for epoch in range(1,a.epochs+1):
  losses=train_epoch(model,awl,tl,opt,device,a.ort_lambda,not a.no_arcface)
  y,pr,rows=inference(model,vl,device); met,cm=metrics(y,pr,len(ds['subjects'])); sched.step()
  rec={'epoch':epoch,'lr':opt.param_groups[0]['lr'],'val_accuracy':met['accuracy'],'val_macro_f1':met['macro_f1'],'val_kappa':met['cohen_kappa']}
  rec.update({'loss_'+k:v for k,v in losses.items()}); hist.append(rec)
  print(f"epoch={epoch:03d} loss={losses['total']:.4f} val_acc={met['accuracy']:.4f} val_f1={met['macro_f1']:.4f}",flush=True)
  if met['macro_f1']>best+1e-6:
   best=met['macro_f1']; stale=0; best_epoch=epoch
   ck={'model_state':model.state_dict(),'awl_state':awl.state_dict(),'architecture':arch,'stats':ds['stats'],'subjects':ds['subjects'],
       'velocity_thresholds':ds['velocity_thresholds'],'gravity_config':ds['gravity_config'],'character_split':ds['character_split'],'data_settings':{'input_mode':a.input_mode,'sample_rate':a.sample_rate,'seq_len':a.seq_len,'min_length':a.min_length,'velocity_detrend':a.velocity_detrend,'gravity_config_digest':ds['gravity_config_digest']},
       'arc_margin':a.arc_margin,'arc_scale':a.arc_scale,'epoch':epoch,'best_val_macro_f1':best,
       'loss_settings':{'ort_lambda':a.ort_lambda,'use_arcface':not a.no_arcface}}
   torch.save(ck,out/'best.pt'); write_report(out/'validation_best',met,cm,rows,ds['subjects'])
  else: stale+=1
  if stale>=a.patience: print(f'Early stopping; best epoch={best_epoch}'); break
 pd.DataFrame(hist).to_csv(out/'train_history.csv',index=False)
 (out/'summary.json').write_text(json.dumps({'best_epoch':best_epoch,'best_val_macro_f1':best,'records_discovered':ds['records_discovered'],'split_counts':ds['split_counts'],'subjects':ds['subjects'],'skipped':ds['skipped']},ensure_ascii=False,indent=2),encoding='utf8')
 print('Saved:',out/'best.pt')
if __name__=='__main__': main()
