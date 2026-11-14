"""List recordings discovered under the extracted IMU data directory."""
import argparse
from collections import Counter
from omc.data import discover_records

p=argparse.ArgumentParser(description='Scan an extracted IMU data directory.')
p.add_argument('--data_path',default='../excel',help='Folder containing subject directories, e.g. ../excel')
p.add_argument('--split_config',default='imu_split_config.json')
a=p.parse_args()
records,ignored,subjects=discover_records(a.data_path,a.split_config)
print('Subjects:',subjects)
print('Number of subjects:',len(subjects))
print('Discovered records:',len(records))
print('Split counts:',dict(Counter(x['split'] for x in records)))
print('Formats:',dict(Counter(x['format'] for x in records)))
print('Records per subject:')
for s in subjects:
    print(' ',s,dict(Counter(x['split'] for x in records if x['subject']==s)))
print('Ignored files:',ignored)
print('\nFirst records:')
for x in records[:20]: print(x['split'],x['subject'],x['character'],x['archive_path'])
