# OMC IMU：9名真实受试者的文件夹识别实验

> **重要数据订正（2026-11-22）**：B、C、D、E与A是同一自然人此前误分出的重复身份。已从数据中删除B/C/D/E，仅保留A。现在只允许9名受试者：A、F、lrx、sg、yl、ypb、zjl、zjq、zpc。旧13类身份识别实验及其A～E混淆分析**不再构成有效的不同人身份识别证据**。必须重新训练9类模型，不可复用旧checkpoint。

## 1. 数据输入方式

本项目**只从已解压的数据文件夹读取数据**，不再支持 ZIP 输入，也不需要维护 `imu_split_manifest.csv`。

AutoDL目录约定：

```text
/root/autodl-tmp/
├── excel/                         # 原始IMU数据（已解压）
│   ├── A/
│   │   ├── 一.xlsx
│   │   ├── 一_trial001.xlsx
│   │   ├── 一_trial002.xlsx
│   │   └── 十.xlsx
│   ├── F/
│   │   └── ...
│   ├── lrx/ ...
│   └── zpc/ ...
└── OMC_IMU_Research/              # GitHub代码仓库
    ├── train.py
    ├── test.py
    ├── scan_data.py
    ├── imu_split_config.json
    └── omc/
```

运行命令中的 `--data_path ../excel` 指向包含保留受试者文件夹（A、F、lrx……）的 `excel` 目录本身。

## 2. 文件命名规则

支持：

- `一.xlsx`：字符「一」的原始记录。
- `一_trial001.xlsx`：字符「一」的第1次重复采集。
- `一_02.xlsx`、`一-2.xlsx`、`一（2）.xlsx`：也会识别为字符「一」的重复采集。
- 支持 `.xlsx` 和 `.xls`。

每次独立采集必须使用唯一文件名，不能覆盖已有trial。所有数据必须放在对应受试者的一级子目录下，例如 `excel/A/一_trial001.xlsx`。根目录的汇总表或说明文件会被忽略。

数据格式约定：原始前三轴加速度类型按 imu_gravity_config.json 区分；A暂定为已去重力线性加速度，其余八人暂定为含重力原始加速度，程序依配置执行近似补偿。后三轴为陀螺仪角速度。采集字段和单位仍需核实。

## 3. 固定字符划分

`imu_split_config.json` 保存字符级划分：35个训练字符、5个验证字符、10个测试字符。当前9人，预期 train/val/test 各为315/45/90，共450条；实际以AutoDL重新扫描为准。该配置还含 active_subjects 和 retired_duplicate_subjects，若B/C/D/E仍在文件夹中，扫描会报错而不会静默计为身份。程序按字符自动分配，某个字符的全部用户、全部trial都进入同一个split。

新增已有字符的重复采集文件时，不需要修改CSV或JSON。例如「一」属于train，则所有用户的 `一.xlsx`、`一_trial001.xlsx` 都属于train；「十」属于test，则其所有trial都属于test。

新增全新字符时，必须先决定其属于train、val还是test，并更新JSON；不能自动猜测实验划分。

当前协议用于已知用户的跨字符身份识别。新增受试者后，模型身份类别数发生变化，需要重新训练。测试旧checkpoint时，如果受试者集合与训练时不同，程序会报错，避免错误评估。

## 4. 安装依赖

当前AutoDL环境使用 PyTorch 2.5.1+cu124。必要依赖：

```bash
pip install numpy pandas scipy scikit-learn openpyxl xlrd==2.0.2
```

## 5. 扫描数据

```bash
cd /root/autodl-tmp/OMC_IMU_Research
python scan_data.py \
  --data_path ../excel \
  --split_config imu_split_config.json
```

扫描程序会打印用户数量、发现的记录数量、train/val/test数量、文件格式以及被忽略文件。

## 6. OMC六轴训练

```bash
python train.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --input_mode accgyro6 \
  --output_dir runs/omc_6axis \
  --epochs 100 \
  --batch_size 16 \
  --seq_len 256 \
  --sample_rate 100 \
  --seed 42
```

质量报告保存在 `cache/qc_*.json`，最佳模型保存在 `runs/omc_9subject_vat/best.pt`。

## 7. 缓存与新增样本

缓存指纹包含每个数据文件的相对路径、文件大小、CRC32、JSON划分和预处理参数。新增、替换或删除文件后，扫描结果和缓存指纹会变化，程序会生成新的特征缓存。通常不需要手动清理缓存。

需要强制重建缓存时：

```bash
python train.py --data_path ../excel --rebuild_cache
```

新增训练trial后需要重新训练模型，旧checkpoint不会自动学习新数据。

## 8. 独立测试

```bash
python test.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --checkpoint runs/omc_6axis/best.pt \
  --split test
```

## 9. 实验约束

- split由字符决定，不由文件或trial随机决定，避免同一字符跨训练/测试泄漏。
- 不要将同一个trial切窗后再把窗口分配到不同split。
- 同一个用户-字符可以有多个trial；所有trial继承该字符的固定split。
- 原始数据只保存在AutoDL，不上传GitHub。
- B/C/D/E并非真实独立身份，已经从当前数据集排除。9类实验必须从头训练，旧的13分类报告不能再被用于讨论不同人的身份混淆。

## 10. 受试者运动差异与跨字符稳定性分析

新增 analyze_subject_variation.py，用于统计六轴原始运动特征、EMF特征、不同受试者的运动分布，以及同一受试者跨字符的特征相似性。可选传入 test.py 生成的 predictions.csv，进一步导出逐受试者指标和错误样本明细。

~~~bash
python analyze_subject_variation.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --output_dir analysis/subject_diagnostics \
  --predictions runs/omc_6axis_vat/evaluation_test/predictions.csv
~~~

详细指标定义、输出文件说明和解释注意事项见 ANALYSIS_GUIDE.md。该脚本只做分析，不训练模型，也不会修改训练缓存或checkpoint。

## 11. 按受试者统一加速度重力处理（暂定采集类型）

根目录新增 imu_gravity_config.json，记录暂定采集方式：

- A：linear_acceleration_subjects，暂认为已去除重力，六轴信号保持原样；B/C/D/E已退役，不再参与身份建模。
- F、lrx、sg、yl、ypb、zjl、zjq、zpc：raw_acceleration_subjects，认为含重力，先估计并移除三轴低频分量。

默认在 raw_acceleration_subjects 上使用二阶零相位 Butterworth 低通（截止频率 0.5 Hz）估计重力，再从前三轴加速度中扣除。该算法属于**估计性重力补偿**，不是利用姿态解算得到的真实重力；低频书写运动也可能被削弱，尤其是时长较短的书写样本。配置判断尚待核实采集API和字段定义。不要擅自将去重力和含重力数据直接合并作为物理一致的线性加速度。

修正后的预处理会同时用于训练、验证、测试、VAT速度标签构造和受试者分析。配置文件内容的 SHA256 会进入缓存指纹和新checkpoint；旧checkpoint无法自动适配新特征，必须重新训练。

建议新训练输出目录，不覆盖旧基线：

~~~bash
python train.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --gravity_config imu_gravity_config.json \
  --input_mode accgyro6 \
  --output_dir runs/omc_9subject_vat \
  --epochs 100 \
  --patience 20 \
  --batch_size 16 \
  --seq_len 256 \
  --sample_rate 100 \
  --seed 42
~~~

测试新checkpoint：

~~~bash
python test.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --gravity_config imu_gravity_config.json \
  --checkpoint runs/omc_6axis_gravity_vat/best.pt \
  --split test
~~~

分析数据时：

~~~bash
python analyze_subject_variation.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --gravity_config imu_gravity_config.json \
  --output_dir analysis/subject_diagnostics_gravity \
  --predictions runs/omc_6axis_gravity_vat/evaluation_test/predictions.csv
~~~

分析 CSV 中 raw_acc_mean_vector_norm 保留原始加速度均值向量模长，corrected_acc_mean_vector_norm 为补偿后对应值。二者只用于诊断，不应被直接解释为采集类型的严格物理证明。

运行测试：

~~~bash
python -m unittest discover -s tests -v
~~~
