# OMC IMU：文件夹自动扫描版（不依赖CSV或ZIP）

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
│   ├── B/
│   │   └── ...
│   └── ...
└── OMC_IMU_Research/              # GitHub代码仓库
    ├── train.py
    ├── test.py
    ├── scan_data.py
    ├── imu_split_config.json
    └── omc/
```

运行命令中的 `--data_path ../excel` 指向包含受试者文件夹（A、B、C……）的 `excel` 目录本身。

## 2. 文件命名规则

支持：

- `一.xlsx`：字符「一」的原始记录。
- `一_trial001.xlsx`：字符「一」的第1次重复采集。
- `一_02.xlsx`、`一-2.xlsx`、`一（2）.xlsx`：也会识别为字符「一」的重复采集。
- 支持 `.xlsx` 和 `.xls`。

每次独立采集必须使用唯一文件名，不能覆盖已有trial。所有数据必须放在对应受试者的一级子目录下，例如 `excel/A/一_trial001.xlsx`。根目录的汇总表或说明文件会被忽略。

数据格式约定：前三轴为已去除重力分量的线性加速度，后三轴为三轴陀螺仪角速度。程序不会再次去除重力。

## 3. 固定字符划分

`imu_split_config.json` 保存字符级划分：35个训练字符、5个验证字符、10个测试字符。程序按字符自动分配，某个字符的全部用户、全部trial都进入同一个split。

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

质量报告保存在 `cache/qc_*.json`，最佳模型保存在 `runs/omc_6axis/best.pt`。

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
