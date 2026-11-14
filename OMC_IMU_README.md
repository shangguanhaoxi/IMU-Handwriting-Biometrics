# OMC IMU：自动扫描数据版（支持解压目录和ZIP，无需维护CSV清单）

## 1. 核心变化

旧版训练代码需要手工维护 `imu_split_manifest.csv`。本版本不再读取该文件，既可以直接扫描已解压的数据目录（推荐），也可以扫描 `excel.zip`，并根据 `imu_split_config.json` 的字符列表自动分配 train/val/test。

**新增重复采集样本时，只需把新文件放进对应受试者目录，不需要修改CSV，也不需要修改当前固定的JSON划分。**

## 2. 数据目录组织规则

推荐直接使用已解压目录 `/root/autodl-tmp/excel`，其内部结构应为：

```text
/root/autodl-tmp/excel/
├── A/
│   ├── 一.xlsx
│   ├── 一_trial001.xlsx
│   └── 十.xlsx
├── B/
│   └── ...
└── 新表.xlsx                 # 根目录元数据文件会被忽略
```

目录模式下，`--data_path ../excel` 指向 `excel` 本身，而不是它的上一级目录。也兼容 ZIP 模式，ZIP 内部仍要求 `excel/<subject>/<character>.xlsx`。

## 2. ZIP 内部组织规则

推荐：

```text
excel.zip
└── excel/
    ├── A/
    │   ├── 一.xlsx
    │   ├── 一_trial001.xlsx
    │   ├── 一_trial002.xlsx
    │   ├── 十.xlsx
    │   └── 十_trial001.xlsx
    ├── B/
    │   ├── 一.xlsx
    │   └── 一_trial001.xlsx
    └── ...
```

识别规则：

- `excel/<受试者>/<字符>.xlsx`：原有文件名，自动识别。
- `excel/<受试者>/<字符>_trial001.xlsx`：推荐的重复采集命名。
- 也支持 `一-02.xlsx`、`一_02.xlsx`、`一（2）.xlsx` 等字符名后有分隔符/数字的文件。
- `.xlsx` 前3列为去重力线性加速度，后3列为角速度；不会再次去重力。
- `.xls` 旧格式由 xlrd 读取，按加速度/陀螺仪工作表和时间戳对齐。
- `excel/新表.xlsx` 这种根目录索引文件会自动忽略。

请勿把两次独立采集覆盖为同一个文件名。每个trial使用唯一文件名。若采集软件导出的文件名不是上述形式，先按该规则重命名即可。

## 3. 固定划分配置

`imu_split_config.json` 保存字符级划分：35个训练字符、5个验证字符、10个测试字符。程序按字符分配，某个字符的全部用户、全部trial都会落入同一个split。

这份JSON可以保持不变。新增某个已知字符的trial会自动继承其split。例如「一」在train中，那么 `一.xlsx`、`一_trial001.xlsx`、`一_trial002.xlsx` 对所有受试者都属于train；「十」在test中，其所有trial都属于test。

当前协议用于已知用户的跨字符识别。新增受试者时，必须为其提供训练字符记录，模型类别数会变化，需要重新训练；测试旧checkpoint时，如果受试者集合发生变化，程序会报错，避免错误评估。

## 4. 路径约定

建议：

```text
/root/autodl-tmp/
├── excel/                    # 已解压的原始数据，推荐使用
├── excel.zip                 # 可选备份
└── OMC_IMU_Research/
    ├── train.py
    ├── test.py
    ├── imu_split_config.json
    └── omc/
```

## 5. 安装依赖

确保AutoDL环境已安装 PyTorch 2.5.1+cu124。补充依赖：

```bash
pip install numpy pandas scipy scikit-learn openpyxl xlrd==2.0.2
```

## 6. 训练

```bash
cd /root/autodl-tmp/OMC_IMU_Research
python train.py \
  --archive ../excel.zip \
  --split_config imu_split_config.json \
  --input_mode accgyro6 \
  --output_dir runs/omc_6axis \
  --epochs 100 --batch_size 16 --seq_len 256 --sample_rate 100
```

程序启动时会打印自动发现的记录数、各split保留数量、跳过记录数和用户列表。质量报告保存在 `cache/qc_*.json`。

## 7. 新增样本后的缓存

本版本缓存指纹包含数据文件的相对路径、大小、CRC、JSON划分和预处理参数。目录模式和ZIP模式均支持自动检测新增或替换的样本。新增/替换文件后，指纹会变化，自动生成新的特征缓存。一般不需要手工删除缓存。

如果需要强制重建：

```bash
python train.py --archive ../excel.zip --rebuild_cache
```

注意：新增数据后应重新训练。若只执行 `test.py`，它会重新扫描数据并评估当前checkpoint；这适合评估同一身份集合的新测试trial，但不代表旧模型已经学习了新增训练trial。

## 8. 独立测试

```bash
python test.py \
  --archive ../excel.zip \
  --split_config imu_split_config.json \
  --checkpoint runs/omc_6axis/best.pt \
  --split test
```

## 9. 关键实验约束

- split由字符决定，不由文件或trial随机决定；这是防止同字符跨训练/测试泄漏的关键。
- 不要把一个trial切成窗口后再把窗口分配到不同split。
- 每个用户-字符可有多个trial；这些trial会一起继承字符split。
- 每次加入数据后，建议保存新版本的 `excel.zip`，以便追溯实验数据版本。
- 如果新增字符，必须先明确该字符应属于train/val/test中的哪一组，再将字符加入对应JSON列表；本次“无需维护CSV”的改动不意味着可以在不定义评估协议的情况下自动分配新字符。
