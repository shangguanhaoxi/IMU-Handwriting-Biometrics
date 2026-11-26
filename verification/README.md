# 未见用户跨字符 IMU 身份验证：首次试验

## 目标和与旧实验的关系

本目录实现手机主人注册后、冻结编码器的 open-set **verification**，而不是对9名用户的闭集9分类。新用户从未参与模型训练，只在本地提供若干注册样本；他人不能通过与主人模板的比较。

当前实验只是基线，不声称具备产品级防攻击或安全性。**不要用旧的 runs/omc_9subject_vat/best.pt**：它在9人数据上全部训练过，不满足训练用户与测试用户分离的要求。

## 固定受试者协议（9名）

见 protocol.json：

| 阶段 | 受试者 | 用途 |
|---|---|---|
| Train | A, F, lrx, sg, yl | 训练OMC编码器；仅使用原先train的35字、175条样本 |
| Dev | ypb, zjl | 选择模型checkpoint、确定接受阈值；不用于梯度训练和归一化统计 |
| Test | zjq, zpc | **最终未知用户**；先注册，再分别模拟主人与冒充者 |

上述三组无受试者重叠。旧版 imu_split_config.json 中固定的字符级划分仍沿用：

- 模型训练：35个训练字符（训练用户175条）。
- 每个Dev/Test主人注册：5个不同字符「一、人、入、口、山」。
- 主人及其他未知用户的查询：剩余45个不同字符（包含30个训练字符、5个原验证字符、10个原测试字符）。
- 另给出仅原测试10字符的独立统计（与训练模型使用的字符不重叠）。
- 冒充者与主人均从同一Dev或同一Test小组抽取，**从不使用训练身份作为最终冒充者**。
- 冻结编码器后，测试用户只使用5个注册样本做模板均值，不进行反向传播。

当前Dev组2人、Test组2人，因此每一组有：2名主人 × 45条真实查询 = 90个genuine trials，2名主人 × 另一名用户45条 = 90个impostor trials；原测试10字符子集则各有20个genuine和20个impostor trials。所有试次不是相互独立的统计观察，**独立用户只有2名**。

## 模型与决策

- 从头初始化 OMC，继续使用 EMF、ArcFace、VAT和正交约束（默认开启）。
- ArcFace只学习5名训练用户；分类头不参与新用户解锁。
- 输入预处理仍采用根目录的 imu_gravity_config.json，训练统计只由5名训练用户的35字符样本拟合。
- 128维模型嵌入归一化；每个手机主人的5个注册向量求均值后再次L2归一化，形成身份模板。
- 每条查询向量与目标主人模板计算余弦相似度。
- **仅在Dev用户试次上**，根据目标经验 FAR=5% 选择固定阈值；Test完全沿用该阈值，不重新标定。
- 每Epoch用Dev ROC-AUC选择checkpoint（同分时用近似EER作辅助），训练结束后才评价Test。

5%的Dev FAR只是为了在小规模样本上观察方法行为，不是满足手机解锁安全要求的证据。

## AutoDL运行

从仓库根目录：

~~~bash
cd /root/autodl-tmp/OMC_IMU_Research
git pull --ff-only origin main

python -m unittest discover -s tests -v

python -m verification.run \
    --data_path ../excel \
    --split_config imu_split_config.json \
    --gravity_config imu_gravity_config.json \
    --protocol verification/protocol.json \
    --output_dir runs/verification_5train_2dev_2test \
    --epochs 100 \
    --patience 20 \
    --batch_size 16 \
    --seed 42
~~~

无需使用之前闭集9分类的训练模型。请不要为了取得更好测试结果而反复调整受试者分组、注册汉字或阈值；这些都应在测试前固定。

## 输出文件

输出均在 runs/verification_5train_2dev_2test：

| 文件 | 内容 |
|---|---|
| best.pt | 在Dev上选出的冻结身份编码器权重、训练统计、Dev阈值、协议 |
| metrics.json | Dev选择指标、Test全部不同字符查询结果、Test未见过的10字符子集结果、逐手机主人FAR/FRR |
| test_trials.csv | 每一条owner-query相似度、正确标签、接受与否 |
| train_history.csv | 每轮训练损失、Dev ROC-AUC、EER及错误率 |
| run_config.json | 本次命令参数 |
| skipped_records.json | 解析失败与被忽略文件清单 |

主要验证指标：FAR（非主人错误通过）、FRR（主人错误被拒）、TAR=1-FRR、ROC-AUC、近似EER。**真正用于解锁决策的是Dev确定的固定阈值对应的FAR/FRR**；Test EER是事后诊断，不参与阈值选择。

## 必须在论文中声明的限制

1. 只有9位真实用户；当前Train仅5人，最终Test仅2人，这**不足以证明对一般未知用户的安全泛化**。
2. 多个字符、多个owner-query配对共享用户和同一条原始记录，不能把180个配对当成180个独立人的安全测试。
3. 设备、采集格式、重力定义和采集会话差异仍可能是身份捷径；当前预处理不等于精确去重力。
4. 通过相似度区分身份并不等于具备防重放、模仿或对抗攻击能力。
5. 未来应扩充真实受试者，补做交叉受试者协议、多个注册数量K以及严格FAR条件的置信度评估。
6. 按用户分离与按字符分离是两条独立轴；同时报告跨字符、未见字符的结果是必要的。
