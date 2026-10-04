# SurvTabDL

将表格深度学习模型适配到右删失生存分析，统一训练、风险评分、生存概率及模型保存接口。

支持 MLP、TabNet、NODE、FT-Transformer、SAINT、TabPFN，并提供 Cox 和 XGBoost 对照。

## 安装与示例

```bash
pip install -e ".[tabnet,xgboost]"
survtabdl demo --model MLP --epochs 30
```

示例自动生成模拟数据，划分训练集与测试集，训练模型，计算 C-index 并保存模型。

## 使用流程

1. 准备特征表 `X`、随访时间 `time` 和结局 `event`（事件为 1，删失为 0）。
2. 创建 `SurvivalEstimator`，指定模型与类别列。
3. 调用 `fit`，在训练集拟合预处理器、模型和 Breslow 基线。
4. 调用 `predict` 获取风险评分，或 `predict_event_probability` 获取指定时间的事件概率。
5. 用 `save` 和 `load` 保存、恢复完整模型。

```python
from survtabdl import SurvivalEstimator, make_survival_data
X, time, event = make_survival_data()
model = SurvivalEstimator("SAINT", categorical_features=["group"], epochs=30)
model.fit(X, time, event)
probability = model.predict_event_probability(X.iloc[:3], [5, 10])
model.save("model.joblib")
```

预测时间与随访时间的单位一致。`batch_size=None` 使用完整风险集，有限批次使用近似风险集。SAINT 预测采用固定训练参考集。FT-Transformer 保留论文中的列注意力版本；TabPFN 使用嵌入加生存预测头。

模型参数及完整接口见英文 README。
