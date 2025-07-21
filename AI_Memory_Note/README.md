# 此文档用于说明AI备忘录、记忆动态管理的构建

![ai_NOTE](https://github.com/kay-cottage/Cognitive-GPT-Toolkit/blob/main/img/932f061ceeb11a3c4e6d4dbd7e3cfce.png)

![ai_NOTE1](https://github.com/kay-cottage/Cognitive-GPT-Toolkit/blob/main/img/d27c1ce9a17a942c929942f53885390.png)

![ai_NOTE](https://github.com/kay-cottage/Cognitive-GPT-Toolkit/blob/main/img/98222bdcb95a0a925f7fa66186366fd.png)

## 🧳 分层记忆系统

| 层级   | 内容         | 保留长度 | 存储形式          | 注入频率   | 用途    |
| ---- | ---------- | ---- | ------------- | ------ | ----- |
| 工作记忆 | 最近 N 轮完整对话 | 短    | 原始文本          | 每轮必注入  | 当前语境  |
| 情景记忆 | 过往步骤摘要     | 中    | 结构化 event 列表  | 需时检索   | 回溯、调试 |
| 知识记忆 | 精炼经验 & 变量  | 长    | key-value/图结构 | 每轮精简注入 | 跨任务学习 |

自动裁剪：超长历史 → 摘要器 → 情景事件；关键洞察 → 知识记忆。

未来扩展：

* 记忆评分（新鲜度、成功率、覆盖度）。
* 置信度驱动的记忆回写（低置信度不写）。

---

## 🔍 海量工具精准召回：`search_tools`

### 问题

工具越多，提示词越乱；AI 容易“忘记”合适工具。

### 原理

1. 预处理：将 *工具名 + docstring + 参数签名* 编码成向量。
2. AI 用自然语言说明意图（例如：“我要把 markdown 转 ppt”）。
3. 向量相似度检索 → Top-K 工具元数据注入下一轮提示。
4. AI 再从 K 个候选中选择 `call_tool`。

### 例子

```
<search_tools>
Convert structured markdown lecture notes into pptx slides with simple theme.
</search_tools>
```

系统返回候选：`md_to_ppt`, `slide_templater`, `deck_outline_helper`。

### 理想效果

* Token 成本显著下降。
* 工具使用命中率提升。
* 支持插件生态：开发者新增工具后自动进入检索库。

---

## 🧪 自进化工具库：`create_tool`

### 动机

当代理多次重复相似 Python 片段时，应抽象成可复用工具，从而：

* 降低重复错误；
* 缩短提示长度；
* 让经验“固化为能力”。


## 📝 AI 自主笔记整理 & 标注习惯学习

### 功能

* 自动抽取标题、术语、定义、公式。
* 根据你常用标记（如TODO, 重点⭐, 疑问？）生成结构化索引。
* 跨文件合并：同一概念多来源对齐。
* 生成“我已懂 / 半懂 / 未懂”仪表盘。

### 流程示意

```
原始笔记.md → parse_annotations() → concept_map → knowledge_score_update() → review_cards.json → 学习仪表盘
```

---


### 使用格式（示例）

```
<create_tool>
name: summarize_errors
code:
"""
def summarize_errors(logs: list) -> dict:
    """总结错误类型及频率。
    参数:
        logs (list[str]): 日志行。
    返回:
        dict: {错误类型: 次数}
    """
    from collections import Counter
    kinds = []
    for line in logs:
        if 'ERROR' in line:
            kinds.append(line.split('ERROR')[-1].strip())
    return dict(Counter(kinds))
"""
</create_tool>
```

### 生命周期

1. 语法/安全检查。
2. 动态 import & 注册。
3. 向量化索引更新（供 `search_tools` 检索）。
4. 持久化至 `tools/user_created/`。
5. 后续任务可直接使用。

### 理想目标效果

* 工具生态自我扩展；
* 用户使用越多，工具库越适配其工作流；
* 形成“个人化 AI 办公开发环境”。

---

