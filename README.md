# Cognitive-GPT-Toolkit

> *“让 AI 学会：自省、记忆、抽象、创造工具、并用这些工具反过来增强自身，实现智能的自举增益。”*


---

## 描述：


> “GPT自调度GPT、多代理协同；自主工具创建与持久化、动态记忆管理、Diff驱动增删改；AI办公自动化（排版、备忘录、PPT制作）、智能笔记整理与标注习惯；智能自举增益与O系列模型复现；个人知识分布建模与因材施教——旨在构建通向AGI的闭环引擎。”




**核心功能亮点**：

* **GPT 自主调动 GPT**：多代理协同，主→子→主循环，性能权重动态分配。
* **自创工具库**：detect & wrap 重复逻辑，`create_tool` 一键生成、注册、持久化。
* **动态记忆管理**：分层记忆（工作/情景/知识），实时注入，回流更新。
* **改良Diff 驱动代码演化**：使用首尾关键字定位内容结构化增删改补、自动回滚与测试驱动优化。
* **AI 办公自动化**：AI Office 套件（备忘录、排版、Diff 文档、PPT 生成）。
* **智能笔记整理**：自动抽取标题/术语/标注习惯，生成学习仪表盘与提问卡。
* **智能自举增益**：多模型（O 系列）协作实验，跨任务经验迁移。
* **个性化学习建模**：知识分布估计、提问预测、掌握度追踪，因材施教。

---

## 🧭 项目愿景

让 AI 既能“学会”你、也能“教会”你：

1. 了解每个用户的知识掌握分布与提问风格；
2. 主动提示可能不懂的概念；
3. 自动整理笔记、生成测验、智能排版办公文档；
4. 在重复劳动中抽象出可重用工具；
5. 用反思驱动自我进化，越用越聪明。

---

## 🧠 核心理念图（Meta Loop）

```
┌──────────────────────────────────────────┐
│              Task Loop (外环)            │
│  用户输入 → 推理 → 行动(call_tool等) → 结果 │
└───────────────┬──────────────────────────┘
                │
                ▼
       ┌──────────────────────┐
       │   <reflect> 元反思   │
       │ - 错误分析           │
       │ - 策略提炼           │
       │ - 经验固化(learnings)│
       └─────────┬────────────┘
                 │
                 ▼
       ┌──────────────────────┐
       │ <update_goal> 调整目标│
       │ - 重设OVERARCHING_GOAL│
       │ - 分解子任务          │
       └─────────┬────────────┘
                 │
                 ▼
       回流 → Memory & Tooling 强化下一轮任务
```

---

## 🧩 特性总览表

| 模块     | 动作/组件           | 解决的问题            | 当前实现                                | 路线图                |
| ------ | --------------- | ---------------- | ----------------------------------- | ------------------ |
| 元认知    | `<reflect>`     | 被动反馈不保留 → 主动学习经验 | 支持结构化写入 `shared_state['learnings']` | 分类经验、置信度、自动推荐复习    |
| 目标管理   | `<update_goal>` | 长期任务漂移           | 更新 `OVERARCHING_GOAL` & 子任务         | 与任务树 / DAG 集成      |
| 记忆     | 分层记忆            | 上下文爆炸、成本高        | 工作/情景/知识 三层抽象                       | 向量化检索、优先级衰减        |
| 工具检索   | `search_tools`  | 百工具噪声            | 向量相似度Top-K                          | 语义 + 功能标签 + 使用统计排序 |
| 自进化工具  | `create_tool`   | 重复逻辑             | 动态生成并注册工具                           | 沙箱执行、安全审计、单测自动生成   |
| 代码演化   | Diff Patch      | 历史大段拷贝浪费         | 结构化 diff 指令                         | AST 级增量变更 & 回滚     |
| 个体学习建模 | 知识分布图谱          | 因材施教             | 基于提问/答错记录估计掌握度                      | 贝叶斯知识追踪 / IRT 模型   |
| 学习辅助   | 问题预测卡片          | 阅读盲点             | 从文本预测高混淆区                           | 主动测验生成器            |
| 办公自动化  | AI Office 套件    | 排版痛苦、重复          | 备忘录 / 文档格式化 / PPT                   | 多模态模板、主题样式、协作接口    |

---

## 🏗️ 架构分层

```
app/
 ├─ coordinator/                     # 主控制循环, meta actions, memory mgmt
 │   └─ Coordinator_GPT_v8_Cognitive.py
 ├─ memory/
 │   ├─ working_memory.py
 │   ├─ episodic_memory.py
 │   └─ knowledge_memory.py
 ├─ tools/
 │   ├─ builtin/                     # 预置工具
 │   ├─ user_created/                # create_tool 动态生成
 │   └─ registry.json                # 工具元数据索引
 ├─ retrieval/
 │   └─ tool_vector_index.npy        # 工具向量嵌入 (模拟)
 ├─ office/
 │   ├─ memo.py
 │   ├─ formatter.py
 │   ├─ ppt_gen.py
 │   └─ note_summarizer.py
 ├─ learner/
 │   ├─ user_profile.py              # 知识掌握度估计
 │   ├─ question_forecast.py         # 预测用户会问啥
 │   └─ gap_detector.py              # 不掌握点定位
 ├─ diff/
 │   ├─ patch_engine.py
 │   └─ ast_edit.py (计划)
 └─ examples/
     ├─ demo_reflect.ipynb
     ├─ demo_tool_rag.ipynb
     └─ demo_office.ipynb
```

---

## 🔁 元认知循环： & \<update\_goal>

### 原理

通过显式动作触发“自省”，将经验固化为结构化数据，而非依赖上下文残留。

### 流程

1. 常规任务执行（call\_tool / execute\_code / respond）。
2. 触发条件：异常、循环、里程碑、用户请求“请总结刚才学到了什么”。
3. `<reflect>` 输出 JSON：

   ```json
   {
     "trigger": "error: API timeout",
     "observation": "重复调用同一工具无效",
     "insight": "应增加超时重试与备选数据源",
     "actionable": ["给ToolManager添加重试参数"],
     "tags": ["robustness", "api-latency"]
   }
   ```
4. 写入 `shared_state['learnings']`，供下轮提示注入。
5. 如需调整总体方向：触发 `<update_goal>`，重写或分解 `OVERARCHING_GOAL`。

### 理想目标效果

* 少走弯路：代理在多轮对话后变得“懂套路”。
* 经验迁移：不同任务中自动引用历史成功策略。
* 团队协作：多代理共享 learnings 池。

---

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

## 🔧 Diff 驱动代码改良 (增删改补)

### 痛点

LLM 生成整段文件 → 易超长、合并冲突、上下文丢失。

### 思路

* 用结构化 diff（行号 / AST patch）描述变更。
* 只提交增量修改 → 更易审阅、版本控制友好。
* 结合 `reflect` 记录失败补丁，自动回滚。

### 示例语法（人读友好伪格式）

```diff
@@ patch:patch_engine.py @@
- def apply_patch(text, patch):
+ def apply_patch(text, patch, validate=True):
+     """Apply patch with optional validation for context drift."""
      ...
```

未来扩展：

* AST 层语义补丁；
* 单测驱动验证；
* 多代理代码审查。

---

## 👤 个体知识分布建模 & 学习预测

目标：构建“因材施教” AI。

### 数据来源

* 用户问答记录（提问频率、错误更正）；
* 用户上传的笔记、标注习惯（高亮、TODO、? 号）；
* 测验任务的正确率；
* 学习路径完成度。

### 模型层级

| 层    | 描述                       | 技术思路                            |
| ---- | ------------------------ | ------------------------------- |
| 概念粒度 | 知识点 mastery score \[0-1] | 贝叶斯知识追踪 / IRT                   |
| 内容覆盖 | 文档 → 概念映射                | NLP 实体识别 + 概念图谱                 |
| 提问预测 | 阅读中可能问啥                  | 语言模型 + 混淆触发词 (e.g., “因此”, “注意”) |

### 用法

* 给定教材章节 → 列出你可能不懂的概念清单。
* 阅读时实时弹卡（flashcard style）。
* 期末复习：按掌握度排序推送练习题。

---

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

## 🗂️ AI 办公自动化套件（AI Office）

子模块目标：减少格式化、搬运、重复劳动时间，把人从“文件苦工”解放出来。

### 模块 & 理想效果

| 模块              | 输入                | 输出       | 说明              |
| --------------- | ----------------- | -------- | --------------- |
| Memo            | 临时想法、语音转录         | 结构化待办/提醒 | 支持优先级、日期识别      |
| Formatter       | 原始文档 (md/doc/txt) | 样式化文档    | 标题层级、目录、编号、引用格式 |
| PPT Generator   | Markdown/大纲/脑图    | pptx 幻灯片 | 支持主题、图表占位符、演讲备注 |
| Note Summarizer | 多份资料              | 去重整合摘要   | 合并冲突、引用溯源       |
| AutoDiff Writer | 文档旧版+新版片段         | 补丁报告     | 适合审稿/版本迭代       |

---

## 🔁 GPT 调度 GPT（智能自举增益）

通过 Coordinator 代理调度：

* 主代理负责任务规划；
* 子代理承担专职任务（检索、代码、排版、测验生成）；
* 结果回流主循环并进入 `reflect`；
* 子代理性能数据用于自动权重选择（哪个模型擅长哪个子任务）。

---

## 🧬 O 系列模型复现（研究向）

> 非官方，仅作研究：构建轻量化 open 模型 pipeline，模拟“大模型 + 小专家模型”协作；评估在记忆蒸馏、策略迁移中的表现。

研究方向：

* 知识蒸馏：从大模型采样“反思+行动”对话，微调小模型。
* 工具使用演示数据自动生成。
* 学习曲线对比（有/无元认知）。

---

## 🧪 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourname/gpt-autoboot-agent.git
cd gpt-autoboot-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 环境变量

```
export OPENAI_API_KEY=...    # 或兼容其他模型API
```

### 4. 运行基础 Demo（元认知循环）

```bash
python examples/demo_reflect.py --model gpt-4o-mini
```

### 5. 运行工具检索 Demo

```bash
python examples/demo_tool_rag.py
```

### 6. 办公自动化示例：Markdown → PPT

```bash
python examples/demo_office.py --input lecture.md --out lecture.pptx
```

---

## 🧪 Python 内嵌示例片段：调用 Coordinator

```python
from coordinator.Coordinator_GPT_v8_Cognitive import CognitiveCoordinator
from memory import WorkingMemory, EpisodicMemory, KnowledgeMemory
from tools import ToolManager

coordinator = CognitiveCoordinator(
    tool_manager=ToolManager.from_registry("app/tools/registry.json"),
    working_memory=WorkingMemory(max_turns=8),
    episodic_memory=EpisodicMemory(store_path="app/memory/episodic.db"),
    knowledge_memory=KnowledgeMemory(store_path="app/memory/knowledge.db")
)

result = coordinator.run(user_query="帮我把这篇学习笔记转成ppt并标出我不熟悉的概念")
print(result)
```

---

## 📈 Roadmap

*

---

## 🤝 如何贡献

欢迎以下类型的贡献：

* 新工具：提交 Python 函数 + docstring；
* 学习建模插件：支持不同学科（编程、数学、语言学习）；
* 记忆后端：SQLite / Chroma / Weaviate / 自定义向量库；
* UI 仪表盘；
* Demo 数据集（匿名化）。

### 提交流程

1. Fork & branch。
2. 新增/修改工具 → `tools/user_created/your_tool.py`。
3. 运行 `scripts/update_registry.py` 生成索引。
4. 提交 PR，并在描述中说明：功能、参数、测试、使用示例。

---

## 🪪 License

本项目以 **MIT License** 开源（除非子目录另有说明）。

---

## 🌐 多语言支持

* README 默认中英混排；
* `docs/` 目录中可分 `zh/` 与 `en/` 版本；
* 模型提示语可配置语言偏好。

---

## 🙏 致谢

灵感来源于：

* 人类元认知学习理论；
* 检索增强生成 (RAG) 框架；
* 程序化工具链自动化工作流；
* 社区多代理实验生态。

欢迎 star、issue、PR，一起把“会学、会反思、会创造工具”的 GPT 智能体做起来！🚀

---

# English Overview (Short)

**GPT AutoBoot Agent** is a self-bootstrapping, meta-cognitive agent framework for LLMs. It learns from feedback, compresses long histories into layered memory, retrieves the right tool out of hundreds, and can *create new tools* when it detects repeated code patterns. It models user knowledge mastery, predicts likely confusion points in documents, and ships with an AI Office toolkit (memoing, formatting, slide generation, diff-based doc evolution). Experimental modules explore open, lightweight “O-style” multi-model orchestration and knowledge distillation.

> See Chinese sections above for full details; English docs coming soon.

---

\*\*下一步：\*\*告诉我你想采用哪个仓库名，我可以帮你生成 `pyproject.toml`、`requirements.txt`、基础目录结构脚本，或继续扩写 docs！
