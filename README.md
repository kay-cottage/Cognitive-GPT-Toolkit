

# Cognitive-GPT-Toolkit

> “让 AI 学会：自省、记忆、抽象、创造工具，并用这些工具反过来增强自身，实现智能的自举增益。”

---

## 项目描述

> “GPT 自调度、GPT 多代理协同；自主创建与持久化工具、动态记忆管理、Diff 驱动增删改；AI 办公自动化（排版、备忘录、PPT 制作）、智能笔记整理与标注；智能自举增益与 O 系列模型复现；个人知识分布建模……旨在构建通向 AGI 的闭环引擎。”

---

## 核心功能亮点

* **GPT 全自主调度：**
  多代理协同，主→子→主循环，自动分解复杂任务；自主编写 Prompt 与代码工具；自主管理调参与调度。

* **智能自举增益**
  基于类 O 系列模型的工具创建与调度实验，实现“用工具增强工具”闭环。

* **自创工具库：**
  自动检测并封装重复逻辑，提供 `create_tool` 一键生成、注册与持久化。

* **动态记忆管理**
  分层存储（工作/情境/知识），结合关键词标签、Embedding 等多种方式动态召回。

* **改良 Diff 驱动编辑套件：**
  以首尾关键字段为锚定位，精确进行增删改补；支持自动回滚与测试驱动优化。

* **AI 办公自动化：**
  智能备忘录、AI Office 套件（文档排版、Diff 文档、PPT 自动生成）。

* **AI JS 逆向工具：**
  自主 AST 交互分析，破解超长 JS 代码。

* **智能笔记整理：**
  自动抽取摘要、生成记忆管理标签、代码进行“mcp式规范化处理”。

* **个性化学习建模：**
  构建用户知识分布模型，预测个人关注重点、盲区及潜在提问。

* **GPT-Playright浏览器控制方案（待扩展）**
  计划接入浏览器，进行复杂任务处理。

* **视觉方案及反馈标签（待扩展）**
  计划接入视觉识别，完善反馈标签系统，提升多模态能力。

```text
          ┌─────────────────────────────┐
          │  改良 Diff 代码增量编辑器   │
          │  备忘录 / 操作日志          │
          └────────────┬────────────────┘
                       │ 上传
                       ▼
         ┌─────────────┐       ⟵ 调用 / 整理
         │   记忆库     │◄─────────────────┐
         │ (Memory DB) │                   │
         │─────────────│                   │
         │ • 存储增量编辑             │
         │ • 存储备忘录 & 日志        │
         │ • 价值代码标注            │
         └──────┬──────┘                   │
                │                          ▼
                │           ┌────────────────────────┐
                └──────────▶│         类 O 模型       │
                            │   （推理 & 决策）       │
                            │────────────────────────│
                            │ • 动态构建记忆          │
                            │ • 自主创建工具          │
                            │   – 整理 & 更新记忆库   │
                            │   – 注册代码至 MCP     │
                            └───────┬────────────────┘
                                    │ 注册代码至
                                    ▼
                            ┌────────────────────────┐
                            │    MCP 服务器集群       │
                            │（代码存储 & 部署服务）  │
                            └────────────────────────┘
```

---

## 🧭 项目愿景

构建一套完整、低成本的信息管理与效率提升方案，打造 AI 驱动的学习与办公工具闭环：

1. **全自主智能体**：自创建工具、自编 Prompt、自管理记忆、自运行代码与多媒体操作，实现“自举式进步”。
2. **增量编辑器**：基于改良 Diff，以关键词锚点精确编辑；自动上传至记忆库，持续优化。
3. **AI 备忘录**：类似文件传输助手，结构化管理所有消息与日志，支持语义检索与记忆输入。
4. **个性化标注插件**：采集高亮与标注习惯，生成微调语料，预测用户知识分布与关注偏好。
5. **JS 逆向分析**：结合 AST 工具与 GPT，自动化执行 JS 逆向工程。
6. **智能学习助手**：

   * 建模用户知识掌握与提问风格
   * 主动提示未知概念
   * 自动整理笔记、代码，生成结构化文档
   * 抽象可复用工具，减少重复劳动
   * 以反思驱动自我演进，使用越多越聪明

---

> **愿景**：通过“自省 → 记忆 → 抽象 → 创造 → 反馈”五步闭环，让 AI 在使用中不断自举，逐步迈向 AGI。

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

## 🔁 [类gpt-O系列模型](https://github.com/kay-cottage/Cognitive-GPT-Toolkit/tree/main/Class_O_Model) GPT 调度 GPT（智能自举增益）

通过 Coordinator 代理调度：

* 主代理负责任务规划；
* 子代理承担专职任务（检索、代码、排版、测验生成）；
* 结果回流主循环并进入 `reflect`；
* 子代理性能数据用于自动权重选择（哪个模型擅长哪个子任务）。

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



## 🧪 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/kay-cottage/Cognitive-GPT-Toolkit
cd Cognitive-GPT-Toolkit
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



\*\*下一步：\*\*告诉我你想采用哪个仓库名，我可以帮你生成 `pyproject.toml`、`requirements.txt`、基础目录结构脚本，或继续扩写 docs！
