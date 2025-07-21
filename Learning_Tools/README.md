# 此文档用于说明AI有关的学习工具


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
