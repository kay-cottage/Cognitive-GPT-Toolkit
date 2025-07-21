# 此文档用于说明改良版Diff增量内容编辑器


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


### 理想效果

* Token 成本显著下降。
* 工具使用命中率提升。
* 支持插件生态：开发者新增工具后自动进入检索库。

---


未来扩展：

* AST 层语义补丁；
* 单测驱动验证；
* 多代理代码审查。

--


