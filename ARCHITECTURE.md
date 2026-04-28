# ARCHITECTURE.md

## 1. 架构目标

本项目按 ISO/IEC 25010 质量属性重构，目标是：

- 可靠：可重试、可恢复、可校验
- 可维护：模块化、类型化、日志化
- 可测试：核心逻辑可 Mock、可单测

## 2. 模块划分

- `knowledge_base_builder.py`：主流程编排（`KnowledgeBaseBuilder`）
- `config.py`：配置加载、参数校验、路径安全、日志
- `api_client.py`：Kimi API 调用封装（重试、超时、工具调用闭环）
- `question_generator.py`：阶段二问题清单生成
- `answer_analyzer.py`：阶段三逐题分析
- `models.py`：Pydantic 数据模型与 JSON Schema
- `storage.py`：JSONL 原子写入与 Markdown 输出

## 3. 系统架构图（文字）

```text
CLI参数 -> config.Settings
          -> KnowledgeBaseBuilder
               |- KimiApiClient
               |- QuestionGenerator
               |- AnswerAnalyzer
               |- AtomicJsonlWriter
               |- MarkdownWriter

Phase1(Research) -> JSONL写入 -> Markdown摘要
Phase2(Questions)-> JSON Schema校验 -> JSONL写入 -> Markdown清单
Phase3(Analyze)  -> 搜索+分析 -> JSON Schema校验 -> JSONL原子写入 -> Markdown答案
```

## 4. 调用关系（文本）

1. `main()` 解析参数并创建 `Settings`
2. `KnowledgeBaseBuilder.run()` 依次调用：
   - `phase1_research()`
   - `phase2_questions()`（或 `resume` 恢复）
   - `AnswerAnalyzer.run()`
3. `QuestionGenerator` 与 `AnswerAnalyzer` 均通过 `KimiApiClient.call()` 发起请求
4. 所有阶段输出通过 `AtomicJsonlWriter` 统一落盘

## 5. 关键质量机制

- 重试：`tenacity` 指数退避（最多 3 次）
- 超时：`httpx.Timeout(connect/read/write/pool)`
- 校验：`jsonschema` 校验 Phase2/Phase3 响应结构
- 原子写入：临时文件 + `os.replace`
- 安全：API Key 不硬编码、日志脱敏、输出路径防穿越
