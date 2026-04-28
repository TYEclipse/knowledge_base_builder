# API_REFERENCE.md

## 1. 命令行接口

### 1.1 主命令

`python knowledge_base_builder.py --topic <主题> [可选参数]`

### 1.2 参数说明

- `--topic` (必填)：知识库主题
- `--audience`：`beginner | intermediate | advanced`
- `--output`：JSONL 输出路径（默认 `./knowledge_base.jsonl`）
- `--markdown-output`：Markdown 输出目录
- `--resume`：断点续传起始题号
- `--max-questions`：最多处理问题数
- `--stream / --no-stream`：是否开启流式输出
- `--verbose / --no-verbose`：日志级别

## 2. 环境变量

- `MOONSHOT_API_KEY`：Kimi API Key（必填）
- `MOONSHOT_BASE_URL`：可选，默认 `https://api.moonshot.cn/v1`
- `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`：可选代理

## 3. 输出数据格式（JSONL）

### 3.1 Phase 1 记录

```json
{"phase":1,"type":"research","topic":"...","timestamp":"...","summary":"..."}
```

### 3.2 Phase 2 记录

```json
{"phase":2,"type":"question_list","level":"beginner","topic":"...","timestamp":"...","questions":["..."]}
```

### 3.3 Phase 3 记录

```json
{"phase":3,"type":"analysis","id":1,"level":"beginner","question":"...","analysis":"...","key_points":["..."],"sources":["..."],"difficulty":"beginner","timestamp":"..."}
```

## 4. 内部类接口（摘要）

- `KnowledgeBaseBuilder.run()`：执行完整流程
- `KimiApiClient.call()`：统一 API 调用入口
- `QuestionGenerator.generate()`：生成三级问题清单
- `AnswerAnalyzer.run()`：逐题分析并写出结果
- `AtomicJsonlWriter.append()/flush()`：原子写入 JSONL
