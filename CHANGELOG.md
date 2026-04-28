# CHANGELOG.md

## [0.2.0] - 2026-04-28

### Added

- 新增模块化架构：`config.py`、`api_client.py`、`question_generator.py`、`answer_analyzer.py`、`models.py`、`storage.py`
- 新增 JSON Schema 校验（Phase2/Phase3）
- 新增 JSONL 原子写入器（临时文件 + 替换）
- 新增 pytest 测试与 GitHub Actions CI
- 新增工程文档：INSTALL / ARCHITECTURE / API_REFERENCE / TEST_PLAN / CONTRIBUTING / SECURITY

### Changed

- 主脚本重构为面向对象编排（`KnowledgeBaseBuilder`）
- 引入结构化日志与敏感信息脱敏
- 增强 CLI 参数校验和输出路径安全检查
- 增强超时配置与错误处理策略

### Fixed

- 修复原实现写入过程异常中断时数据一致性风险
- 修复弱校验导致 JSON 响应结构不确定的问题
