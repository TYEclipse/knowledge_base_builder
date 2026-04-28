# TEST_PLAN.md

## 1. 测试目标

验证重构后在可靠性、可维护性、可测试性、安全性方面满足预期。

## 2. 测试范围

- 单元测试：API 客户端、JSON 解析
- 集成测试：主流程关键路径（后续扩展）
- 兼容性测试：Python 3.8/3.10/3.12

## 3. 测试环境

- OS：Windows 11 / Ubuntu / macOS
- Python：3.8 - 3.12
- 工具：pytest、GitHub Actions

## 4. 测试用例清单

| 用例ID    | 模块       | 场景                     | 预期结果       |
| --------- | ---------- | ------------------------ | -------------- |
| UT-API-01 | api_client | Markdown 包裹 JSON 解析  | 正确解析对象   |
| UT-API-02 | api_client | 空 choices 提取文本      | 返回默认值     |
| UT-API-03 | api_client | call 统计 usage          | 统计字段递增   |
| UT-JSON-01| api_client | 非法 JSON                | 抛出异常       |
| UT-JSON-02| api_client | 空对象 JSON              | 返回 `{}`      |

## 5. 通过准则

- pytest 全部通过
- CI 三个 Python 版本任务通过
- 无阻塞级（P0）缺陷

## 6. 后续建议

- 增加 `QuestionGenerator` 与 `AnswerAnalyzer` 的 Mock 集成测试
- 增加断点续传与原子写入恢复测试
- 覆盖率目标提升到 ≥ 70%
