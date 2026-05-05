# Kimi API 知识库构建器（Windows 版）

基于 Kimi API（`kimi-k2.6`）自动构建结构化知识库（JSON Lines 格式）。

核心流程：

1. 主题调研（联网搜索）
2. 三级问题清单生成（初级/中级/高级）
3. 逐题深度分析并写入 JSONL

---

## 1. 环境要求

- Windows 10/11
- Python 3.10+（建议 3.11 或以上）
- 可用的 Moonshot/Kimi API Key

---

## 2. 快速开始（PowerShell）

> 以下命令均在项目根目录执行：`D:\TY_Files\projects\knowledge_base_builder`

### 2.1 创建并激活虚拟环境

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2.2 安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果遇到 `ReadTimeoutError`（例如访问 `pypi.org` 超时），可临时使用清华镜像：

```powershell
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt
```

如需长期使用镜像（推荐国内网络环境）：

```powershell
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

> 注意：镜像 URL 末尾的 `simple` 不能省略，且必须使用 `https`。

### 2.3 配置 API Key

推荐方式：在项目根目录使用 `.env` 文件配置（脚本会自动读取）。

先复制示例文件：

```powershell
Copy-Item .\.env.example .\.env
```

然后编辑 `.env`，填入真实密钥：

```dotenv
MOONSHOT_API_KEY=你的-api-key
# 可选：不填则默认使用 https://api.moonshot.cn/v1
# MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
```

你也可以临时使用环境变量（但脚本启动时会优先读取并采用 `.env` 中的值）：

```powershell
$env:MOONSHOT_API_KEY = "你的-api-key"
```

### 2.4 运行脚本

```powershell
python .\knowledge_base_builder.py --topic "量子计算"
```

---

## 3. 常用运行示例（Windows）

### 基础用法

```powershell
python .\knowledge_base_builder.py --topic "量子计算"
```

### 指定受众和输出文件

```powershell
python .\knowledge_base_builder.py --topic "React 18" --audience intermediate --output .\react_kb.jsonl
```

### 断点续传（从第 50 个问题继续）

```powershell
python .\knowledge_base_builder.py --topic "Docker" --resume 50
```

### 快速测试（仅处理 10 个问题）

```powershell
python .\knowledge_base_builder.py --topic "哲学" --max-questions 3
```

---

## 4. 参数说明

- `--topic`（必填）：知识库主题
- `--audience`：目标受众级别，可选：`beginner` / `intermediate` / `advanced`
- `--output`：输出 JSONL 文件路径（默认：`./knowledge_base.jsonl`）
- `--resume`：断点续传起始序号（默认：`0`）
- `--max-questions`：最大问题数（默认：`300`）

---

## 5. 输出格式说明

输出为 JSON Lines（每行一个 JSON 对象），主要包含三类记录：

- `phase = 1`：主题调研摘要
- `phase = 2`：三级问题清单
- `phase = 3`：逐题深度分析结果

你可以用任意支持 JSONL 的工具做后处理、检索或向量化。

---

## 6. 常见问题（Windows）

### 6.1 激活虚拟环境时报脚本执行策略错误

先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

然后再执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 6.2 提示未找到 `MOONSHOT_API_KEY`

优先检查项目根目录 `.env` 是否存在且包含：

```dotenv
MOONSHOT_API_KEY=你的-api-key
```

如果你在终端里设置过旧值，请优先更新 `.env`，或先清理当前会话中的旧环境变量后再运行：

```powershell
Remove-Item Env:MOONSHOT_API_KEY -ErrorAction SilentlyContinue
$env:MOONSHOT_API_KEY = "你的-api-key"
```

并在同一个终端会话中运行脚本。

### 6.3 网络或 API 调用偶发失败

脚本已内置重试机制（指数退避）。如失败可直接重跑，或用 `--resume` 从中断点继续。

### 6.4 `pip install` 超时或下载很慢

先确认网络可访问外网；若不稳定，建议切换到镜像源：

```powershell
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install --upgrade pip
python -m pip install openai tenacity tqdm
```

查看当前 pip 源配置：

```powershell
pip config list
```

### 6.5 怀疑 `$web_search` 没有正常调用，如何打开调试日志

在 `.env` 中增加以下配置（推荐）：

```dotenv
# 开启联网工具调试日志
KIMI_TOOL_DEBUG=1

# 单条调试日志最大字符数（可选，默认 800）
KIMI_TOOL_DEBUG_MAX_CHARS=1200
```

开启后会输出：

- 每一轮 `finish_reason=tool_calls` 的轮次与工具数量
- 每个工具调用的 `id/name/arguments`（截断预览）
- `$web_search` 返回参数中的 `usage.total_tokens`（若有）
- 回填给 `role=tool` 的 `content`（截断预览）
- 联网完成后的 `prompt/completion/total tokens`

> 调试日志默认关闭；只有 `KIMI_TOOL_DEBUG=1` 时才会输出。

---

## 7. 退出虚拟环境

```powershell
deactivate
```

---

## 8. 项目文件（重构后）

- `knowledge_base_builder.py`：主入口与流程编排（`KnowledgeBaseBuilder`）
- `config.py`：配置、参数校验、日志与路径安全
- `api_client.py`：Kimi API 客户端封装（重试、超时、联网工具闭环）
- `question_generator.py`：三级问题清单生成
- `answer_analyzer.py`：逐题搜索与分析
- `models.py`：Pydantic 数据模型与 JSON Schema
- `storage.py`：JSONL 原子写入与 Markdown 输出
- `tests/`：pytest 单元测试
- `.github/workflows/ci.yml`：CI 自动化测试
- `.env`：本地私有配置（自动加载，不要提交）
- `.env.example`：环境变量示例模板（可提交）
- `.gitignore`：Git 忽略规则（已包含 `.env`）
- `README.md`：本说明文档

---

## 9. 质量属性（ISO/IEC 25010）

本项目在重构后重点覆盖以下质量属性：

- **Reliability（可靠性）**：tenacity 指数退避重试、请求超时控制、JSON Schema 响应校验、JSONL 原子写入。
- **Maintainability（可维护性）**：模块化拆分、面向对象主流程、完整类型注解与 Google-style docstring、结构化日志。
- **Testability（可测试性）**：pytest 单测、可注入客户端便于 Mock、CI 自动执行测试。
- **Portability（可移植性）**：兼容 Windows/Linux/macOS，支持 Python 3.8-3.12，支持代理环境变量。
- **Security（安全性）**：API Key 仅环境变量/.env、日志脱敏、输出路径安全检查（防路径穿越）。
- **Usability（易用性）**：文档补齐（INSTALL/ARCHITECTURE/API_REFERENCE/TEST_PLAN 等），CLI 参数说明与错误提示优化。

---

## 10. 扩展文档

- `INSTALL.md`：三平台安装指南
- `ARCHITECTURE.md`：架构设计说明
- `API_REFERENCE.md`：接口与数据格式
- `TEST_PLAN.md`：测试计划与用例
- `CHANGELOG.md`：版本变更记录
- `CONTRIBUTING.md`：贡献流程
- `SECURITY.md`：安全策略

祝你构建知识库顺利 🚀
