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
python -m pip install openai tenacity tqdm
```

如果遇到 `ReadTimeoutError`（例如访问 `pypi.org` 超时），可临时使用清华镜像：

```powershell
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple openai tenacity tqdm
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
python .\knowledge_base_builder.py --topic "Kubernetes" --max-questions 10
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

---

## 7. 退出虚拟环境

```powershell
deactivate
```

---

## 8. 项目文件（当前）

- `knowledge_base_builder.py`：主脚本
- `.env`：本地私有配置（自动加载，不要提交）
- `.env.example`：环境变量示例模板（可提交）
- `.gitignore`：Git 忽略规则（已包含 `.env`）
- `README.md`：本说明文档

祝你构建知识库顺利 🚀
