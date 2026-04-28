# INSTALL.md

## 1. 概述

本文档说明 `knowledge_base_builder` 在 Windows、Linux、macOS 上的安装与运行步骤。

## 2. 前置条件

- Python 3.8 - 3.12
- 可用的 Moonshot/Kimi API Key
- 能访问 `https://api.moonshot.cn/v1`

## 3. 安装步骤

### 3.1 Windows PowerShell

1. 创建虚拟环境：`python -m venv .venv`
2. 激活：`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` 后执行 `\.venv\Scripts\Activate.ps1`
3. 安装依赖：`python -m pip install --upgrade pip` 与 `python -m pip install -r requirements.txt`
4. 创建配置：复制 `.env.example` 为 `.env`，填写 `MOONSHOT_API_KEY`

### 3.2 Linux/macOS (bash/zsh)

1. 创建虚拟环境：`python3 -m venv .venv`
2. 激活：`source .venv/bin/activate`
3. 安装依赖：`python -m pip install --upgrade pip` 与 `python -m pip install -r requirements.txt`
4. 创建配置：复制 `.env.example` 为 `.env`，填写 `MOONSHOT_API_KEY`

## 4. 运行验证

执行：`python knowledge_base_builder.py --topic "量子计算" --max-questions 5`

成功时会生成：

- `knowledge_base.jsonl`
- `knowledge_base_markdown/`

## 5. 代理与网络

项目自动读取系统代理环境变量（如 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY`）。

## 6. 常见问题

- 缺少依赖：确认已在虚拟环境中执行 `pip install -r requirements.txt`
- Key 无效：检查 `.env` 中 `MOONSHOT_API_KEY`
- 路径安全错误：输出路径需位于项目目录内
