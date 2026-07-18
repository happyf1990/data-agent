# Data Agent Rule Knowledge Base

一个面向智能问答的规则文档处理工具：解析 PDF/DOCX 制度文件，按制度类型选择对应切割策略，使用本地 BGE API 生成 embedding，写入 Milvus 向量知识库，并保留元数据用于过滤、溯源和回答引用。

## 功能

- PDF、DOCX 文档解析；也可通过 `--document-format pdf|docx` 显式指定解析格式。
- 支持不同制度文件类型的切割策略：`policy`（制度/规章）、`contract`（合同/协议）、`faq`（问答手册）、`generic`（通用文本）。
- 规则标题识别：支持“第一章”“第 1 条”“1.2”“一、”等常见制度编号；合同策略额外识别甲方/乙方、违约责任、争议解决等条款；FAQ 策略识别 Q/问/问题。
- 可配置 chunk 大小与重叠窗口。
- 默认调用本地 BGE embedding API，并将向量写入 Milvus。
- Milvus 向量存储记录 `text`、`embedding`、`metadata`。
- 元数据包含文件名、文件类型、原始路径、文档类型、解析格式、章节标题、页码、chunk 序号和自定义业务字段。
- 检索结果可生成中文 grounded Q&A 提示词，要求答案基于证据并附引用编号。

## 安装

```bash
pip install -e '.[dev]'
```

## 本地 BGE API

默认 embedding provider 会请求 OpenAI-compatible 的本地 BGE 接口：

```text
POST http://localhost:8000/v1/embeddings
{"model":"bge-m3","input":["文本1","文本2"]}
```

可通过环境变量或命令行覆盖：

```bash
export BGE_API_URL=http://localhost:8000/v1/embeddings
```

## 写入知识库

```bash
rule-kb ingest ./docs/rule.pdf ./docs/policy.docx \
  --milvus-uri http://localhost:19530 \
  --collection rule_knowledge_base \
  --embedding bge \
  --bge-api-url http://localhost:8000/v1/embeddings \
  --bge-model bge-m3 \
  --document-type policy \
  --document-format auto \
  --metadata '{"department":"hr","version":"2026"}'
```

如果同为 PDF/DOCX 但内容结构不同，可切换制度类型来改变切割方式：

```bash
# 合同/协议：尽量围绕条款、甲乙方、责任等块切割
rule-kb ingest ./docs/contract.pdf --document-type contract

# FAQ/操作问答：尽量围绕问题-答案块切割
rule-kb ingest ./docs/faq.docx --document-type faq --document-format docx
```

`--embedding hash` 仅用于离线单元测试或无 BGE 服务时的本地验证，不建议生产使用。

## 检索问答上下文

```bash
rule-kb search "员工年假如何计算？" \
  --milvus-uri http://localhost:19530 \
  --collection rule_knowledge_base \
  --embedding bge \
  --limit 5
```

可以通过 Milvus filter 对元数据过滤，例如仅检索 HR 制度文档：

```bash
rule-kb search "员工年假如何计算？" --filter 'metadata["department"] == "hr" and metadata["document_type"] == "policy"'
```
