# Data Agent Notebook Review

这一版以 **notebook 本地验证** 为主，不提供 CLI 或直接调用 Dify 的业务接口。目标是先把 PDF 解析、OCR、分块、Milvus 入库和 Dify 本地上传文件导出流程跑通，便于 review。

## 当前流程

1. 在 notebook 中配置一个或一批本地 PDF。
2. 对扫描版 PDF 使用开源、可本地部署的 PaddleOCR，本地依赖以 `paddlepaddle==3.3.0` 为准。
3. 按文档类型分块：`manual`、`policy`、`contract`、`faq`、`generic`。
4. 导出本地 Dify 上传文件：`.txt` 便于直接上传 Dify Knowledge，`.jsonl` 保留 chunk metadata 便于检查。
5. 可选：用本地 BGE embedding 将 chunks 写入 Milvus。
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
pip install paddlepaddle==3.3.0 paddleocr pymupdf
```

> `pymupdf` 用于把 PDF 页面渲染成图片，再交给 PaddleOCR 识别。

## Notebook

主 notebook：

```text
notebooks/pdf_to_milvus_and_dify_rag.ipynb
```

Notebook 显式分成以下功能块：

- 本地 PDF 路径与批量 PDF 配置。
- PaddleOCR 本地 OCR 函数。
- 文本抽取 + OCR fallback + 分块。
- 目录页解析检查。
- Dify 本地上传文件导出。
- 可选 Milvus 入库功能块。
- 批量 PDF 使用方式。

默认先测试一个 PDF：

```python
PDF_FILES = [Path("./docs/manual.pdf").resolve()]
```

批量处理时改为：

```python
PDF_FILES = sorted(Path("./docs").glob("*.pdf"))
```

## Dify 使用方式

当前版本不直接请求 Dify API。Notebook 会导出本地文件：

- `outputs/dify_upload/<pdf_name>.dify.txt`：推荐先手工上传 Dify Knowledge 测试。
- `outputs/dify_upload/<pdf_name>.dify.jsonl`：保留 `content`、`keywords`、`metadata`，用于 review 或后续脚本导入。

## Milvus 使用方式

Notebook 保留 Milvus 功能块，但真正入库语句默认注释。确认本地 BGE 和 Milvus 都启动后，再取消注释：

```python
# for pdf_path, chunks in all_chunks_by_pdf.items():
#     upserted = kb.upsert_chunks(chunks)
#     print(pdf_path.name, "upserted:", upserted)
```
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

## 图片型/扫描版 PDF 如何处理和组织

如果 PDF 页面像截图一样，本质是扫描图片，`pypdf` 往往抽不出文本。推荐流程是：

1. 先对 PDF 页面做 OCR，得到每页文本，并保留页码、版面块坐标等信息。
2. 对目录页使用 `parse_toc_entries()` 识别 `1.1 标题 ...... 1-1` 这类目录行，形成章节号、标题、页码标签和层级。
3. 正文使用 `--document-type manual`，按 `1`、`1.1`、`1.1.1` 等编号切割。
4. 入库 metadata 建议保存 `document_type=manual`、`chapter_number`、`section_title`、`page_number`、`page_label`、`bbox`、`ocr_engine`、`ocr_confidence` 等字段。
5. 问答检索时先用 metadata 过滤手册类型或章节范围，再召回 chunk，最后由 LLM 基于证据回答。

目录文本解析示例：

```python
from data_agent import parse_toc_entries

ocr_text = """
1 安全说明 ........ 1-1
1.1 人员防护 ........ 1-1
1.1.1 个人防护装备 ........ 1-1
"""

entries = parse_toc_entries(ocr_text)
for entry in entries:
    print(entry.number, entry.title, entry.page_label, entry.level)
```

## 代码调用方式

当前工具既提供 CLI，也提供可被业务系统直接调用的 Python 接口。典型用法：

```python
from data_agent import LocalBGEEmbeddingProvider, RuleKnowledgeService

embedding = LocalBGEEmbeddingProvider(
    api_url="http://localhost:8000/v1/embeddings",
    model="bge-m3",
    dimension=1024,
)

service = RuleKnowledgeService.connect(
    milvus_uri="http://localhost:19530",
    collection_name="rule_knowledge_base",
    embedding_provider=embedding,
)

service.ingest_file(
    "./docs/manual.pdf",
    document_type="manual",
    document_format="pdf",
    metadata={"department": "maintenance", "version": "2026"},
)

prompt = service.answer_prompt(
    "焊前需要做哪些安全检查？",
    limit=5,
    metadata_filter='metadata["document_type"] == "manual"',
)
print(prompt)
```

## OCR 当前怎么接入

当前项目不在进程内绑定某个 OCR 引擎，而是通过 `LocalOcrApiProvider` 对接本地 OCR HTTP 服务。这样可以按部署环境选择 PaddleOCR、Tesseract、云厂商 OCR 或企业内部版面分析服务。

默认约定：

```text
POST http://localhost:8001/ocr/pdf
{"path":"/absolute/path/manual.pdf"}
```

OCR 服务返回分页文本：

```json
{
  "pages": [
    {
      "page_number": 1,
      "text": "1 安全说明 ........ 1-1\n1.1 人员防护 ........ 1-1",
      "metadata": {"ocr_engine": "paddleocr", "ocr_confidence": 0.96}
    }
  ]
}
```

解析器处理 PDF 时会先尝试普通文本抽取；如果文本为空，并且传入了 OCR provider，则自动调用 OCR，再用 OCR 文本继续按 `document_type` 切割。CLI 用法：

```bash
rule-kb ingest ./docs/scanned-manual.pdf \
  --document-type manual \
  --document-format pdf \
  --ocr-api-url http://localhost:8001/ocr/pdf
```

代码调用方式：

```python
from data_agent import LocalBGEEmbeddingProvider, LocalOcrApiProvider, RuleKnowledgeService

service = RuleKnowledgeService.connect(
    milvus_uri="http://localhost:19530",
    collection_name="rule_knowledge_base",
    embedding_provider=LocalBGEEmbeddingProvider(),
)

service.ingest_file(
    "./docs/scanned-manual.pdf",
    document_type="manual",
    document_format="pdf",
    ocr_provider=LocalOcrApiProvider("http://localhost:8001/ocr/pdf"),
    metadata={"source_kind": "scanned_pdf"},
)
```

## 同时支持 Milvus 和 Dify RAG

分块后的内容可以走两条链路：

1. **本项目自管 RAG**：对 chunk 调本地 BGE embedding，然后写入 Milvus，后续用 `rule-kb search` 或 `RuleKnowledgeService.answer_prompt()` 检索。
2. **Dify RAG**：复用同一批 chunk，导出为 Dify 友好的 segment JSONL，或通过 `DifyKnowledgeClient` 调 Dify Knowledge API 创建文本型文档，让 Dify 负责索引和问答应用编排。

导出 JSONL：

```bash
rule-kb export-dify-jsonl ./docs/manual.pdf \
  --output ./manual.dify.jsonl \
  --document-type manual \
  --document-format pdf \
  --metadata '{"department":"maintenance"}'
```

代码调用 Dify：

```python
from data_agent import DifyKnowledgeClient, RuleKnowledgeService

client = DifyKnowledgeClient(
    api_base_url="https://your-dify.example.com/v1",
    api_key="DIFY_KNOWLEDGE_API_KEY",
)

service.push_file_to_dify(
    "./docs/manual.pdf",
    dify_client=client,
    dataset_id="your-dataset-id",
    document_type="manual",
    document_format="pdf",
)
```

如果你希望完全保留本工具已经切好的 chunk，可以先用 Dify 的 `create-by-text` 创建一个空/合并文本 document，再用 `create_segments()` 将 `chunks_to_dify_segments()` 的结果作为自定义 segments 写入对应 document。

## Notebook review 示例

`notebooks/pdf_to_milvus_and_dify_rag.ipynb` 提供了端到端本地测试说明：读取一个本地 PDF，配置 PaddleOCR 风格的本地 OCR HTTP 接口，配置本地 BGE embedding 接口，显式展示解析分块、目录解析、Milvus 入库、Dify JSONL 导出和 Dify Knowledge API 调用代码块。Notebook 中真实入库和真实 Dify API 调用默认注释，便于 review 时逐块检查后再手动执行。
