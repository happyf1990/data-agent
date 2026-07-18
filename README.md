# Data Agent Notebook Review

这一版以 **notebook 本地验证** 为主，不提供 CLI 或直接调用 Dify 的业务接口。目标是先把 PDF 解析、OCR、分块、Milvus 入库和 Dify 本地上传文件导出流程跑通，便于 review。

## 当前流程

1. 在 notebook 中配置一个或一批本地 PDF。
2. 对扫描版 PDF 使用开源、可本地部署的 PaddleOCR，本地依赖以 `paddlepaddle==3.3.0` 为准。
3. 按文档类型分块：`manual`、`policy`、`contract`、`faq`、`generic`。
4. 导出本地 Dify 上传文件：`.txt` 便于直接上传 Dify Knowledge，`.jsonl` 保留 chunk metadata 便于检查。
5. 可选：用本地 BGE embedding 将 chunks 写入 Milvus。

## 安装

### Notebook 调用是否必须 `pip install -e .`？

不必须。Notebook 第 0 步会把仓库 `src/` 加到 `sys.path`，所以在仓库根目录启动 notebook 时，可以直接 import 当前源码。

如果只做 notebook review，安装第三方依赖即可：

```bash
pip install paddlepaddle==3.3.0 paddleocr pymupdf pymilvus pypdf python-docx
```

如果希望在任意目录都能 `import data_agent`，或希望用 editable package 方式开发，再执行：

```bash
pip install -e '.[dev]'
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
