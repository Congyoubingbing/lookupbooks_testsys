"""lookupbooks_sys

落地 note.docx 的多书问答流程（Step0~Step4），并作为上层系统把
pdf_txt_align_qwen_checked(v12.5.0) 的章节切分输出组织为可用于后续问答/检索的 library 结构。

核心能力：
- library 导入：把 outputs/<book_id>/chapters/*.txt 统一编号为 book{i}/chapter{j}.txt
- 生成摘要：summarybook.txt + book{i}_chapter_summary.txt（每条约 100~200 字）
- 问答闭环：严格执行 note.docx Step0~Step4（含 A/B/C 判定与答案择优）

注意：本包复用 pdf_txt_align 的 ApiPool/TaskSession 作为 API 调度策略：
“一个 pipeline 占用一个 key，完成才释放”。
"""

__version__ = "0.5.1"
