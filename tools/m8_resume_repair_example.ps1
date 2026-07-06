# 进入项目根目录后执行
# 先生成修复计划
python .\tools\m8_resume_repair.py plan `
  --batch-dir .\runs\batch\YOUR_M8_BATCH_DIR

# 再执行续跑修复（默认读取 .\runs\batch\YOUR_M8_BATCH_DIR\_repair\m8_resume_qids.txt）
python .\tools\m8_resume_repair.py repair `
  --batch-dir .\runs\batch\YOUR_M8_BATCH_DIR `
  --agent-config .\config\agent.yaml `
  --library-root .\library

# 修复后重新评测；由于修复脚本已删除被修复题目的 judge cache，--skip-existing 是安全的
python .\run.py eval-batch `
  --batch-dir .\runs\batch\YOUR_M8_BATCH_DIR `
  --agent-config .\config\agent.yaml `
  --judge `
  --skip-existing

# 如果 plan 阶段还发现了无法续跑修复的题（例如根本没有 qa_result.json），导出一个最小 q0 子批次
python .\tools\m8_resume_repair.py export-rerun-batch `
  --batch-dir .\runs\batch\YOUR_M8_BATCH_DIR
