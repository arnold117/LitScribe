# LitScribe

Multi-agent literature review engine powered by DeepAgents.

## Pipeline
plan → discover → read → graphrag → synthesize → review (loop-back if needed)

## Conventions
- DashScope (百炼) API for LLM, default qwen-plus
- Papers: SQLite + ChromaDB vectors
- Skills evolve via EMA success rates across sessions
- Support English + Chinese research questions
- Always call check_pipeline_status after each pipeline step
- Follow the recommendation field from check_pipeline_status
