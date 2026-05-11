# LinkedIn Post Draft

---

🔬 Built an AI-powered literature review engine that actually verifies its citations.

After months of iteration, I'm open-sourcing **LitScribe** — a multi-agent system that generates comprehensive academic literature reviews from a single research question.

**What makes it different from "just asking ChatGPT"?**

→ Searches 6 academic databases (325M+ papers) — not relying on LLM's training data
→ Every citation is verified against the source paper (83-100% grounding accuracy)
→ Automatically detects contradictions between papers and presents them as critical analysis
→ Reviewer agent debates with synthesizer agent to improve quality (2 rounds)
→ System decides which steps to re-run if quality is low (metacognitive loop)

**The output includes:**
• Thematic review with Pandoc [@key] citations + BibTeX
• Methodology comparison table
• Research timeline (foundational → frontier)
• Statistical summary + suggested figures

**Benchmark (5 domains × 12 papers):**
Biology: score 0.82 | CS: 0.72 | Medicine: 0.65 | Chemistry: 0.55
Average ~2 min per review, 1,500-2,000 words

**Tech stack:** DeepAgents + DeepSeek V4 + 6 search APIs + GraphRAG + FastAPI

Three things no other tool does:
1. Contradiction-aware synthesis (detect + weave into narrative)
2. Search-augmented refinement ("add a section about X" → searches new papers first)
3. Metacognitive quality loop (system reasons about what went wrong)

Try it: github.com/arnold117/LitScribe

Built with Python, open-source (MIT). Feedback welcome.

#AI #AcademicResearch #LiteratureReview #NLP #OpenSource #DeepSeek #MultiAgent

---

## 中文版

🔬 做了一个能验证引用真实性的 AI 文献综述引擎。

开源了 **LitScribe** — 输入研究问题，自动搜索论文、分析、写综述、验证引用。

**和直接问 ChatGPT 的区别？**

→ 搜 6 个学术数据库（3.25 亿+论文），不靠大模型记忆
→ 每条引用都和原文核对（准确率 83-100%）
→ 自动发现论文之间的矛盾观点，写进综述的批判性分析
→ Reviewer 和 Synthesizer 两个 agent 辩论 2 轮提升质量
→ 系统自己判断哪步该重跑（元认知循环）

**输出包含：**
• 主题综述 + [@key] 引用 + BibTeX
• 方法论对比表 + 研究时间线
• 统计摘要 + 插图建议

**5 领域基准测试：**
生物 0.82 | 计算机 0.72 | 医学 0.65 | 化学 0.55
每篇综述约 2 分钟，1500-2000 字

三个没有其他工具做到的能力：
1. 矛盾感知综合 — 检测矛盾 + 织入叙事
2. 搜索增强修改 — "加一段 delivery methods" → 先搜论文再写
3. 元认知质量循环 — 系统推理哪里出了问题

试试：github.com/arnold117/LitScribe

Python 开源（MIT），欢迎反馈。

#AI #学术研究 #文献综述 #开源 #多智能体
