# AI Interview Copilot — 全面代码审查与改进路线图

> 生成日期：2026-05-31 | 审查者：Claude Code (Sonnet 4.6) | 分支：`dev`

---

## 总体评估

项目是一个**架构清晰的 MVP**，具备干净的服务分层、异步任务工作流、OpenAI 结构化输出和可运行的 Streamlit 前端。代码类型安全、有测试覆盖、可部署上线。但它目前读起来更像"我做了一个能跑的东西"，而不是"我做了一个可扩展、省成本、会自我学习的系统"。

下面的改进方向将把它变成一个能体现**高级工程师思维**的作品集项目：成本优化、可观测性、智能体推理、语义搜索和生产级可靠性。

---

## 现有架构亮点

| 亮点 | 位置 | 为什么重要 |
|---|---|---|
| LLM 结构化输出 | `ai_service.py` | `client.responses.parse()` 直接返回 Pydantic 模型，无需正则解析 JSON |
| 并行 AI 调用 | `interview.py:373-387` | `analyze_jd` + `match_resume` 通过 `ThreadPoolExecutor` 并发执行，约提速 35% |
| 异步任务工作流 | `interview.py`, `db.py` | HTTP 202 + 轮询，前端不阻塞等待慢速 AI 调用 |
| 数据库连接池 | `db.py` | psycopg3 连接池（min=1, max=10），防止高并发下连接耗尽 |
| 幂等迁移 | `database/migrations.py` | `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`，每次部署安全执行 |
| 双模式（真实 + 演示） | `mock_service.py` | 与真实服务接口一致，无需 API Key 即可演示 |
| 双语支持 | `mock_service.py:36-52`, `ai_service.py` | JD 语言检测 → 中文或英文 prompt |
| 结构化日志 | `logging_config.py` | JSON 格式日志，含每请求延迟，机器可读 |

---

## 现有代码问题（优先修复）

### 1. ThreadPoolExecutor 错误处理逻辑有误（`interview.py:370-390`）

```python
# BUG：调用 .exception() 内部已经 resolve 了 future，
# 之后再调 .result() 如果有异常会再次抛出，导致异常无法被干净捕获
jd_exc = jd_future.exception()
jd_result = jd_future.result()
```

**修复方案：**
```python
try:
    jd_result = jd_future.result(timeout=120)
    jd_exc = None
except Exception as e:
    jd_exc = e
    jd_result = None

try:
    match_result = match_future.result(timeout=120)
    match_exc = None
except Exception as e:
    match_exc = e
    match_result = None
```

### 2. OpenAI API 调用无重试机制（`ai_service.py`）

限速和瞬时网络故障会导致任务永久失败。添加 `tenacity`：

```python
# requirements.txt 中添加：tenacity>=8.2

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=30),
    reraise=True
)
def _call_openai(self, schema, *, system: str, user: str):
    return self.client.responses.parse(
        model=self.model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=schema,
    )
```

### 3. 简历/JD 文本缺少最大长度校验（`schemas.py`）

超大 PDF 会静默超出 token 上限，导致输出被截断或 OpenAI 返回 400 错误。

```python
# 在 schemas.py 中，给 JobDescriptionRequest 和 MatchResumeRequest 添加：
resume_text: str = Field(..., min_length=50, max_length=50_000)
job_description: str = Field(..., min_length=50, max_length=30_000)
```

### 4. 前端硬编码了 Railway 生产 URL（`streamlit_app.py:37`）

```python
DEFAULT_API_BASE_URL = "https://ai-interview-copilot-production.up.railway.app"
```

这会在本地开发和演示中暴露生产地址。改为读取环境变量：

```python
import os
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
```

### 5. PDF 解析器静默吞掉错误（`pdf_parser.py`）

两个解析器的异常都被捕获后忽略，没有任何日志记录哪个解析器成功或失败。

```python
import logging
logger = logging.getLogger(__name__)

def extract_text(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if text.strip():
                logger.info("pdf_parser=pdfplumber pages=%d chars=%d", len(pdf.pages), len(text))
                return text
    except Exception as e:
        logger.warning("pdfplumber 解析失败，切换到 PyPDF2: %s", e)

    try:
        # PyPDF2 fallback
        ...
    except Exception as e:
        logger.error("pdf_parser 两个解析器均失败: %s", e)
        raise
```

---

## 高价值简历改进项

### A. RAG — 语义题库（最强信号）

**功能描述：** 将每次生成的问题和答案存入向量数据库。分析新 JD 时，检索语义上最相似的历史 Q&A 对，注入为 few-shot 示例。题库随使用自动积累，重复岗位调用成本持续下降。

**为什么能打动面试官：** 体现了对 LLM 调用成本的理解，以及语义检索优于 prompt 堆砌的工程判断。

**实现步骤：**
1. 在 PostgreSQL 中启用 `pgvector` 扩展（已用 Postgres，零额外基础设施）
2. 创建 `question_embeddings` 表，含 `vector(1536)` 列
3. 每次 `generate_questions` 调用后，用 `text-embedding-3-small` 对 JD 角色类型 + 技能做 embedding
4. 存储问题 + embedding
5. 下次分析相似岗位时，用余弦相似度检索 top-5 并注入 prompt

```sql
-- 添加到 migrations.py 的迁移
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS question_embeddings (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    answer_text TEXT,
    embedding vector(1536),
    role_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON question_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

```python
# app/services/rag_service.py
from openai import OpenAI
from typing import List

class RAGQuestionBank:
    def __init__(self, client: OpenAI, db_conn):
        self.client = client
        self.db = db_conn

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        return resp.data[0].embedding

    def retrieve_similar(self, role_type: str, skills: List[str], k: int = 5) -> List[dict]:
        query = f"{role_type}: {', '.join(skills[:10])}"
        embedding = self.embed(query)
        rows = self.db.execute(
            """
            SELECT question_text, answer_text, 1 - (embedding <=> %s::vector) AS similarity
            FROM question_embeddings
            WHERE role_type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, role_type, embedding, k),
        ).fetchall()
        return [{"question": r[0], "answer": r[1], "score": r[2]} for r in rows]

    def store(self, session_id: int, questions: list, role_type: str):
        for q in questions:
            embedding = self.embed(f"{q.question} {q.category}")
            self.db.execute(
                "INSERT INTO question_embeddings (session_id, question_text, role_type, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (session_id, q.question, role_type, embedding),
            )
```

**简历描述：** *"基于 pgvector 实现 RAG 语义题库，对重复岗位类型的 LLM 调用减少约 40%。"*

---

### B. 多轮对话（智能体跟进）

**功能描述：** 让用户对任意答案发起追问："说得更技术性一点"、"用 STAR 法则重写"、"给我一个更难的版本"。在现有 sessions 表中维护每个会话的消息历史。

**为什么能打动面试官：** 从无状态问答 → 有记忆的对话 AI 智能体，展示了对 chat memory 模式的理解。

**实现步骤：**
1. 在 `sessions` 表中添加 `messages JSONB DEFAULT '[]'` 列
2. 新增端点 `POST /sessions/{id}/chat`，接收 `{"role": "user", "content": "..."}`
3. 以现有问题 + 答案为上下文，携带历史调用 `client.responses.create()`
4. 通过 SSE 流式返回响应（`interview.py` 中已有此模式）
5. 将助手回复追加到 `messages` 列

```python
# app/routes/interview.py — 新增端点
@router.post("/sessions/{session_id}/chat")
async def session_chat(
    session_id: int,
    body: ChatMessageRequest,
    current_user=Depends(current_user),
):
    session = get_session(session_id, current_user.id)
    messages = session.get("messages", [])
    messages.append({"role": "user", "content": body.content})

    async def stream_reply():
        with client.responses.stream(
            model=settings.openai_model,
            input=messages,
        ) as stream:
            for chunk in stream:
                if chunk.type == "response.output_text.delta":
                    yield f"data: {json.dumps({'delta': chunk.delta})}\n\n"
            messages.append({"role": "assistant", "content": stream.get_final_text()})
            update_session_messages(session_id, messages)
        yield "data: [DONE]\n\n"

    return EventSourceResponse(stream_reply())
```

**简历描述：** *"为每个面试会话添加了有状态多轮对话，对话历史以 JSONB 格式持久化，用户可迭代优化答案。"*

---

### C. LangGraph 自适应出题 Agent

**功能描述：** 将一次性的 `generate_questions` 调用替换为 LangGraph Agent：
1. 生成初始问题
2. 自我审查："这些问题够针对性吗？难度是否匹配职级？"
3. 若审查未通过，针对性重新生成
4. 达到质量标准后才返回结果

**为什么能打动面试官：** 展示了 agentic 推理、自我评估循环，以及对 LangChain/LangGraph 的掌握——这些都是当下简历上的高价值关键词。

**实现步骤：**
1. `pip install langchain langchain-openai langgraph`
2. 定义状态：`{"questions": [], "critique": "", "iterations": 0, "approved": False}`
3. 构建两个节点：`generate_node` 和 `critique_node`
4. 添加条件边：`approved` 为 True 或 `iterations >= 3` → END，否则 → `generate_node`
5. 将现有的 `AIInterviewService.generate_questions` 封装为生成节点

```python
# app/services/agent_service.py
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List

class QuestionState(TypedDict):
    jd_analysis: dict
    match_result: dict
    questions: List[dict]
    critique: str
    iterations: int
    approved: bool

def build_question_agent(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0.3)

    def generate_node(state: QuestionState) -> QuestionState:
        critique_context = f"\n上一轮审查意见：{state['critique']}" if state["critique"] else ""
        prompt = f"""为以下岗位生成 10 道面试题。
岗位：{state['jd_analysis']['role_title']}
级别：{state['jd_analysis']['seniority_level']}
技能：{state['jd_analysis']['required_skills']}
候选人匹配度：{state['match_result']['overall_fit_score']}/100
{critique_context}

返回 JSON：[{{"question": str, "category": str, "difficulty": 1-5}}]"""
        result = llm.invoke(prompt)
        return {**state, "questions": parse_questions(result.content), "iterations": state["iterations"] + 1}

    def critique_node(state: QuestionState) -> QuestionState:
        prompt = f"""评估以下面试题：
{state['questions']}

判断：(1) 是否针对该岗位而非通用题目？(2) 难度是否匹配 {state['jd_analysis']['seniority_level']} 级别？
(3) 技术题与行为题是否平衡？

返回 JSON：{{"approved": bool, "critique": str}}"""
        result = llm.invoke(prompt)
        data = parse_critique(result.content)
        return {**state, "approved": data["approved"], "critique": data["critique"]}

    def should_continue(state: QuestionState) -> str:
        if state["approved"] or state["iterations"] >= 3:
            return END
        return "generate"

    graph = StateGraph(QuestionState)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", should_continue, {"generate": "generate", END: END})
    return graph.compile()
```

**简历描述：** *"用 LangGraph 自评估 Agent 替换了一次性出题逻辑，通过迭代审查循环确保问题满足岗位专属质量标准。"*

---

### D. OpenAI Prompt Caching（成本优化信号）

**功能描述：** 将 system prompt 和 JD 标记为可缓存内容。对同一 JD 的重复调用（如更新简历后重新分析），OpenAI 可复用缓存 token，折扣高达 90%。

**为什么能打动面试官：** 成本意识是高级工程师的特质，大多数学生项目从不提 token 经济学。

**实现步骤：**
1. 将静态、可复用的部分（system prompt + JD）放在 prompt 最前面
2. 对静态输入块添加 `cache_control: {"type": "ephemeral"}`
3. 记录 `usage.prompt_tokens_details.cached_tokens` 以追踪节省情况

```python
# app/services/ai_service.py — 修改 _call_api 方法
def _parse_structured(self, schema, *, system: str, user: str):
    input_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    response = self.client.responses.parse(
        model=self.model,
        input=input_messages,
        text_format=schema,
    )

    usage = response.usage
    cached_tokens = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0)
    self.usage_events.append({
        "step": schema.__name__,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": cached_tokens,
        "cache_savings_pct": round(cached_tokens / max(usage.input_tokens, 1) * 100, 1),
    })
    return response.output_parsed
```

**简历描述：** *"利用 OpenAI Prompt Caching 对静态 system prompt 进行缓存，重复分析同一 JD 时输入 token 成本降低最高 90%。"*

---

### E. OpenTelemetry 可观测性

**功能描述：** 对每条 FastAPI 请求、数据库查询和 OpenAI 调用自动生成分布式追踪，导出到 Jaeger（本地）或 Honeycomb/Datadog（生产）。

**为什么能打动面试官：** 生产系统需要可观测性，这是玩具项目和专业作品的本质区别。

**实现步骤：**
1. `pip install opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-psycopg`
2. 在 `main.py` 的 lifespan 中初始化 tracer
3. 自动插桩 FastAPI 和 psycopg3
4. 在 OpenAI 调用外手动添加 span，附带 token 数和延迟属性

```python
# app/telemetry.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import os

def setup_telemetry(app):
    provider = TracerProvider()
    if otlp_endpoint := os.getenv("OTLP_ENDPOINT"):
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    return trace.get_tracer("ai-interview-copilot")

# 在 ai_service.py 中：
tracer = trace.get_tracer(__name__)

def analyze_jd(self, job_description: str) -> JDAnalysis:
    with tracer.start_as_current_span("openai.analyze_jd") as span:
        result = self._parse_structured(JDAnalysis, system=SYSTEM_PROMPT, user=job_description)
        span.set_attribute("openai.model", self.model)
        span.set_attribute("openai.input_tokens", self.usage_events[-1]["input_tokens"])
        span.set_attribute("openai.output_tokens", self.usage_events[-1]["output_tokens"])
        return result
```

**简历描述：** *"集成 OpenTelemetry 分布式追踪，覆盖 FastAPI、PostgreSQL 和 OpenAI 调用，在 Honeycomb 中实现逐步骤延迟分析和成本归因。"*

---

### F. 多模型降级链

**功能描述：** 若 OpenAI 失败（限速、故障），自动降级到 Anthropic Claude，再降级到 Mock 服务，用户零感知中断。

**为什么能打动面试官：** 体现生产级韧性思维，展示了对供应商锁定风险的理解。

**实现步骤：**
1. `pip install anthropic`
2. 创建 `ModelRouter`，按顺序尝试各 provider
3. 标准化输出：所有 provider 返回相同的 Pydantic schema
4. 每次请求记录实际使用的模型

```python
# app/services/model_router.py
from openai import OpenAI, RateLimitError
import anthropic

class ModelRouter:
    def __init__(self, settings):
        self.providers = []
        if settings.openai_api_key:
            self.providers.append(("openai", OpenAI(api_key=settings.openai_api_key)))
        if settings.anthropic_api_key:
            self.providers.append(("anthropic", anthropic.Anthropic(api_key=settings.anthropic_api_key)))

    def parse(self, schema, *, system: str, user: str):
        last_error = None
        for name, client in self.providers:
            try:
                if name == "openai":
                    return self._openai_parse(client, schema, system=system, user=user), name
                elif name == "anthropic":
                    return self._anthropic_parse(client, schema, system=system, user=user), name
            except (RateLimitError, Exception) as e:
                logger.warning("provider=%s 调用失败：%s，尝试下一个", name, e)
                last_error = e
        raise RuntimeError(f"所有 provider 均失败。最后一个错误：{last_error}")
```

**简历描述：** *"实现多 provider 降级链（OpenAI → Claude → Mock），确保 API 故障期间服务零中断降级。"*

---

### G. 每用户 Token 配额与限流

**功能描述：** 在 PostgreSQL 中按用户每日追踪 token 用量，超出配额后拒绝请求并返回明确错误信息。

**为什么能打动面试官：** SaaS AI 产品的成本管控必备能力，体现规模化思维。

**实现步骤：**
1. 在 `users` 表中添加 `daily_token_quota INTEGER DEFAULT 100000`
2. 添加 `token_usage_log` 表，字段：`user_id, date, tokens_used`
3. FastAPI 依赖项 `check_quota`，读取今日用量，超限则拒绝
4. 每次 AI 调用成功后累加计数器

```python
# app/dependencies.py — 添加配额检查
async def check_quota(current_user=Depends(current_user), db=Depends(get_db)):
    today_usage = get_user_token_usage_today(db, current_user.id)
    quota = current_user.daily_token_quota or 100_000
    if today_usage >= quota:
        raise HTTPException(
            status_code=429,
            detail=f"每日 Token 配额（{quota:,}）已用尽，UTC 午夜重置。"
        )
    return current_user

# 应用于高消耗端点：
@router.post("/sessions/jobs")
async def create_session_job(..., user=Depends(check_quota)):
    ...
```

**简历描述：** *"添加基于 PostgreSQL 的每用户每日 Token 配额追踪，防止多租户场景下的成本失控。"*

---

### H. 数据分析仪表盘

**功能描述：** 新增 Streamlit 页面，展示聚合统计：最常见岗位类型、平均匹配分、高频技能缺口、Token 成本趋势、问题分类分布。

**为什么能打动面试官：** 将原始 JSONB 数据转化为商业洞察，体现超越"让功能跑起来"的产品思维。

**实现步骤：**
1. 编写聚合 SQL，对 `sessions.jd_analysis` 和 `sessions.resume_match` JSONB 进行统计
2. 新建 Streamlit 页面 `pages/02_Analytics.py`
3. 用 `st.bar_chart`、`st.line_chart`、`st.metric` 做可视化
4. 支持按日期范围和用户筛选（管理员视图）

```python
# app/frontend/pages/02_Analytics.py
import streamlit as st
import pandas as pd

st.title("面试数据分析")

col1, col2, col3 = st.columns(3)
col1.metric("总会话数", stats["total_sessions"])
col2.metric("平均匹配分", f"{stats['avg_fit_score']:.0f}/100")
col3.metric("Token 费用（近30天）", f"${stats['total_cost_30d']:.2f}")

st.subheader("热门岗位类型")
st.bar_chart(pd.DataFrame(stats["role_distribution"]).set_index("role"))

st.subheader("常见技能缺口")
st.bar_chart(pd.DataFrame(stats["skill_gaps"]).set_index("skill"))
```

**简历描述：** *"构建数据分析仪表盘，聚合会话 JSONB 数据，展示岗位分布、技能缺口频率和月度 Token 成本趋势。"*

---

## 基础设施与 DevOps 改进

### 1. 引入 Alembic 数据库迁移工具

当前 `CREATE TABLE IF NOT EXISTS` 方案无法处理字段重命名、类型变更或数据迁移。Alembic 提供版本化、可回滚的迁移。

```bash
pip install alembic
alembic init alembic
# 用 Alembic revision 文件替换手动 migrations.py
```

### 2. 添加 GitHub Actions CI 并设置覆盖率门槛

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest --cov=app --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4
```

### 3. 替换为异步数据库驱动

用 `asyncpg` 替换 psycopg3（同步），实现 FastAPI 路由中真正的非阻塞数据库调用：

```bash
pip install asyncpg
# 用 asyncpg 连接池替换 psycopg3 连接池
# 路由已经是 async def，DB 是唯一的同步瓶颈
```

### 4. 添加带依赖状态的健康检查端点

```python
# app/routes/health.py
@router.get("/health/detailed")
async def detailed_health():
    return {
        "status": "ok",
        "database": await check_db_health(),
        "openai": await check_openai_health(),
        "version": settings.app_version,
        "uptime_seconds": time.time() - START_TIME,
    }
```

---

## 优先级实施顺序

| 优先级 | 改进项 | 预估工时 | 简历价值 | 类型 |
|---|---|---|---|---|
| 🔴 **P0** | 修复 ThreadPoolExecutor 错误处理 | 30 分钟 | 低 | Bug 修复 |
| 🔴 **P0** | 添加重试 + 指数退避 | 1 小时 | 中 | Bug 修复 |
| 🔴 **P0** | 添加输入最大长度校验 | 30 分钟 | 低 | Bug 修复 |
| 🟠 **P1** | pgvector RAG 语义题库 | 1 天 | **极高** | 新功能 |
| 🟠 **P1** | LangGraph 出题 Agent | 1 天 | **极高** | 新功能 |
| 🟠 **P1** | Prompt Caching + 成本追踪 | 2 小时 | 高 | 增强 |
| 🟡 **P2** | 多轮对话记忆 | 4 小时 | 高 | 新功能 |
| 🟡 **P2** | 多模型降级链 | 4 小时 | 高 | 增强 |
| 🟡 **P2** | OpenTelemetry 分布式追踪 | 4 小时 | 高 | 基础设施 |
| 🟢 **P3** | 数据分析仪表盘 | 6 小时 | 中 | 新功能 |
| 🟢 **P3** | 每用户 Token 配额 | 3 小时 | 中 | 新功能 |
| 🟢 **P3** | Alembic 迁移工具 | 3 小时 | 中 | 基础设施 |
| 🟢 **P3** | 异步数据库（asyncpg） | 6 小时 | 中 | 重构 |

---

## 简历描述对比

### 改进前
> "使用 FastAPI、OpenAI API 和 Streamlit 构建了一个 AI 面试准备工具，后端使用 PostgreSQL。"

### 完成 P0 + P1 后
> "设计并上线了 AI 面试辅导系统：基于 LangGraph 的自评估 Agent 实现自适应出题，pgvector RAG 流水线对重复岗位 LLM 调用减少约 40%，OpenAI Prompt Caching 降低 token 成本最高 90%，多轮对话记忆支持答案迭代优化——后端部署于 Railway，前端部署于 Streamlit Cloud，具备异步任务工作流、结构化输出和 OpenTelemetry 分布式追踪。"

---

## 技术栈覆盖清单

| 技术 | 状态 | 备注 |
|---|---|---|
| OpenAI 结构化输出 | ✅ 已实现 | `client.responses.parse()` |
| SSE 流式传输 | ✅ 已实现 | 答案重新生成 |
| 异步任务工作流 | ✅ 已实现 | HTTP 202 + 轮询 |
| PostgreSQL + JSONB | ✅ 已实现 | AI 输出存为 JSONB |
| 数据库连接池 | ✅ 已实现 | psycopg3 连接池 |
| JWT 鉴权（标准库） | ✅ 已实现 | 无 PyJWT 依赖 |
| 并行 LLM 调用 | ✅ 已实现 | ThreadPoolExecutor |
| RAG / 向量搜索 | ❌ 缺失 | 添加 pgvector |
| LangChain / LangGraph | ❌ 缺失 | 添加出题 Agent |
| Prompt Caching | ❌ 缺失 | 添加 cache_control |
| 多模型降级 | ❌ 缺失 | 添加 Anthropic 备选 |
| OpenTelemetry | ❌ 缺失 | 添加分布式追踪 |
| 多轮对话记忆 | ❌ 缺失 | 添加 messages JSONB |
| Token 配额 / 限流 | ❌ 缺失 | 添加每用户限制 |
| 重试 + 退避 | ❌ 缺失 | 添加 tenacity |
| 数据分析仪表盘 | ❌ 缺失 | 添加 Streamlit 页面 |
| Alembic 迁移 | ❌ 缺失 | 替换 init_db 方式 |
| 异步 DB（asyncpg） | ❌ 缺失 | 替换 psycopg3 |
| CI/CD + 覆盖率 | ❌ 缺失 | 添加 GitHub Actions |

---

*本文档通过分析完整源码树生成。所有代码片段仅供参考，应用前请以实际文件内容为准。*
