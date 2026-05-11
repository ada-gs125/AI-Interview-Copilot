from __future__ import annotations

from typing import Iterator

from app.schemas import (
    AnswerResult,
    AnswerSet,
    GenerateAnswerRequest,
    GenerateQuestionsRequest,
    InterviewQuestion,
    JDAnalysis,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    ResumeMatchRequest,
    RoleType,
    SkillItem,
)


def _use_chinese(output_language: OutputLanguage, job_description: str = "") -> bool:
    if output_language == "Chinese":
        return True
    if output_language == "English":
        return False
    return any("\u4e00" <= char <= "\u9fff" for char in job_description)


class MockAIInterviewService:
    def analyze_jd(
        self,
        job_description: str,
        role_type: RoleType,
        output_language: OutputLanguage = "Match job description language",
    ) -> JDAnalysis:
        if _use_chinese(output_language, job_description):
            return JDAnalysis(
                role_summary=f"这是一个偏 {role_type} 的岗位，重点关注工程基础、项目落地和 AI 应用能力。",
                required_technical_skills=[
                    SkillItem(name="Python", importance="must_have", evidence="Demo mode sample skill."),
                    SkillItem(name="API design", importance="must_have", evidence="Demo mode sample skill."),
                    SkillItem(name="LLM applications", importance="preferred", evidence="Demo mode sample skill."),
                ],
                preferred_skills=[
                    SkillItem(name="FastAPI", importance="preferred", evidence="Demo mode sample skill."),
                    SkillItem(name="SQL", importance="preferred", evidence="Demo mode sample skill."),
                ],
                likely_interview_topics=["项目架构", "API 设计", "数据库基础", "LLM 应用流程"],
                backend_system_design_topics=["任务队列", "缓存", "服务拆分", "数据持久化"],
                ai_llm_topics=["结构化输出", "提示词设计", "幻觉控制", "成本优化"],
                seniority_signals=["能端到端交付功能", "能解释技术取舍"],
                red_flags_or_ambiguities=["Demo mode 使用示例结果，不代表真实 JD 分析。"],
            )

        return JDAnalysis(
            role_summary=f"A {role_type} role focused on backend execution, applied AI workflows, and project ownership.",
            required_technical_skills=[
                SkillItem(name="Python", importance="must_have", evidence="Demo mode sample skill."),
                SkillItem(name="API design", importance="must_have", evidence="Demo mode sample skill."),
                SkillItem(name="LLM applications", importance="preferred", evidence="Demo mode sample skill."),
            ],
            preferred_skills=[
                SkillItem(name="FastAPI", importance="preferred", evidence="Demo mode sample skill."),
                SkillItem(name="SQL", importance="preferred", evidence="Demo mode sample skill."),
            ],
            likely_interview_topics=["Project architecture", "API design", "Database fundamentals", "LLM workflows"],
            backend_system_design_topics=["Queues", "Caching", "Service boundaries", "Persistence"],
            ai_llm_topics=["Structured outputs", "Prompt design", "Hallucination control", "Cost optimization"],
            seniority_signals=["Can ship end-to-end features", "Can explain tradeoffs"],
            red_flags_or_ambiguities=["Demo mode returns sample results, not a real JD analysis."],
        )

    def match_resume(self, request: ResumeMatchRequest) -> ResumeMatch:
        if _use_chinese(request.output_language, request.job_description):
            return ResumeMatch(
                overall_fit_score=72,
                strong_matches=[
                    {
                        "skill": "Python / backend foundations",
                        "resume_evidence": "Demo mode assumes the resume includes relevant backend or project experience.",
                        "strength": "strong",
                    },
                    {
                        "skill": "AI workflow thinking",
                        "resume_evidence": "Demo mode highlights this project as an AI workflow example.",
                        "strength": "partial",
                    },
                ],
                missing_skills=["真实模式会根据简历和 JD 提取具体缺口。"],
                transferable_strengths=["项目交付能力", "学习能力", "结构化表达"],
                suggested_positioning_strategy="把经历包装成端到端 AI 工具交付：从文档解析、LLM 编排、结构化输出到前端展示。",
                recommended_project_talking_points=["为什么使用 FastAPI + Streamlit", "如何控制 LLM 幻觉", "如何降低 API 调用成本"],
                resume_risks_to_prepare_for=["准备好解释 demo mode 与真实 OpenAI 调用模式的区别。"],
            )

        return ResumeMatch(
            overall_fit_score=72,
            strong_matches=[
                {
                    "skill": "Python / backend foundations",
                    "resume_evidence": "Demo mode assumes the resume includes relevant backend or project experience.",
                    "strength": "strong",
                },
                {
                    "skill": "AI workflow thinking",
                    "resume_evidence": "Demo mode highlights this project as an AI workflow example.",
                    "strength": "partial",
                },
            ],
            missing_skills=["Real mode will extract exact gaps from the resume and JD."],
            transferable_strengths=["Project delivery", "Learning ability", "Structured communication"],
            suggested_positioning_strategy="Frame the work as an end-to-end AI tool: document parsing, LLM orchestration, structured output, and a usable frontend.",
            recommended_project_talking_points=["Why FastAPI + Streamlit", "How hallucinations are controlled", "How API cost is reduced"],
            resume_risks_to_prepare_for=["Be ready to explain the difference between demo mode and real OpenAI calls."],
        )

    def generate_questions(self, request: GenerateQuestionsRequest) -> QuestionSet:
        chinese = _use_chinese(request.output_language, request.job_description)
        if chinese:
            return QuestionSet(
                technical_questions=[
                    InterviewQuestion(question="你如何设计这个项目的后端 API？", why_it_matters="考察 REST 设计和模块边界。", difficulty="medium"),
                    InterviewQuestion(question="你如何保证 LLM 输出可以被系统稳定解析？", why_it_matters="考察结构化输出和错误处理。", difficulty="hard"),
                ],
                project_deep_dive_questions=[
                    InterviewQuestion(question="这个项目里最重要的工程取舍是什么？", why_it_matters="考察项目理解深度。", difficulty="medium"),
                ],
                system_design_questions=[
                    InterviewQuestion(question="如果要支持多用户和历史记录，你会如何设计数据库？", why_it_matters="考察扩展性。", difficulty="medium"),
                ],
                behavioral_questions=[
                    InterviewQuestion(question="讲一次你快速学习并交付新技术项目的经历。", why_it_matters="考察学习能力和表达。", difficulty="warmup"),
                ],
            )

        return QuestionSet(
            technical_questions=[
                InterviewQuestion(question="How would you design the backend APIs for this project?", why_it_matters="Tests REST design and service boundaries.", difficulty="medium"),
                InterviewQuestion(question="How do you make LLM output reliable enough for application code?", why_it_matters="Tests structured outputs and error handling.", difficulty="hard"),
            ],
            project_deep_dive_questions=[
                InterviewQuestion(question="What was the most important engineering tradeoff in this project?", why_it_matters="Tests project depth.", difficulty="medium"),
            ],
            system_design_questions=[
                InterviewQuestion(question="How would you redesign persistence for multiple users and saved histories?", why_it_matters="Tests scalability thinking.", difficulty="medium"),
            ],
            behavioral_questions=[
                InterviewQuestion(question="Tell me about a time you learned a new technology quickly and shipped with it.", why_it_matters="Tests learning speed and communication.", difficulty="warmup"),
            ],
        )

    def generate_answer(self, request: GenerateAnswerRequest) -> AnswerResult:
        chinese = _use_chinese(request.output_language, request.job_description or request.question)
        if chinese:
            return AnswerResult(
                category=request.category,
                question=request.question,
                concise_answer="我会先说明项目目标，再讲清楚输入、处理流程和输出。这个项目展示了我把 PDF 解析、后端 API、LLM 结构化输出和前端展示串起来的能力。",
                resume_evidence_used=["Demo mode sample evidence."],
                honesty_guardrail="Demo mode 不使用真实简历证据，正式模式会只基于上传简历生成答案。",
            )
        return AnswerResult(
            category=request.category,
            question=request.question,
            concise_answer="I would explain the product goal first, then walk through the input, processing workflow, and output. The project shows that I can connect PDF parsing, backend APIs, structured LLM output, and a usable frontend.",
            resume_evidence_used=["Demo mode sample evidence."],
            honesty_guardrail="Demo mode does not use real resume evidence; real mode grounds answers in the uploaded resume.",
        )

    def stream_answer(self, request: GenerateAnswerRequest) -> Iterator[str]:
        text = self.generate_answer(request).concise_answer
        chunk_size = 25
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    def generate_answers_for_question_set(
        self,
        *,
        resume_text: str,
        job_description: str = "",
        role_type: RoleType,
        output_language: OutputLanguage,
        questions: QuestionSet,
    ) -> AnswerSet:
        if _use_chinese(output_language, job_description):
            categories = (
                ("技术问题", questions.technical_questions),
                ("项目深挖问题", questions.project_deep_dive_questions),
                ("系统设计问题", questions.system_design_questions),
                ("行为面试问题", questions.behavioral_questions),
            )
        else:
            categories = (
                ("Technical", questions.technical_questions),
                ("Project Deep-Dive", questions.project_deep_dive_questions),
                ("System Design", questions.system_design_questions),
                ("Behavioral", questions.behavioral_questions),
            )
        answers: list[AnswerResult] = []
        for category, items in categories:
            for item in items:
                answers.append(
                    self.generate_answer(
                        GenerateAnswerRequest(
                            resume_text=resume_text,
                            job_description=job_description or None,
                            role_type=role_type,
                            output_language=output_language,
                            category=category,
                            question=item.question,
                        )
                    )
                )
        return AnswerSet(answers=answers)
