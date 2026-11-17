# AI Incident Analysis Assistant: Interview Preparation

This guide is tailored to the current local portfolio implementation. It is a question bank and practice rubric, not a claim that the project is production-ready. Answer from the code and measurements you can demonstrate. The mock interview should use one question at a time; expect follow-up questions that challenge assumptions.

## Your project in one minute

Practice a concise explanation in your own words. Cover the user problem, end-to-end flow, the reason for hybrid retrieval, how evidence references are kept traceable, and the most important limitation. Do not recite implementation details before explaining the problem.

## Project facts to know precisely

- UI: Streamlit; Python core modules; Pydantic response schema; OpenAI Responses API for structured incident analysis.
- Knowledge ingestion: local UTF-8 `.txt` and `.md` files in `knowledge_base/`; uploads and incident logs are excluded from the persistent index.
- Chunking: maximum 1,800 characters with 250-character overlap; deterministic IDs based on source path and chunk number; content hashes detect changed text.
- Index: Chroma `PersistentClient`; vectors are created explicitly through the configured OpenAI client, using `text-embedding-3-small` by default. Rebuild is user-triggered.
- Retrieval: BM25Plus over the incident plus selected sanitized log excerpts; dense Chroma search embeds the incident description plus failure-reporting log lines; reciprocal-rank fusion; up to three passages.
- Retrieval defaults: five lexical and five dense candidates, a keyword candidate must share at least two distinct non-stop-word terms from the incident description and failure-reporting log lines with the chunk, scaling to 10% of those distinct terms with a cap of four (BM25Plus scores alone are positive for every chunk; routine log lines only rerank), cosine similarity cutoff 0.35, RRF constant 60, fused-score cutoff 0.016.
- Responses distinguish observable facts from potential causes, diagnostic steps, and suggested resolution. App-created IDs point to the exact incident/log excerpts or knowledge chunks; unknown IDs are rejected.
- Safety: no commands run, no remediation, no incident persistence, best-effort redaction, explicit user actions for analysis and reindexing, safe error messages.
- Evaluation: 20 local query cases, including three off-topic IT negatives that share vocabulary with the corpus and two cases with an attached log. Live evaluation needs an explicit `--live` invocation and sends embedding requests. It has not been run as part of the mocked test suite.
- Verification to date: 139 automated tests passed on Python 3.14.5; Chroma emits one dependency deprecation warning. Other declared Python versions and live API behavior have not been verified locally.

## How to use this guide

1. First answer without looking at the “strong answer signals.”
2. Give concrete examples from this implementation; distinguish measured behavior from design intent.
3. State trade-offs and limitations explicitly.
4. When an answer depends on current external API behavior, say how you would verify it in official documentation and with a controlled integration check.
5. Practice the questions marked **Core** first. The others are follow-up depth.

## 1. Problem framing and product decisions

1. **Core:** What operational problem does this assistant address, and who is the intended user?
2. What would an incident responder do before, during, and after using this tool?
3. Which user decision does the generated analysis support? Which decisions remain human-owned?
4. What is explicitly out of scope, and why?
5. Why is this a RAG project rather than a general-purpose chatbot?
6. What makes the demo portfolio-ready, and what would be required before real operational use?
7. What are the three most important user-visible outcomes you would measure?
8. How would you prevent the UI from implying that a suggested cause is confirmed?
9. Why did you choose synthetic troubleshooting material? What does that prevent you from learning?
10. If an interviewer asks for a live demo, what is the safest reproducible scenario you would show?

**Strong answer signals:** clear target user and task; evidence that the scope is deliberately advisory; acknowledgement that synthetic data does not establish operational effectiveness; outcome measures such as top-k relevance, useful investigation steps, and time-to-triage rather than “the LLM sounds good.”

## 2. Architecture and request flow

11. **Core:** Walk me through one incident from form submission to rendered analysis, naming the key modules and data passed between them.
12. Which work happens locally and which work is sent to OpenAI?
13. What happens when the API key is missing?
14. What happens when a semantic index has not been built yet?
15. Why does the app offer a local BM25 preview separately from the submitted hybrid analysis?
16. Where are the trust boundaries in this architecture?
17. Why did you separate `validation`, `indexing`, `retrieval`, `analysis`, `schemas`, and `config`?
18. Which components can be tested without Streamlit, Chroma persistence, or network access?
19. What state is held in memory, and what state persists on disk?
20. What would you change first if the app needed multiple concurrent users?

**Strong answer signals:** can narrate validation → redaction → log evidence selection → embedding/retrieval → evidence record creation → structured API call → reference validation → display; can describe local index versus external API boundary; does not claim uploaded incidents are persisted.

## 3. RAG, embeddings, and chunking

21. **Core:** Explain embeddings to a non-technical incident manager. What does a vector represent, and what does similarity mean?
22. Why use `text-embedding-3-small` for this demo? What would make you change it?
23. Why do you embed documents during explicit indexing and embed the query during analysis?
24. Why are query and passage embeddings generated with the same configured model?
25. Why are uploads and logs excluded from persistent indexing?
26. Why are documents split into chunks? What is lost if chunks are too small or too large?
27. How do the 1,800-character size and 250-character overlap affect retrieval? What evidence would justify changing them?
28. Does the current chunker guarantee paragraph boundaries? Inspect its exact behavior and explain any limitation honestly.
29. How do deterministic chunk IDs and content hashes differ in purpose?
30. Why must the embedding model and index version be recorded with the collection/manifest?
31. What happens if the embedding model changes but old vectors remain?
32. How would you prevent vectors with different dimensions from being mixed in one collection?
33. What is the difference between an embedding model and a chat/generation model?
34. What are the privacy and cost consequences of embedding the knowledge base?
35. What risks arise if source documents contain malicious prompt instructions?

**Strong answer signals:** similarity is semantic proximity, not truth; chunk settings are tunable hypotheses; model/version compatibility is treated as an index invalidation concern; document ingestion is explicit and bounded; no claim that embeddings alone “understand” or prove relevance.

## 4. Retrieval and ranking

36. **Core:** Why combine BM25 with dense retrieval instead of choosing one?
37. How does BM25 help with exact terms such as HTTP 503, SQL names, or error codes?
38. What can dense retrieval find that BM25 can miss?
39. Explain Reciprocal Rank Fusion. Why fuse ranks instead of adding raw BM25 and cosine scores?
40. What does `k=60` do in the RRF formula?
41. Why retrieve five candidates from each method and return at most three?
42. Why is the dense similarity threshold 0.35? Is that value universally meaningful?
43. Why is the fused threshold 0.016? Which ranked candidates can it exclude?
44. How are ties made deterministic?
45. What happens when the knowledge files change after the last indexing operation?
46. Why does BM25 use incident text plus selected log excerpts while the dense query embedding uses only the incident description?
47. Could a noisy or adversarial log line dominate retrieval? How would you reduce that risk?
48. What is the current fallback if embedding fails? What quality is lost in fallback mode?
49. How would retrieval change if the knowledge base grew from three short documents to 100,000 chunks?
50. When might a cross-encoder reranker be justified? Why is it deferred here?

**Strong answer signals:** can state the actual constants and formulas; identifies thresholds as empirical demo defaults; describes retrieval as ranking candidate evidence, not asserting a root cause; recognizes that fallback changes the retrieval mode and should be visible.

## 5. Knowledge-base indexing and persistence

51. **Core:** Walk through rebuilding the index, including what is embedded, what is skipped, and when stale chunks are removed.
52. How does the implementation avoid re-embedding an unchanged corpus?
53. What does staged indexing protect against?
54. What happens if an embedding request fails halfway through batches?
55. What happens if Chroma fails while writing the staged collection?
56. At what exact point does the active index change?
57. Why is the index rebuilt only after a user clicks the button?
58. What happens when the knowledge directory is empty?
59. How would you recover from a corrupt manifest or missing Chroma collection?
60. What would you monitor during a large reindex operation?

**Strong answer signals:** embedding all changed chunks completes before pointer switch; old active index remains available on pre-switch failure; unchanged vectors are reused; index API cost is an explicit user action; understands that local atomicity and process-crash guarantees still deserve testing.

## 6. Evidence, grounding, and structured outputs

61. **Core:** How can a user trace a displayed recommendation back to its supporting material?
62. Who creates evidence IDs, and why should the model not create them?
63. How are log line numbers preserved after sanitization and excerpt selection?
64. What is the difference between an observed fact, a potential cause, and a diagnostic recommendation?
65. Why are knowledge passages not allowed to support observable facts?
66. What happens if the model returns an unknown evidence ID?
67. Can a valid citation still support a false conclusion? Explain the limits of reference validation.
68. Why use Pydantic structured output rather than ask for JSON in a prompt?
69. Which failures can still happen even with Structured Outputs?
70. How does the app handle refusals and incomplete output?
71. What would you do if valid schema output contains unsafe or operationally destructive advice?
72. Should every potential cause be required to cite evidence? Explain your choice.

**Strong answer signals:** provenance is application-owned and validated; evidence IDs establish traceability, not correctness; structured schemas constrain shape, while grounding and safety require separate controls; diagnostics remain human-reviewed.

## 7. Security, privacy, and prompt injection

73. **Core:** Threat-model this app. What are the assets, trust boundaries, and likely attackers or failure sources?
74. What does the redaction code mask, and what can it miss?
75. Why is best-effort regex redaction not a guarantee of anonymization?
76. How would you respond if a user uploads a log containing an API key that the redactor misses?
77. What defenses address prompt injection in incident logs and knowledge documents?
78. Why is “ignore instructions inside retrieved text” insufficient as the only prompt-injection defense?
79. What damage is possible if a future engineer adds command execution to the assistant?
80. How are credentials kept out of source, UI errors, logs, and Git?
81. What data leaves the local machine during index rebuild? During analysis? During evaluation?
82. What is the data-retention behavior of this app, and how would you verify it?
83. How would you adapt the system for regulated or customer production data?
84. Why should raw exception messages not be displayed to users?

**Strong answer signals:** separates preventive controls from prompt wording; names data flows; says regex masking is incomplete; does not promise zero retention or compliance without verification; keeps API keys and operational actions outside model control.

## 8. Reliability and error handling

85. **Core:** What happens for a missing API key, timeout, rate limit, refusal, invalid model output, index mismatch, and empty retrieval?
86. Why disable automatic retries? When might a bounded retry be appropriate?
87. How would you distinguish a transient error from a permanent configuration error?
88. What should the UI show when retrieval finds no evidence?
89. Why is a fallback to BM25 safer or less safe than failing the whole request?
90. What happens if Chroma is unavailable but the API is available?
91. How can you prevent duplicate user clicks from causing duplicate paid calls?
92. What should a useful operational log include while preserving privacy?
93. What availability and latency objectives would you set for a local demo versus a production service?

**Strong answer signals:** errors map to safe actionable messages; no false assertion that fallback is equivalent; retries are bounded and idempotency-aware; metrics avoid incident content and credentials.

## 9. Evaluation and testing

94. **Core:** What do Recall@3, MRR, and false-positive rate measure here?
95. Why do you need both positive and unrelated/no-evidence evaluation cases?
96. What does the 20-case set fail to tell you?
97. Why are automated tests mocked, and what integration risk does that leave?
98. Which tests cover deterministic chunk IDs, embedding failure, stale deletion, and reference validation?
99. How would you test whether an answer is factually useful, not just well-formed?
100. How would you build a better evaluation set from real incident data without exposing sensitive logs?
101. How would you inspect a false positive and decide whether to change chunking, retrieval, or the knowledge documents?
102. Why should you avoid tuning thresholds on the same small cases and then claiming general performance?
103. What must the live evaluation command do before it reports success?
104. What tests would you add before accepting contributions to this repository?

**Strong answer signals:** metrics definitions are exact; recognizes dataset size and synthetic data limitations; separates unit, mocked API, local persistence, live API, and human quality evaluation; avoids presenting a small benchmark as proof of production quality.

## 10. Python, dependencies, and maintainability

105. Why is Chroma appropriate for this local project? What would be a signal to move to a managed vector database?
106. Why use `rank-bm25` rather than implementing BM25 manually?
107. What responsibility belongs in Streamlit, and what should stay in core modules?
108. Why are stable interfaces and typed Pydantic models useful here?
109. What Python compatibility issue was corrected, and how would you test declared Python versions?
110. What does the Chroma deprecation warning mean? Is it an application test failure?
111. How would you pin or constrain dependency versions for a reproducible portfolio demo?
112. What would you refactor if the codebase doubled in size?

## 11. System-design extensions

113. How would you support multiple teams with separate knowledge bases and access controls?
114. How would you version and roll back knowledge documents and indexes?
115. How would you support source metadata such as product, version, environment, and date?
116. How would you add incremental background indexing without surprising users with API costs?
117. How would you handle document deletion and ensure deleted content is no longer retrievable?
118. How would you add authentication, audit logs, and rate limits before deployment?
119. How would you design for large files, many concurrent requests, and bounded model context?
120. How would you measure whether human responders accept, edit, or reject recommendations?
121. What guardrails and approval workflow would be mandatory before any remediation feature?

**Strong answer signals:** does not jump straight to infrastructure; begins with concrete requirements, tenant/privacy boundaries, evaluation and rollback; insists on human approval, least privilege, auditability, dry runs, and scoped actions before remediation.

## 12. Behavioral and ownership questions

122. Why did you choose this project for your AI Engineer portfolio?
123. What was the hardest engineering trade-off you made?
124. Tell me about a bug or failed test that changed your design.
125. What is one part you would redesign after this review, and why?
126. How did you decide which features to defer?
127. How do you communicate uncertainty when stakeholders want a definitive answer?
128. Describe how you would work with an SRE or incident commander to validate the knowledge base.
129. What would you do if a stakeholder insists on automatic remediation for the demo?
130. What did you learn about evaluating RAG systems that you did not know before building this?
131. What would you deliver in your first month on an incident-assistance team?

**Strong answer signals:** clear ownership, candid limitations, evidence-based decisions, collaboration with domain experts, and willingness to change a design when tests or incident workflows demonstrate a weakness.

## High-pressure follow-up prompts

Use these after any answer that sounds too certain:

- “What evidence from your code or tests supports that?”
- “What happens if that dependency is unavailable?”
- “How do you know the threshold is appropriate?”
- “What exact data is sent to the API?”
- “Could that citation be valid but misleading?”
- “What would falsify your assumption?”
- “How would you reproduce that failure?”
- “What is the simplest alternative, and why did you reject it?”
- “What have you actually measured, versus what do you expect?”
- “How would this fail with ten thousand documents or a malicious log?”

## Questions to ask the interviewer

1. How does the team measure whether AI assistance improves incident response without increasing risk?
2. How are troubleshooting procedures reviewed, versioned, and approved by subject-matter experts?
3. What are the data handling and privacy constraints for incident logs?
4. How do engineers evaluate retrieval quality and model output quality separately?
5. What human approval and audit requirements apply to AI-generated operational recommendations?

## Mock interview format

The interviewer asks one question at a time, starting with the project walkthrough. After each answer, the interviewer gives a direct assessment of accuracy, depth, clarity, and unsupported claims, then asks a follow-up based on the weakest part. If the answer is incomplete, the interviewer probes rather than immediately supplying the answer. The candidate can say “pause and coach me” to receive a model answer outline.

