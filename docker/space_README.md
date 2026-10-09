---
title: OrbitDesk
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Enterprise helpdesk copilot demo (RAG + agent + guardrails)
---

# OrbitDesk

Helpdesk copilot for the **fictional** company Orbit Systems: answers from 12 made-up HR policies and IT
runbooks with verified citations, role-based access control, and an agent that can check leave balances
and raise IT tickets. All data is fictional.

Generation needs the `GROQ_API_KEY` Space secret; without it, the UI explains that generation is off and
retrieval still works. Optional `DATABASE_URL` (hosted Postgres with pgvector, e.g. Neon) switches the data,
vectors and full-text search to Postgres; otherwise the Space uses SQLite + Chroma. The MCP server is at
`/mcp/` (bearer token required). Source, evals and architecture docs: see the GitHub repository.
