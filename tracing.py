"""Phoenix / OpenInference tracing setup.

Importing this module (before the agent is built) sends OpenTelemetry traces
to a Phoenix server. The OpenInference LangChain instrumentor covers LangGraph
too, so every agent run produces a trace with spans for the graph nodes,
each LLM call, and each tool call.

Configure with environment variables (see .env.example):
- PHOENIX_COLLECTOR_ENDPOINT: where Phoenix is running (default http://localhost:6006)
- PHOENIX_PROJECT_NAME: the project the traces appear under in Phoenix
"""

import os

from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "travel-agent"),
    batch=True,  # export spans in the background so tracing doesn't slow down responses
    auto_instrument=True,  # turns on every installed OpenInference instrumentor (here: LangChain/LangGraph)
)
