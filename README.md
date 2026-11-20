# LangGraph Agent with Web Search

A simple LangGraph agent that can search the web using DuckDuckGo, exposed via a FastAPI server.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- OpenAI API key

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

3. Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_actual_api_key
```

## Running the API

Start the FastAPI server:

```bash
poetry run uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?"}'
```

**Response:**

```json
{
  "response": "Based on my search, here are the latest developments in AI..."
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```


## Observability with Arize Phoenix

Every request to the agent is traced with OpenTelemetry using the OpenInference LangChain instrumentor (which also covers LangGraph) and sent to a local Phoenix server. Each trace shows the graph nodes, every LLM call (prompts, responses, token counts) and every tool call (inputs and outputs).

### 1. Start Phoenix

Phoenix runs in its own Python environment (Python 3.10+), separate from the app, so its dependencies don't clash with the app's:

```bash
conda create -n phoenix python=3.11 -y
conda activate phoenix
pip install arize-phoenix
phoenix serve
```

Phoenix is then available at `http://localhost:6006`. Leave this terminal running.

### 2. Configure the app

Add the Phoenix settings from `.env.example` to your `.env`:

```
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=travel-agent
```

### 3. Run the app and send a request

```bash
poetry run uvicorn api:app --reload
```

Traces appear in Phoenix under the `travel-agent` project.

### How it's wired

`tracing.py` calls `phoenix.otel.register(..., auto_instrument=True)`, which sets up an OpenTelemetry tracer provider that exports to Phoenix and turns on the installed OpenInference instrumentors. `api.py` imports it first, before the agent is built, so all LangChain/LangGraph activity is captured. No changes to the agent code were needed.

## Project Structure

```
se-interview/
├── pyproject.toml   # Poetry dependencies
├── .env.example     # Environment variable template
├── README.md        # This file
├── agent.py         # LangGraph agent implementation
├── tools.py         # Custom agent tools (weather forecast)
├── tracing.py       # Phoenix / OpenInference tracing setup
└── api.py           # FastAPI server
```

## Extension: Weather Forecast Tool

The agent has a second tool, `get_weather_forecast`, in `tools.py`. Given a city and a number of days, it returns each day's conditions, high and low temperatures, and chance of rain.

**Example queries:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am going to Lisbon this weekend. What should I pack?"}'
```

- "What's the weather in Tokyo for the next 5 days?" (weather tool only)
- "Find me things to do in Barcelona on the dry days this week" (weather tool, then web search)
- "What are the best museums in Paris?" (web search only)

**Example tool output (structured, illustrative values):**

```json
{
  "city": "Lisbon",
  "country": "Portugal",
  "latitude": 38.71667,
  "longitude": -9.13333,
  "timezone": "Europe/Lisbon",
  "forecast": [
    {"date": "2026-11-18", "conditions": "Partly cloudy", "temp_max_c": 18.2, "temp_min_c": 11.4, "precipitation_chance_pct": 10}
  ]
}
```

### Design decisions

- **Why weather:** it is a common travel need (packing, choosing days for outdoor plans) that web search answers poorly, because search results are unstructured and often stale. It also gives the agent a real choice between tools, which makes tool-selection behaviour visible in traces.
- **Data source:** [Open-Meteo](https://open-meteo.com/) geocoding and forecast APIs. They are free, need no API key, and return clean JSON, so the project still only needs an OpenAI key. The tool uses `httpx`, which the OpenAI SDK already installs, so no new dependencies were added.
- **Structured output:** the tool's inputs and outputs are defined as Pydantic models (`WeatherForecastInput`, `WeatherForecast`, `DailyForecast`). The input schema and the tool's docstring are what the LLM sees when deciding whether and how to call it. The output is validated and returned as a dict. Raw weather codes are converted to readable text, and field names include units (`temp_max_c`, `precipitation_chance_pct`), so the LLM does not have to guess.
- **Errors as data:** an unknown city or an unreachable API returns a structured `{"error": ...}` object instead of raising. The agent can then explain the problem to the user instead of the request failing with a 500.
- **Tool node changes (`agent.py`):** structured tool results are serialized to JSON before being passed back to the LLM as a `ToolMessage`. Any tool exception, including DuckDuckGo rate limits, is caught and returned as an error message with `status="error"`, so one failing tool no longer crashes the whole request.
- **Prompting:** the system prompt now frames the agent as a travel assistant and says when to use each tool, so the LLM picks the weather tool for weather questions and search for everything else.
- **Input guardrails:** `days` is clamped to 1–16 (Open-Meteo's limit), so an out-of-range request still returns a result.

## How It Works

1. The agent receives a user message via the `/chat` endpoint
2. It calls GPT-4o with the message and available tools (DuckDuckGo search)
3. If the LLM decides to search, it executes the search and feeds results back
4. The loop continues until the LLM provides a final response
5. The response is returned to the user
