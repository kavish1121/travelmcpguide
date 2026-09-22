"""Small MCP client used by the travel planner.

The client starts MCP servers lazily and exposes only the tools required by
the application.  Aviation directory tools such as ``list_airports`` and
``list_airlines`` are deliberately not exposed or called.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


# ---------------------------------------------------------------------------
# Configuration and small flow logger
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

logger = logging.getLogger("travel.mcp_client")
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)-7s [mcp] %(message)s",
)
logger.setLevel(log_level)
# Uvicorn may install its own root handler before this module is imported.
# Apply the configured minimum there as well, otherwise provider DEBUG payloads
# can still leak into the console even when the application logger is INFO.
root_logger = logging.getLogger()
if root_logger.level == logging.NOTSET or root_logger.level < log_level:
    root_logger.setLevel(log_level)
for handler in root_logger.handlers:
    if handler.level == logging.NOTSET or handler.level < log_level:
        handler.setLevel(log_level)

# HTTP/MCP/LangSmith DEBUG logs can contain complete URLs, including API keys.
# Keep provider internals quiet; the application flow logs below remain visible.
for noisy_logger in (
    "httpx",
    "httpcore",
    "urllib3",
    "urllib3.connectionpool",
    "requests",
    "groq",
    "openai",
    "mcp",
    "langsmith",
    "langsmith.client",
    "langchain",
    "langchain_core",
    "langchain_mcp_adapters",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _present(value: str | None) -> str:
    return "present" if value else "missing"


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_SERVER_PATH = PROJECT_DIR / "custom_weather_mcp_server.py"


def _redact(text: str) -> str:
    for secret in (
        TAVILY_API_KEY,
        AVIATIONSTACK_API_KEY,
        OPENWEATHER_API_KEY,
        GROQ_API_KEY,
    ):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?i)([?&](?:tavilyApiKey|access_key|api_key|token)=)[^&\s\"']+",
        r"\1[REDACTED]",
        text,
    )


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            redacted = _redact(message)
            if redacted != message:
                record.msg = redacted
                record.args = ()
        except Exception:
            # Logging must never interrupt the travel request.
            pass
        return True


for handler in logging.getLogger().handlers:
    handler.addFilter(_RedactFilter())

logger.info("Initializing MCP client project=%s", PROJECT_DIR)
logger.info(
    "Configuration TAVILY_API_KEY=%s AVIATIONSTACK_API_KEY=%s "
    "OPENWEATHER_API_KEY=%s GROQ_API_KEY=%s",
    _present(TAVILY_API_KEY),
    _present(AVIATIONSTACK_API_KEY),
    _present(OPENWEATHER_API_KEY),
    _present(GROQ_API_KEY),
)


# Preserve the environment when starting local stdio servers.
AVIATION_ENV = os.environ.copy()
AVIATION_ENV["AVIATION_STACK_API_KEY"] = AVIATIONSTACK_API_KEY or ""

WEATHER_ENV = os.environ.copy()
WEATHER_ENV["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY or ""


# Use a model available in the current Groq setup.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
)


# ---------------------------------------------------------------------------
# MCP server configuration
# ---------------------------------------------------------------------------

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/?tavilyApiKey="
                f"{quote(TAVILY_API_KEY or '', safe='')}"
            ),
        },
        "aviationstack": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["aviationstack-mcp"],
            "env": AVIATION_ENV,
        },
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(WEATHER_SERVER_PATH)],
            "env": WEATHER_ENV,
        },
    }
)

logger.info("MCP servers configured: tavily, aviationstack, weather")


# Only these Aviationstack API-backed operations are used by the application.
# In particular, list_airports and list_airlines are never called.
ALLOWED_AVIATION_TOOLS = {
    "get_flight_status",
    "flights_with_airline",
}


async def _discover_tools(server_name: str) -> dict:
    """Discover and cache tool definitions for one server."""
    logger.info("Discovering tools server=%s", server_name)
    started = time.perf_counter()
    tools = await client.get_tools(server_name=server_name)

    if server_name == "aviationstack":
        tools = [tool for tool in tools if tool.name in ALLOWED_AVIATION_TOOLS]

    tool_map = {tool.name: tool for tool in tools}
    logger.info(
        "Tools ready server=%s count=%d names=%s elapsed_ms=%.0f",
        server_name,
        len(tool_map),
        ", ".join(sorted(tool_map)) or "none",
        (time.perf_counter() - started) * 1000,
    )
    return tool_map


async def get_all_tools():
    """Return available tools without stopping if one server is unavailable."""
    all_tools = []
    for server_name in ("tavily", "aviationstack", "weather"):
        try:
            tool_map = await _discover_tools(server_name)
            all_tools.extend(tool_map.values())
        except Exception:
            logger.exception("Tool discovery failed server=%s", server_name)
    logger.info("Tool discovery complete total=%d", len(all_tools))
    return all_tools


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------

search_tool = None


async def initialize_mcp():
    global search_tool
    if search_tool is not None:
        return
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is missing from .env")
    tools = await _discover_tools("tavily")
    search_tool = tools.get("tavily_search")
    if search_tool is None:
        raise RuntimeError(
            f"Tavily tool tavily_search not found; available={sorted(tools)}"
        )


async def tavily_mcp_search(query: str):
    await initialize_mcp()
    logger.info("Calling tool server=tavily tool=tavily_search")
    started = time.perf_counter()
    try:
        result = await search_tool.ainvoke({"query": query})
        logger.info(
            "Tool completed server=tavily tool=tavily_search elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception:
        logger.exception("Tool failed server=tavily tool=tavily_search")
        raise


# ---------------------------------------------------------------------------
# Aviationstack
# ---------------------------------------------------------------------------

aviation_tools = {}


async def initialize_aviation_tools():
    global aviation_tools
    if aviation_tools:
        return
    if not AVIATIONSTACK_API_KEY:
        raise RuntimeError("AVIATIONSTACK_API_KEY is missing from .env")
    aviation_tools = await _discover_tools("aviationstack")
    if not aviation_tools:
        raise RuntimeError("No allowed Aviationstack tools were returned")


async def aviation_mcp_call(tool_name: str, tool_args: dict | None = None):
    """Call only an allowed Aviationstack API tool."""
    if tool_name not in ALLOWED_AVIATION_TOOLS:
        raise ValueError(
            f"Aviationstack tool {tool_name!r} is disabled. "
            f"Allowed tools: {sorted(ALLOWED_AVIATION_TOOLS)}"
        )

    await initialize_aviation_tools()
    tool = aviation_tools.get(tool_name)
    if tool is None:
        raise RuntimeError(
            f"Aviationstack tool {tool_name!r} is unavailable; "
            f"available={sorted(aviation_tools)}"
        )

    logger.info("Calling tool server=aviationstack tool=%s", tool_name)
    started = time.perf_counter()
    try:
        result = await tool.ainvoke(tool_args or {})
        logger.info(
            "Tool completed server=aviationstack tool=%s elapsed_ms=%.0f",
            tool_name,
            (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception:
        logger.exception("Tool failed server=aviationstack tool=%s", tool_name)
        raise


async def get_flight_status_mcp(flight_iata: str):
    if not isinstance(flight_iata, str) or not flight_iata.strip():
        raise ValueError("flight_iata is required, for example BG376")
    return await aviation_mcp_call(
        "get_flight_status",
        {"flight_iata": flight_iata.strip().upper()},
    )


async def get_airline_flights_mcp(
    airline_name: str,
    number_of_flights: int = 20,
):
    if not isinstance(airline_name, str) or not airline_name.strip():
        raise ValueError("airline_name is required")
    return await aviation_mcp_call(
        "flights_with_airline",
        {
            "airline_name": airline_name.strip(),
            "number_of_flights": number_of_flights,
        },
    )


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    global weather_tool, forecast_tool
    if weather_tool is not None and forecast_tool is not None:
        return
    if not WEATHER_SERVER_PATH.is_file():
        raise FileNotFoundError(f"Weather server not found: {WEATHER_SERVER_PATH}")

    tools = await _discover_tools("weather")
    weather_tool = tools.get("get_current_weather")
    forecast_tool = tools.get("get_forecast")
    missing = [
        name for name, tool in {
            "get_current_weather": weather_tool,
            "get_forecast": forecast_tool,
        }.items()
        if tool is None
    ]
    if missing:
        raise RuntimeError(f"Missing weather tools={missing}; available={sorted(tools)}")


async def weather_mcp_search(city: str):
    await initialize_weather_tools()
    logger.info("Calling tool server=weather tool=get_current_weather city=%s", city)
    started = time.perf_counter()
    try:
        result = await weather_tool.ainvoke({"city": city})
        logger.info(
            "Tool completed server=weather tool=get_current_weather elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception:
        logger.exception("Tool failed server=weather tool=get_current_weather")
        raise


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()
    logger.info("Calling tool server=weather tool=get_forecast city=%s", city)
    started = time.perf_counter()
    try:
        result = await forecast_tool.ainvoke({"city": city})
        logger.info(
            "Tool completed server=weather tool=get_forecast elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception:
        logger.exception("Tool failed server=weather tool=get_forecast")
        raise


# ---------------------------------------------------------------------------
# Destination extraction
# ---------------------------------------------------------------------------

def extract_destination(query: str) -> str:
    logger.info("Extracting destination query_chars=%d", len(query))
    response = llm.invoke(
        "Extract only the destination city or country.\n"
        f"Query:\n{query}\n"
        "Return only the destination name."
    )
    destination = response.content.strip()
    logger.info("Destination extracted=%s", destination)
    return destination
