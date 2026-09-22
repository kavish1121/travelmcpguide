"""Simple LangGraph travel planner backed by the local MCP client."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import operator
import re
import uuid
from typing import Annotated, TypedDict
from urllib.parse import parse_qs, urlsplit

import certifi
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from mcp_client import (
    extract_destination,
    forecast_mcp_search,
    get_flight_status_mcp,
    logger as mcp_logger,
    tavily_mcp_search,
    weather_mcp_search,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# psycopg's async connection requires the selector loop on Windows.
if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = mcp_logger.getChild("backend")
logger.setLevel(mcp_logger.level)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    max_tokens=1000,
)

DEFAULT_ORIGIN_CITY = os.getenv("DEFAULT_ORIGIN_CITY", "Delhi")
DEFAULT_ORIGIN_COUNTRY = os.getenv("DEFAULT_ORIGIN_COUNTRY", "India")
GATEWAY_CITIES = {
    "india": "Delhi",
    "bangladesh": "Dhaka",
    "new zealand": "Wellington",
    "united states": "New York",
    "usa": "New York",
}
DESTINATION_CITIES = {
    "japan": "Tokyo",
    "new zealand": "Wellington",
    "india": "Delhi",
}


# ---------------------------------------------------------------------------
# State and utility functions
# ---------------------------------------------------------------------------

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_iata: str
    origin: str
    destination: str
    flight_results: str
    hotel_results: str
    itinerary: str
    weather_results: str
    llm_calls: int


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from .env")
    if "sslmode" not in parse_qs(urlsplit(database_url).query):
        database_url += ("&" if "?" in database_url else "?") + "sslmode=require"
    return database_url


def clip(value, limit: int = 3500) -> str:
    """Keep prompts bounded so they do not exceed the model token limit."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit]


def normalized_flight(value: str) -> str:
    """Validate and normalize an IATA flight number such as NZ5231."""
    normalized = re.sub(r"\s+", "", value or "").upper()
    if not re.fullmatch(r"(?:[A-Z]{2}|[A-Z]\d|\d[A-Z])\d{1,4}[A-Z]?", normalized):
        raise ValueError("Provide an IATA flight number such as NZ5231")
    return normalized


def extract_flight_numbers(value) -> list[str]:
    """Extract only flight numbers supported by nearby flight wording.

    This avoids treating times such as ``on 01:40`` or phrases such as
    ``up to 9 months`` as flight numbers.
    """
    text = clip(value, 12000)
    result = []
    pattern = re.compile(
        r"(?<![A-Z0-9])([A-Z]{2}\s*\d{1,4})(?![A-Z0-9])",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 90):match.start()]
        after = text[match.end():match.end() + 90]
        context = before + " " + after
        # Require an explicit flight label nearby. Do not infer a flight
        # number merely from a time, date, airport code, or result identifier.
        if not re.search(
            r"\bflight(?:\s+(?:number|no\.?))?\b|\bflight\s*(?:is|:|#)",
            context,
            flags=re.IGNORECASE,
        ):
            continue
        number = re.sub(r"\s+", "", match.group(1)).upper()
        if number not in result:
            result.append(number)
    return result[:3]


def tool_has_records(value) -> bool:
    """Return true only when an Aviationstack response contains flight data."""
    text = clip(value, 10000)
    return bool(
        re.search(r'"count"\s*:\s*[1-9]\d*', text)
        or re.search(r'"data"\s*:\s*\[\s*\{', text)
    )


def verified_flight_prompt(value) -> str:
    """Return only flight data that contains an actual API record.

    Tavily discovery is useful for finding possible flight numbers, but it is
    not a flight-status source.  Do not pass those unverified candidates to an
    LLM as if they were bookable or confirmed flights.
    """
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        data = None
    if not isinstance(data, dict) or not data.get("flights"):
        return (
            "No verified flight record was returned by AviationStack. "
            "Do not state a flight number, airline, fare, schedule, or availability."
        )
    return clip(
        json.dumps(
            {
                "requested_route": data.get("requested_route"),
                "source": data.get("source"),
                "flights": data.get("flights"),
                "limitations": data.get("limitations", []),
            },
            ensure_ascii=False,
            default=str,
        )
    )


def extract_airline_names(value) -> list[str]:
    text = clip(value, 12000)
    matches = re.findall(
        r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,3}\s+"
        r"(?:Airlines?|Airways))\b",
        text,
    )
    result = []
    for name in matches:
        name = re.sub(r"\s+", " ", name).strip()
        if name.casefold().startswith(("schedule", "schedules")):
            continue
        if name.casefold() not in {item.casefold() for item in result}:
            result.append(name)
    return result[:3]


def route_from_query(query: str) -> tuple[str, str]:
    """Use an explicit origin when present; otherwise use the configured default."""
    origin = DEFAULT_ORIGIN_CITY
    match = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z .'-]*?)(?=\s+(?:including|with|for|to|and|under)\b|[,.;]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        origin_value = re.sub(r"\s+", " ", match.group(1)).strip()
        origin = GATEWAY_CITIES.get(origin_value.casefold(), origin_value.title())
    return origin, ""


async def destination_from_query(query: str) -> str:
    extracted = await asyncio.to_thread(extract_destination, query)
    extracted = re.sub(r"\s+", " ", extracted).strip(" .,\n")
    key = extracted.casefold()
    return DESTINATION_CITIES.get(key, extracted)


async def call_llm(messages):
    response = await llm.ainvoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def flight_agent(state: TravelState):
    query = state["user_query"]
    origin, _ = route_from_query(query)
    destination = await destination_from_query(query)
    logger.info("Flight flow origin=%s destination=%s", origin, destination)

    try:
        explicit_flight = state.get("flight_iata", "")
        if explicit_flight:
            flight_numbers = [explicit_flight]
            airlines = []
            logger.info("Flight discovery skipped explicit_flight=%s", explicit_flight)
        else:
            search_query = (
                f"flight number airline schedule from {origin} to {destination}; "
                "include exact IATA flight numbers and airports"
            )
            logger.info("Flight discovery started")
            discovery = await tavily_mcp_search(search_query)
            flight_numbers = extract_flight_numbers(discovery)
            airlines = extract_airline_names(discovery)
            logger.info(
                "Flight discovery complete candidates=%s airlines=%s",
                flight_numbers,
                airlines,
            )

        records = []
        for flight_number in flight_numbers:
            try:
                status = await get_flight_status_mcp(flight_number)
                if tool_has_records(status):
                    records.append({"flight_iata": flight_number, "status": status})
                else:
                    logger.info("No current Aviationstack record flight=%s", flight_number)
            except Exception as error:
                logger.warning("Flight status failed flight=%s error=%s", flight_number, error)

        if records:
            flight_data = {
                "requested_route": {"origin": origin, "destination": destination},
                "source": (
                    "explicit flight + Aviationstack get_flight_status"
                    if explicit_flight
                    else "Tavily discovery + Aviationstack get_flight_status"
                ),
                "flights": records,
                "unverified_airlines_found": airlines,
                "limitations": [
                    "Current status only; fares, seats, and future availability are not verified."
                ],
            }
        else:
            flight_data = {
                "requested_route": {"origin": origin, "destination": destination},
                "source": "Tavily route discovery",
                "flights": [],
                "unverified_flight_candidates": flight_numbers,
                "unverified_airlines_found": airlines,
                "message": (
                    "No verified current Aviationstack record was returned for the "
                    "discovered flight candidates."
                    if flight_numbers
                    else "No source-backed flight number was found; no flight was invented."
                ),
            }
        flight_result = json.dumps(flight_data, ensure_ascii=False, default=str)
    except Exception as error:
        logger.exception("Flight flow failed")
        destination = destination or "the requested destination"
        flight_result = f"Flight information unavailable: {error}"

    return {
        "origin": origin,
        "destination": destination,
        "flight_results": flight_result,
        "messages": [AIMessage(content="Flight information fetched.")],
    }


async def hotel_agent(state: TravelState):
    destination = state.get("destination") or "the destination"
    logger.info("Hotel flow started destination=%s", destination)
    try:
        result = await tavily_mcp_search(
            f"Hotels in {destination}; official websites, nightly rates, and budget options"
        )
        hotel_result = clip(result)
    except Exception as error:
        logger.exception("Hotel flow failed")
        hotel_result = f"Hotel information unavailable: {error}"
    return {
        "hotel_results": hotel_result,
        "messages": [AIMessage(content="Hotel information fetched.")],
    }


async def weather_agent(state: TravelState):
    city = state.get("destination") or ""
    logger.info("Weather flow started city=%s", city)
    if not city:
        return {"weather_results": "Weather lookup unavailable: destination is missing."}

    current = forecast = ""
    try:
        current = clip(await weather_mcp_search(city), 1800)
    except Exception as error:
        logger.exception("Current weather failed")
        current = f"Current weather unavailable: {error}"
    try:
        forecast = clip(await forecast_mcp_search(city), 1800)
    except Exception as error:
        logger.exception("Forecast failed")
        forecast = f"Forecast unavailable: {error}"

    return {
        "weather_results": f"Current weather:\n{current}\n\nForecast:\n{forecast}",
        "messages": [AIMessage(content="Weather information fetched.")],
    }


async def itinerary_agent(state: TravelState):
    logger.info("Itinerary flow started")
    prompt = f"""Create a practical travel itinerary.

User request:
{clip(state['user_query'], 1800)}

Flight information:
{verified_flight_prompt(state['flight_results'])}

Hotel information:
{clip(state['hotel_results'])}

Weather information:
{clip(state['weather_results'])}

Keep the answer concise, budget-aware, and clearly state anything unverified.
Never turn an unverified discovery candidate into a confirmed flight.  Do not
invent flight numbers, fares, schedules, seats, or availability.
"""
    try:
        itinerary = await call_llm([
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ])
    except Exception as error:
        logger.exception("Itinerary flow failed")
        itinerary = f"Itinerary generation unavailable: {error}"
    return {
        "itinerary": itinerary,
        "messages": [AIMessage(content=itinerary)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def final_agent(state: TravelState):
    logger.info("Final response flow started")
    prompt = f"""Prepare the final travel response.

User request: {clip(state['user_query'], 1500)}
Route: {state.get('origin', '')} to {state.get('destination', '')}

Flights:
{verified_flight_prompt(state['flight_results'])}

Hotels:
{clip(state['hotel_results'], 2200)}

Weather:
{clip(state['weather_results'], 2200)}

Itinerary:
{clip(state['itinerary'], 2600)}

Use these sections: Trip Summary, Flight Information, Hotel Suggestions,
Weather Information, Day-by-Day Itinerary, Estimated Budget, and Final
Recommendations. Do not invent flight prices or availability.
If no verified flight record is present, say so plainly and do not mention any
unverified candidate as an actual flight.
"""
    try:
        answer = await call_llm([
            SystemMessage(content="You are a professional travel assistant."),
            HumanMessage(content=prompt),
        ])
    except Exception as error:
        logger.exception("Final response flow failed")
        answer = state.get("itinerary") or f"Final response unavailable: {error}"
    return {
        "messages": [AIMessage(content=answer)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Graph and public execution functions
# ---------------------------------------------------------------------------

graph = StateGraph(TravelState)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

async def run_travel_agent_async(
    user_input: str,
    thread_id: str | None = None,
    flight_iata: str | None = None,
):
    thread_id = thread_id or f"user_{uuid.uuid4().hex}"
    logger.info("Travel flow started thread_id=%s", thread_id)
    # PostgresSaver is synchronous and cannot service graph.ainvoke().
    # Keep the async saver alive for the complete graph invocation.
    async with AsyncPostgresSaver.from_conn_string(get_database_url()) as checkpointer:
        await checkpointer.setup()
        travel_graph = graph.compile(checkpointer=checkpointer)
        result = await travel_graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "flight_iata": flight_iata or "",
                "origin": "",
                "destination": "",
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "itinerary": "",
                "llm_calls": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    logger.info("Travel flow completed thread_id=%s", thread_id)
    return {
        "thread_id": thread_id,
        "answer": result["messages"][-1].content,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    flight_iata: str | None = None,
):
    """Synchronous entry point for scripts; FastAPI should await the async one."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_travel_agent_async(user_input, thread_id, flight_iata)
        )
    raise RuntimeError("In async code, await run_travel_agent_async(...) instead")
