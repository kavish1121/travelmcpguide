"""FastAPI frontend for the simplified travel planner."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import normalized_flight, run_travel_agent_async
from mcp_client import logger


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="Travel Agent AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.1.0",
)

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None
    flight_iata: str | None = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    user_message = request_data.message.strip()
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Message cannot be empty."},
        )

    flight_iata = (request_data.flight_iata or "").strip()
    if flight_iata:
        try:
            flight_iata = normalized_flight(flight_iata)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid flight_iata. Use an IATA flight number such as NZ5231.",
                },
            )

    thread_id = (request_data.thread_id or "").strip() or None
    logger.info(
        "Travel request received message_chars=%d explicit_flight_present=%s",
        len(user_message),
        bool(flight_iata),
    )

    try:
        # The backend graph is asynchronous; do not call asyncio.run here.
        result = await run_travel_agent_async(
            user_input=user_message,
            thread_id=thread_id,
            flight_iata=flight_iata or None,
        )
        logger.info("Travel request completed")
        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result.get("flight_results", ""),
                "hotel_results": result.get("hotel_results", ""),
                "weather_results": result.get("weather_results", ""),
                "itinerary": result.get("itinerary", ""),
                "llm_calls": result.get("llm_calls", 0),
            }
        )
    except Exception:
        logger.exception("Travel request failed")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Travel planning failed. Check the application logs.",
            },
        )


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Travel Planner API is running"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
