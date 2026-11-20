"""Custom tools for the travel assistant agent.

get_weather_forecast looks up a city with the Open-Meteo geocoding API, then
fetches a daily forecast from the Open-Meteo forecast API. Both APIs are free
and need no API key.

The tool returns structured output: a dict validated against the
WeatherForecast Pydantic model, or a ToolError dict when something goes wrong.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT_SECONDS = 10.0
MAX_FORECAST_DAYS = 16  # Open-Meteo's limit

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ---------- Input schema (what the LLM sees when deciding to call the tool) ----------


class WeatherForecastInput(BaseModel):
    city: str = Field(
        description="Name of the city to get the forecast for, e.g. 'Lisbon' or 'Tokyo'. "
        "City name only, without the country."
    )
    days: int = Field(
        default=3,
        description=f"Number of days to forecast, starting today (1 to {MAX_FORECAST_DAYS}).",
    )


# ---------- Output schemas (structured output returned to the agent) ----------


class DailyForecast(BaseModel):
    date: str
    conditions: str
    temp_max_c: float | None
    temp_min_c: float | None
    precipitation_chance_pct: int | None


class WeatherForecast(BaseModel):
    city: str
    country: str | None
    latitude: float
    longitude: float
    timezone: str | None
    forecast: list[DailyForecast]


class ToolError(BaseModel):
    error: str


# ---------- Tool ----------


@tool("get_weather_forecast", args_schema=WeatherForecastInput)
def get_weather_forecast(city: str, days: int = 3) -> dict:
    """Get the daily weather forecast for a travel destination: conditions,
    high/low temperatures in Celsius, and chance of rain for each day.
    Use this whenever the user asks about weather, what to pack, or the best
    days for outdoor activities at a destination. Forecasts are available
    for up to 16 days ahead."""
    days = max(1, min(days, MAX_FORECAST_DAYS))

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
            geo = client.get(
                GEOCODING_URL,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results")
            if not results:
                return ToolError(
                    error=f"Could not find a city named '{city}'. Try a different spelling or a nearby larger city."
                ).model_dump()
            place = results[0]

            weather = client.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )
            weather.raise_for_status()
            daily = weather.json()["daily"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return ToolError(error=f"Weather service unavailable: {exc}").model_dump()

    forecast = [
        DailyForecast(
            date=date,
            conditions=WEATHER_CODES.get(code, f"Unknown (code {code})"),
            temp_max_c=t_max,
            temp_min_c=t_min,
            precipitation_chance_pct=rain,
        )
        for date, code, t_max, t_min, rain in zip(
            daily["time"],
            daily["weather_code"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
    ]

    return WeatherForecast(
        city=place["name"],
        country=place.get("country"),
        latitude=place["latitude"],
        longitude=place["longitude"],
        timezone=place.get("timezone"),
        forecast=forecast,
    ).model_dump()
