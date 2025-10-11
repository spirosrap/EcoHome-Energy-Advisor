"""
Tools for EcoHome Energy Advisor Agent
"""
import hashlib
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

try:  # Support running as module or script
    from .models.energy import DatabaseManager  # type: ignore
except ImportError:  # pragma: no cover
    from models.energy import DatabaseManager

try:
    from .rag_setup import build_vector_store, VECTOR_DIR  # type: ignore
except ImportError:  # pragma: no cover
    from rag_setup import build_vector_store, VECTOR_DIR

PERSIST_DIR = VECTOR_DIR


def _embedding_kwargs() -> Dict[str, str]:
    """Build embedding client settings from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VOCAREUM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI credentials. Set OPENAI_API_KEY or VOCAREUM_API_KEY."
        )

    base_url = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_HOST")
    )
    if not base_url and os.getenv("VOCAREUM_API_KEY"):
        base_url = "https://openai.vocareum.com/v1"

    kwargs = {"openai_api_key": api_key}
    if base_url:
        kwargs["openai_api_base"] = base_url
    return kwargs


def _seeded_random(*keys: str) -> random.Random:
    """Create a deterministic random generator from provided keys."""
    seed_str = "|".join(keys).encode("utf-8")
    digest = hashlib.sha256(seed_str).hexdigest()[:16]
    seed = int(digest, 16) % (2**32)
    return random.Random(seed)


def _seasonal_temperature(base_date: datetime) -> float:
    """Return a seasonal offset based on day of year."""
    day_of_year = base_date.timetuple().tm_yday
    # Simple sinusoidal model: peak in July (~day 200), trough in January (~day 20)
    return 10 * math.sin((2 * math.pi * (day_of_year - 200)) / 365.25)


# Initialize database manager
db_manager = DatabaseManager()
_VECTORSTORE_CACHE: Optional[Chroma] = None


def _get_vectorstore(force_refresh: bool = False) -> Chroma:
    """Return a Chroma vector store, building it if necessary."""
    global _VECTORSTORE_CACHE
    persist_directory = PERSIST_DIR
    embeddings = OpenAIEmbeddings(**_embedding_kwargs())

    chroma_path = persist_directory / "chroma.sqlite3"
    if force_refresh or not chroma_path.exists():
        build_vector_store(force=True)
        _VECTORSTORE_CACHE = None

    if _VECTORSTORE_CACHE is None:
        _VECTORSTORE_CACHE = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )
    return _VECTORSTORE_CACHE

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
    
    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)
    
    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
        E.g:
        forecast = {
            "location": ...,
            "forecast_days": ...,
            "current": {
                "temperature_c": ...,
                "condition": random.choice(["sunny", "partly_cloudy", "cloudy"]),
                "humidity": ...,
                "wind_speed": ...
            },
            "hourly": [
                {
                    "hour": ..., # for hour in range(24)
                    "temperature_c": ...,
                    "condition": ...,
                    "solar_irradiance": ...,
                    "humidity": ...,
                    "wind_speed": ...
                },
            ]
        }
    """
    try:
        days = int(days)
    except (TypeError, ValueError):
        return {"error": "Days must be an integer between 1 and 7."}

    if days < 1 or days > 7:
        days = max(1, min(days, 7))

    location = location.strip() or "Unknown"
    generated_at = datetime.now(timezone.utc)

    baseline_rng = _seeded_random(location.lower())
    location_temp_offset = baseline_rng.uniform(-4.0, 4.0)
    location_humidity_bias = baseline_rng.uniform(-5.0, 5.0)
    location_wind_bias = baseline_rng.uniform(-1.5, 1.5)

    condition_multipliers = {
        "sunny": {"probability": 0.38, "irradiance": 1.0, "humidity": -8},
        "partly_cloudy": {"probability": 0.28, "irradiance": 0.7, "humidity": 0},
        "cloudy": {"probability": 0.22, "irradiance": 0.45, "humidity": 6},
        "rainy": {"probability": 0.12, "irradiance": 0.2, "humidity": 12},
    }

    hourly_forecast = []
    daily_summaries = []

    for day_index in range(days):
        day_dt = generated_at + timedelta(days=day_index)
        seasonal_offset = _seasonal_temperature(day_dt)
        day_rng = _seeded_random(location.lower(), str(day_dt.date()))

        base_temp = 18 + location_temp_offset + seasonal_offset + day_rng.uniform(-2.0, 2.0)
        amplitude = 7.5 + day_rng.uniform(-1.0, 1.5)

        # Adjust condition weights slightly per day
        condition_labels = list(condition_multipliers.keys())
        base_weights = []
        for label in condition_labels:
            base = condition_multipliers[label]["probability"]
            tweak = day_rng.uniform(-0.05, 0.05)
            if label == "sunny":
                tweak += max(0, location_temp_offset) / 20
            elif label in ("cloudy", "rainy"):
                tweak += max(0, -location_temp_offset) / 25
            base_weights.append(max(0.05, base + tweak))
        total_weight = sum(base_weights)
        weights = [w / total_weight for w in base_weights]
        dominant_condition = day_rng.choices(condition_labels, weights=weights, k=1)[0]
        condition_factor = condition_multipliers[dominant_condition]

        day_high = -float("inf")
        day_low = float("inf")
        day_humidity_total = 0.0
        day_wind_total = 0.0

        for hour in range(24):
            hour_dt = day_dt.replace(hour=hour, minute=0, second=0, microsecond=0)
            diurnal = math.sin(math.pi * (hour - 6) / 12)
            temp = base_temp + amplitude * diurnal + day_rng.uniform(-0.6, 0.6)
            humidity = min(100, max(30, 55 + condition_factor["humidity"] + location_humidity_bias + day_rng.uniform(-8, 8)))
            wind_speed = max(0, 8 + location_wind_bias + day_rng.uniform(-2.5, 2.5))
            wind_speed = round(wind_speed, 1)

            irradiance = 0.0
            if 6 <= hour <= 18:
                solar_curve = max(0.0, math.sin(math.pi * (hour - 6) / 12))
                irradiance = round(850 * solar_curve * condition_factor["irradiance"], 1)

            temp = round(temp, 1)
            humidity = round(humidity, 1)

            condition = dominant_condition
            if day_rng.random() < 0.2:
                condition = day_rng.choice(condition_labels)

            hourly_entry = {
                "timestamp": hour_dt.isoformat(),
                "date": hour_dt.date().isoformat(),
                "hour": hour,
                "temperature_c": temp,
                "condition": condition,
                "humidity": humidity,
                "wind_speed_kph": wind_speed,
                "solar_irradiance_wm2": irradiance,
            }
            hourly_forecast.append(hourly_entry)

            day_high = max(day_high, temp)
            day_low = min(day_low, temp)
            day_humidity_total += humidity
            day_wind_total += wind_speed

        daily_summaries.append({
            "date": day_dt.date().isoformat(),
            "condition": dominant_condition,
            "temperature_high_c": round(day_high, 1),
            "temperature_low_c": round(day_low, 1),
            "average_humidity": round(day_humidity_total / 24, 1),
            "average_wind_speed_kph": round(day_wind_total / 24, 1),
            "sunrise_local": day_dt.replace(hour=6, minute=18, second=0, microsecond=0).isoformat(),
            "sunset_local": day_dt.replace(hour=18, minute=42, second=0, microsecond=0).isoformat(),
        })

    current_hour = generated_at.astimezone(timezone.utc).hour
    current_date = generated_at.date().isoformat()
    current_data = next(
        (entry for entry in hourly_forecast if entry["date"] == current_date and entry["hour"] == current_hour),
        hourly_forecast[0],
    )

    return {
        "location": location,
        "forecast_generated_at": generated_at.isoformat(),
        "forecast_days": days,
        "current": current_data,
        "daily": daily_summaries,
        "hourly": hourly_forecast,
    }

@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.
    
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates 
        E.g: 
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        target_date = datetime.now().date()
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return {"error": "Date must be in YYYY-MM-DD format."}
    
    rng = _seeded_random("pricing", str(target_date))
    is_weekend = target_date.weekday() >= 5

    hourly_rates = []
    tier_definitions = []

    if is_weekend:
        tier_definitions = [
            ("off_peak", range(0, 8), 0.115, 0.0, 280),
            ("mid_peak", range(8, 20), 0.165, 0.02, 320),
            ("off_peak", range(20, 24), 0.115, 0.0, 280),
        ]
    else:
        tier_definitions = [
            ("off_peak", range(0, 6), 0.11, 0.0, 270),
            ("shoulder", range(6, 10), 0.16, 0.01, 310),
            ("mid_peak", range(10, 17), 0.19, 0.03, 340),
            ("peak", range(17, 21), 0.29, 0.06, 400),
            ("shoulder", range(21, 23), 0.16, 0.01, 300),
            ("off_peak", range(23, 24), 0.11, 0.0, 270),
        ]

    tier_lookup = {}
    for period, hours_range, rate, demand_charge, carbon in tier_definitions:
        for hour in hours_range:
            tier_lookup[hour] = (period, rate, demand_charge, carbon)

    for hour in range(24):
        period, base_rate, demand_charge, carbon = tier_lookup.get(
            hour, ("off_peak", 0.12, 0.0, 280)
        )

        variability = rng.uniform(-0.01, 0.015)
        temperature_factor = 0.0
        if period == "peak":
            temperature_factor = rng.uniform(0.0, 0.02)
        elif period == "mid_peak":
            temperature_factor = rng.uniform(-0.005, 0.01)

        final_rate = max(0.08, base_rate + variability + temperature_factor)

        carbon_adjustment = rng.uniform(-15, 20)
        carbon_intensity = int(max(150, carbon + carbon_adjustment))

        hourly_rates.append({
            "hour": hour,
            "period": period,
            "rate_usd_per_kwh": round(final_rate, 3),
            "demand_charge_usd_per_kwh": round(demand_charge, 3),
            "estimated_carbon_intensity_g_per_kwh": carbon_intensity,
        })

    average_rate = round(sum(item["rate_usd_per_kwh"] for item in hourly_rates) / 24, 3)
    peak_rate = max(hourly_rates, key=lambda item: item["rate_usd_per_kwh"])

    return {
        "date": target_date.isoformat(),
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "average_rate_usd": average_rate,
        "peak_rate_usd": peak_rate["rate_usd_per_kwh"],
        "peak_period_hour": peak_rate["hour"],
        "hourly_rates": hourly_rates,
    }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    if not query or not isinstance(query, str):
        return {"error": "Query must be a non-empty string."}

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, 10))

    try:
        vectorstore = _get_vectorstore()
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
