"""
Utility for initializing the EcoHome SQLite database with sample data."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Dict, Tuple

try:  # Allow running both as module and as script
    from .models.energy import DatabaseManager, EnergyUsage, SolarGeneration  # type: ignore
except ImportError:  # pragma: no cover
    from models.energy import DatabaseManager, EnergyUsage, SolarGeneration

# Paths relative to this module so execution works from any CWD.
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "energy_data.db"


def _should_populate(db_manager: DatabaseManager) -> bool:
    """Return True if either table is empty."""
    session = db_manager.get_session()
    try:
        usage_exists = session.query(EnergyUsage.id).first() is not None
        generation_exists = session.query(SolarGeneration.id).first() is not None
        return not (usage_exists and generation_exists)
    finally:
        session.close()


def _populate_energy_usage(db_manager: DatabaseManager, rng: random.Random,
                           start_date: datetime, days: int) -> int:
    """Insert synthetic energy usage samples modeled on the notebook logic."""
    device_profiles = {
        "EV": {"base_kwh": 10, "variation": 5, "peak_hours": {18, 19, 20, 21}},
        "HVAC": {"base_kwh": 2, "variation": 1, "peak_hours": {12, 13, 14, 15, 16, 17}},
        "appliance": {"base_kwh": 1.5, "variation": 0.5, "peak_hours": {19, 20, 21, 22}},
    }
    appliance_options = ["Dishwasher", "Washing Machine", "Dryer"]

    records = 0
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        for hour in range(24):
            timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            for device_type, profile in device_profiles.items():
                variation = rng.uniform(-profile["variation"], profile["variation"])
                peak_multiplier = 1.5 if hour in profile["peak_hours"] else 0.8
                consumption = max(0.0, (profile["base_kwh"] + variation) * peak_multiplier)
                price_per_kwh = 0.15 if hour in profile["peak_hours"] else 0.10
                cost = consumption * price_per_kwh

                device_name = {
                    "EV": "Tesla Model 3",
                    "HVAC": "Main AC Unit",
                    "appliance": rng.choice(appliance_options),
                }[device_type]

                db_manager.add_usage_record(
                    timestamp=timestamp,
                    consumption_kwh=consumption,
                    device_type=device_type,
                    device_name=device_name,
                    cost_usd=cost,
                )
                records += 1
    return records


def _populate_solar_generation(db_manager: DatabaseManager, rng: random.Random,
                               start_date: datetime, days: int) -> int:
    """Insert synthetic solar generation samples modeled on the notebook logic."""
    weather_conditions = {
        "sunny": {"multiplier": 1.0, "probability": 0.4},
        "partly_cloudy": {"multiplier": 0.6, "probability": 0.3},
        "cloudy": {"multiplier": 0.3, "probability": 0.2},
        "rainy": {"multiplier": 0.1, "probability": 0.1},
    }

    records = 0
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        weather_choice = rng.choices(
            population=list(weather_conditions.keys()),
            weights=[cfg["probability"] for cfg in weather_conditions.values()],
        )[0]
        weather_multiplier = weather_conditions[weather_choice]["multiplier"]

        for hour in range(24):
            if not 6 <= hour <= 18:
                continue

            timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            hour_factor = 1 - abs(hour - 12) / 6  # Peak production at noon
            base_generation = 5.0 * hour_factor
            generation = max(0.0, base_generation * weather_multiplier * rng.uniform(0.8, 1.2))
            if generation <= 0:
                continue

            base_temp = 20 + rng.uniform(-5, 5)
            temp_factor = 1.0 if 15 <= base_temp <= 35 else 0.9
            irradiance = 800 * hour_factor * weather_multiplier

            db_manager.add_generation_record(
                timestamp=timestamp,
                generation_kwh=generation,
                weather_condition=weather_choice,
                temperature_c=base_temp * temp_factor,
                solar_irradiance=irradiance,
            )
            records += 1
    return records


def initialize_database(days: int = 30, seed: int = 42,
                        db_path: Path = DEFAULT_DB_PATH) -> Dict[str, int]:
    """
    Create tables and populate sample data if the database is empty.

    Returns a summary dictionary with counts of inserted records.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_manager = DatabaseManager(str(db_path))
    db_manager.create_tables()

    if not _should_populate(db_manager):
        return {"usage_records": 0, "generation_records": 0}

    rng = random.Random(seed)
    start_date = datetime.now() - timedelta(days=days)

    usage_records = _populate_energy_usage(db_manager, rng, start_date, days)
    generation_records = _populate_solar_generation(db_manager, rng, start_date, days)

    return {
        "usage_records": usage_records,
        "generation_records": generation_records,
    }


def table_counts(db_path: Path = DEFAULT_DB_PATH) -> Tuple[int, int]:
    """Return the current record counts for both tables."""
    db_manager = DatabaseManager(str(db_path))
    session = db_manager.get_session()
    try:
        usage_count = session.query(EnergyUsage).count()
        generation_count = session.query(SolarGeneration).count()
        return usage_count, generation_count
    finally:
        session.close()


if __name__ == "__main__":
    summary = initialize_database()
    usage_count, generation_count = table_counts()
    print(
        "Database initialization complete.\n"
        f"New usage records: {summary['usage_records']}\n"
        f"New generation records: {summary['generation_records']}\n"
        f"Total usage records: {usage_count}\n"
        f"Total generation records: {generation_count}"
    )
