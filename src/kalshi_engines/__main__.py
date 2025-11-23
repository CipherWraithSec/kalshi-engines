import argparse
import logging
from pathlib import Path

from kalshi_engines.agents.mention import MentionAgent
from kalshi_engines.agents.nfl import NFLAgent
from kalshi_engines.agents.weather_oil import WeatherOilAgent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalshi Quant Engines – Omniverse Take-Home")
    parser.add_argument(
        "--engine", choices=["mention", "nfl", "weather-oil"], required=True)
    parser.add_argument("--demo", action="store_true",
                        help="Run on sample data")
    args = parser.parse_args()

    if args.engine == "mention":
        agent = MentionAgent(demo=args.demo)
    elif args.engine == "nfl":
        agent = NFLAgent(demo=args.demo)
    else:
        agent = WeatherOilAgent(demo=args.demo)

    result = agent.run()
    print("\n=== FINAL SIGNAL ===")
    print(result)


if __name__ == "__main__":
    main()
