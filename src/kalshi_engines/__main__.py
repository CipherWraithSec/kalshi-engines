
import argparse
import logging
from kalshi_engines.agents.mention import MentionAgent
from kalshi_engines.agents.nfl import NFLAgent
from kalshi_engines.agents.weather_oil import WeatherOilAgent

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Omniverse Fund – Kalshi Engines")
    parser.add_argument(
        "--engine", choices=["mention", "nfl", "weather-oil"], required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    log.info(f"Starting {args.engine} engine (demo={args.demo})")

    if args.engine == "mention":
        result = MentionAgent(demo=args.demo).run()
    elif args.engine == "nfl":
        result = NFLAgent(demo=args.demo).run()
    else:
        result = WeatherOilAgent(demo=args.demo).run()

    log.info("Run complete")
    print("\nFINAL SIGNAL:")
    for k, v in result.items():
        print(f"  {k:20} : {v}")


if __name__ == "__main__":
    main()
