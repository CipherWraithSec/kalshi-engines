from kalshi_engines.agents.mention import MentionAgent
from kalshi_engines.agents.nfl import NFLAgent
from kalshi_engines.agents.weather_oil import WeatherOilAgent

if __name__ == "__main__":
    MentionAgent()._load_or_train()
    NFLAgent()._load_or_train()
    WeatherOilAgent()._load_or_train()
    print("All models trained and saved.")
