"""Runtime configuration and logging setup."""
import logging

# Set up clean logging for both local dev and Docker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

log = logging.getLogger(__name__)
