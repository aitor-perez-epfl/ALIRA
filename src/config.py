from pathlib import Path
from dotenv import dotenv_values

config = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
