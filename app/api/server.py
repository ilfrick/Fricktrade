from fastapi import FastAPI

from app.utils.config import load_config


app = FastAPI(title="Autotrader API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    cfg = load_config("/app/config/config.yaml")
    cfg["brokers"]["alpaca"]["api_key"] = "***"
    cfg["brokers"]["alpaca"]["api_secret"] = "***"
    return cfg
