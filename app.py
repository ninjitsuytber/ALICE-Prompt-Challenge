from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.challenges import get_challenge, shuffle_challenge_ids
from services import database as db
from services.image_gen import (
    GENERATED_DIR,
    generate_image_from_prompt,
    get_engine_status,
    save_runtime_keys,
    warmup_engine,
)
from services.kiosk import (
    get_kiosk_status,
    launch_kiosk,
    schedule_kiosk_exit,
    verify_secret_phrase,
)
from services.similarity import compare_images, warmup_similarity

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 128
    await asyncio.to_thread(warmup_similarity)
    await asyncio.to_thread(warmup_engine)
    yield


app = FastAPI(title="ALICE Prompt Challenge", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")


class StartRoundRequest(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=40)


class PickSlotRequest(BaseModel):
    slot_index: int = Field(..., ge=1, le=10)


class SubmitPromptRequest(BaseModel):
    prompt: str = Field(default="", max_length=800)
    auto_submit: bool = False


class EngineConfigRequest(BaseModel):
    hf_token: str | None = None
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None


class KioskEnterRequest(BaseModel):
    origin: str | None = None


class KioskExitRequest(BaseModel):
    phrase: str = Field(default="", max_length=128)


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": get_engine_status()}


@app.post("/api/round/start")
async def start_round(payload: StartRoundRequest):
    player_name = " ".join(payload.player_name.split())
    if not player_name:
        raise HTTPException(status_code=400, detail="Player name cannot be empty.")

    shuffled = shuffle_challenge_ids()
    round_info = db.create_round(player_name=player_name, shuffled_ids=shuffled)
    return {
        "round_id": round_info["round_id"],
        "player_name": round_info["player_name"],
        "status": round_info["status"],
        "slots": list(range(1, 11)),
    }


@app.post("/api/round/{round_id}/reshuffle")
async def reshuffle_round(round_id: str):
    round_data = db.get_round(round_id)
    if not round_data:
        raise HTTPException(status_code=404, detail="Round not found.")
    if round_data["status"] != "picking" or round_data["picked_slot"] is not None:
        raise HTTPException(status_code=400, detail="Cannot reshuffle after picking a card.")

    ok = db.reshuffle_round(round_id, shuffle_challenge_ids())
    if not ok:
        raise HTTPException(status_code=400, detail="Could not reshuffle round.")
    return {"round_id": round_id, "reshuffled": True, "slots": list(range(1, 11))}


@app.post("/api/round/{round_id}/pick")
async def pick_blind_slot(round_id: str, payload: PickSlotRequest):
    result = db.pick_slot(round_id, payload.slot_index)
    if not result:
        raise HTTPException(status_code=404, detail="Round not found.")

    challenge = get_challenge(result["picked_challenge_id"])
    return {
        "round_id": result["round_id"],
        "player_name": result["player_name"],
        "picked_slot": result["picked_slot"],
        "challenge_code": challenge.code,
        "reference_image_url": challenge.public_url,
        "status": result["status"],
    }


@app.post("/api/round/{round_id}/submit")
async def submit_prompt(round_id: str, payload: SubmitPromptRequest):
    prompt_text = " ".join(payload.prompt.split())
    if not payload.auto_submit and len(prompt_text) < 3:
        raise HTTPException(status_code=400, detail="Please enter a descriptive prompt (at least 3 characters).")

    round_data = db.get_round(round_id)
    if not round_data:
        raise HTTPException(status_code=404, detail="Round not found.")
    if not round_data["picked_challenge_id"] or not round_data["picked_slot"]:
        raise HTTPException(status_code=400, detail="Pick one of the 10 hidden cards before submitting.")

    challenge = get_challenge(round_data["picked_challenge_id"])
    gen_prompt = prompt_text if prompt_text else "empty blank white canvas"
    saved_prompt = prompt_text if prompt_text else "(No prompt submitted)"

    gen_result = await asyncio.to_thread(generate_image_from_prompt, gen_prompt, round_id)
    sim_result = await asyncio.to_thread(compare_images, challenge.file_path, gen_result.file_path)

    entry = db.save_submission(
        round_id=round_id,
        player_name=round_data["player_name"],
        challenge_id=challenge.id,
        challenge_code=challenge.code,
        picked_slot=int(round_data["picked_slot"]),
        prompt_text=saved_prompt,
        generated_image_url=gen_result.public_url,
        similarity_score=sim_result.overall,
        semantic_score=sim_result.semantic,
        style_score=sim_result.style,
        color_score=sim_result.color,
        composition_score=sim_result.composition,
        generator_source=gen_result.source,
    )

    return {
        "round_id": round_id,
        "player_name": entry["player_name"],
        "picked_slot": entry["picked_slot"],
        "challenge_code": entry["challenge_code"],
        "reference_image_url": challenge.public_url,
        "generated_image_url": entry["generated_image_url"],
        "prompt_text": entry["prompt_text"],
        "similarity": {
            "overall": entry["similarity_score"],
            "semantic": entry["semantic_score"],
            "style": entry["style_score"],
            "color": entry["color_score"],
            "composition": entry["composition_score"],
        },
        "generator_source": entry["generator_source"],
        "fallback_reason": gen_result.fallback_reason,
        "rank": entry["rank"],
        "total_entries": entry["total_entries"],
        "created_at": entry["created_at"],
    }


@app.get("/api/leaderboard")
async def get_leaderboard(limit: int | None = Query(default=None, ge=1, le=10000)):
    entries = db.list_leaderboard(limit=limit)
    total = len(entries) if limit is None else db.count_leaderboard()
    return {"entries": entries, "total": total}


@app.post("/api/leaderboard/reset")
async def reset_leaderboard():
    deleted = db.clear_leaderboard()
    return {"deleted": deleted}


@app.get("/api/config/engine")
async def engine_config():
    return get_engine_status()


@app.post("/api/config/engine")
async def update_engine_config(payload: EngineConfigRequest):
    return save_runtime_keys(
        hf_token=payload.hf_token,
        cloudflare_account_id=payload.cloudflare_account_id,
        cloudflare_api_token=payload.cloudflare_api_token,
        gemini_key=payload.gemini_api_key,
        openai_key=payload.openai_api_key,
    )


@app.get("/api/kiosk/status")
async def kiosk_status():
    return get_kiosk_status()


@app.post("/api/kiosk/enter")
async def kiosk_enter(payload: KioskEnterRequest | None = None):
    origin = payload.origin if payload else None
    try:
        return await asyncio.to_thread(launch_kiosk, origin)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/kiosk/exit")
async def kiosk_exit(payload: KioskExitRequest):
    if not verify_secret_phrase(payload.phrase):
        raise HTTPException(status_code=403, detail="Invalid secret phrase.")
    return await asyncio.to_thread(schedule_kiosk_exit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
