import os
import sys
import base64
import tempfile
import subprocess
import traceback
import time
from typing import Optional

import runpod
from huggingface_hub import snapshot_download

# ----- Config -----
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
MODEL_DIR = os.environ.get("ACESTEP_MODEL_DIR", "/app/models")
DIT_CONFIG = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "vllm")
DEVICE = os.environ.get("ACESTEP_DEVICE", "cuda")
DEFAULT_AUDIO_FMT = os.environ.get("ACESTEP_AUDIO_FORMAT", "mp3")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

GENRE_PRESETS = {
    "pop": "upbeat pop music with catchy melody, electric guitars, and synth pads",
    "rock": "energetic rock music with distorted guitars, heavy drums, and driving bass",
    "electronic": "electronic dance music with heavy bass, synth leads, and four-on-the-floor beats",
    "classical": "orchestral classical music with strings, woodwinds, and brass",
    "jazz": "smooth jazz with saxophone, piano, upright bass, and brushed drums",
    "ambient": "ambient atmospheric music with pads, textures, and minimal percussion",
    "lofi": "lo-fi hip hop with mellow piano, soft drums, and vinyl crackle",
    "cinematic": "cinematic orchestral music with epic strings, brass, and percussion",
    "rnb": "R&B with soulful vocals, smooth bass, and groove drums",
    "hiphop": "hip hop with heavy 808 bass, trap drums, and melodic synth",
    "folk": "warm acoustic folk with fingerpicked guitar and gentle vocals",
    "metal": "heavy metal with aggressive guitars, double kick drums, and growling vocals",
    "country": "country music with acoustic guitar, fiddle, and steady drums",
    "blues": "blues with gritty guitar, shuffling drums, and harmonica",
}

_dit_handler = None
_llm_handler = None
_models_loaded = False


# ============================================================
# Startup: download models (once per cold start)
# ============================================================
def _ensure_models():
    """Download DiT + LM weights on first cold start."""
    dit_dest = os.path.join(MODEL_DIR, DIT_CONFIG)
    lm_dest = os.path.join(MODEL_DIR, LM_MODEL)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if already downloaded
    dit_ready = os.path.isfile(os.path.join(dit_dest, "model.safetensors")) or \
                os.path.isfile(os.path.join(dit_dest, "pytorch_model.bin"))
    lm_ready = os.path.isfile(os.path.join(lm_dest, "model.safetensors")) or \
               os.path.isfile(os.path.join(lm_dest, "pytorch_model.bin"))

    if dit_ready and lm_ready:
        print(f"[startup] Models already cached at {MODEL_DIR}, skipping download.", flush=True)
        return

    print(f"[startup] Downloading models to {MODEL_DIR}", flush=True)

    if not dit_ready:
        print(f"[startup]   DiT: ACE-Step/{DIT_CONFIG}", flush=True)
        t0 = time.time()
        snapshot_download(
            repo_id=f"ACE-Step/{DIT_CONFIG}",
            local_dir=dit_dest,
            max_workers=16,
            resume_download=True,
            cache_dir=os.environ.get("HF_HOME", "/app/hf_cache"),
            token=HF_TOKEN,
        )
        print(f"[startup]   DiT done in {time.time()-t0:.1f}s", flush=True)

    if not lm_ready:
        print(f"[startup]   LM: ACE-Step/{LM_MODEL}", flush=True)
        t0 = time.time()
        snapshot_download(
            repo_id=f"ACE-Step/{LM_MODEL}",
            local_dir=lm_dest,
            max_workers=16,
            resume_download=True,
            cache_dir=os.environ.get("HF_HOME", "/app/hf_cache"),
            token=HF_TOKEN,
        )
        print(f"[startup]   LM done in {time.time()-t0:.1f}s", flush=True)

    print("[startup] All models ready.", flush=True)


def _load_models():
    """Initialize ACE-Step DiT + LM handlers. Called once per worker."""
    global _dit_handler, _llm_handler, _models_loaded
    if _models_loaded:
        return

    _ensure_models()

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    print(f"[ACE-Step] Loading DiT: {DIT_CONFIG} on {DEVICE}", flush=True)
    _dit_handler = AceStepHandler()
    _dit_handler.initialize_service(project_root=PROJECT_ROOT, config_path=DIT_CONFIG, device=DEVICE)
    print("[ACE-Step] DiT ready", flush=True)

    print(f"[ACE-Step] Loading LM: {LM_MODEL}", flush=True)
    _llm_handler = LLMHandler()
    _llm_handler.initialize(checkpoint_dir=MODEL_DIR, lm_model_path=LM_MODEL, backend=LM_BACKEND, device=DEVICE)
    print("[ACE-Step] LM ready — worker online", flush=True)
    _models_loaded = True


def _encode_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _wav_to_mp3(wav_path, mp3_path):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
            check=True, capture_output=True
        )
        return True
    except Exception:
        return False


def _upload_to_s3(local_path, s3_key):
    """Upload file to S3. Returns presigned URL or None."""
    try:
        import boto3
        s3 = boto3.client("s3", region_name=S3_REGION)
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600,
        )
        return url
    except Exception as e:
        print(f"[s3] Upload failed: {e}", file=sys.stderr, flush=True)
        return None


def handler(job):
    """RunPod serverless entry point."""
    try:
        _load_models()
    except Exception as exc:
        return {"error": f"Model init failed: {exc!s}", "traceback": traceback.format_exc()}

    raw = job.get("input", {})
    prompt = raw.get("prompt", "")
    lyrics = raw.get("lyrics", "")
    duration = float(raw.get("duration", 30))
    genre_preset = raw.get("genre_preset", "")
    bpm = raw.get("bpm")
    key = raw.get("key", "")
    instrumental = bool(raw.get("instrumental", False))
    seed = int(raw.get("seed", -1))
    audio_format = raw.get("audio_format", DEFAULT_AUDIO_FMT)
    upload_s3 = bool(raw.get("upload_s3", False)) and bool(S3_BUCKET)

    # Build caption
    if genre_preset and genre_preset.lower() in GENRE_PRESETS:
        caption = GENRE_PRESETS[genre_preset.lower()]
        if prompt:
            caption = f"{caption}, {prompt}"
    else:
        caption = prompt or "instrumental background music"

    duration = max(10.0, min(600.0, duration))
    is_turbo = "turbo" in DIT_CONFIG

    try:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            caption=caption,
            lyrics=lyrics if not instrumental else "",
            instrumental=instrumental,
            duration=duration,
            seed=seed,
            shift=3.0 if is_turbo else 1.0,
        )
        if bpm is not None:
            params.bpm = int(bpm)
        if key:
            params.keyscale = key

        config = GenerationConfig(batch_size=1, audio_format="wav")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_music(_dit_handler, _llm_handler, params, config, save_dir=tmp_dir)

            if not result.success:
                return {"error": result.error or "Generation failed"}
            if not result.audios:
                return {"error": "No audio generated"}

            audio = result.audios[0]
            wav_path = audio["path"]
            actual_seed = audio.get("params", {}).get("seed", seed)
            sample_rate = audio.get("sample_rate", 48000)

            # Convert to mp3 if requested
            final_path = wav_path
            if audio_format == "mp3":
                mp3_path = os.path.join(tmp_dir, "output.mp3")
                if _wav_to_mp3(wav_path, mp3_path):
                    final_path = mp3_path
                else:
                    audio_format = "wav"

            # Upload to S3 if requested
            if upload_s3:
                s3_key = f"acestep/{int(time.time())}_{seed}.{audio_format}"
                audio_url = _upload_to_s3(final_path, s3_key)
                if audio_url:
                    return {
                        "audio_url": audio_url,
                        "audio_format": audio_format,
                        "sample_rate": sample_rate,
                        "duration_sec": duration,
                        "seed": actual_seed,
                        "caption": caption,
                    }

            # Default: return base64
            audio_b64 = _encode_file(final_path)
            return {
                "audio_base64": audio_b64,
                "audio_format": audio_format,
                "sample_rate": sample_rate,
                "duration_sec": duration,
                "seed": actual_seed,
                "caption": caption,
            }
    except Exception as exc:
        return {"error": f"Generation error: {exc!s}", "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
