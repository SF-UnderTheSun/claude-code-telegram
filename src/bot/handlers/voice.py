"""Voice message handler — transcribes audio via Groq Whisper and routes to Claude."""

import os
import tempfile
from typing import Optional

import structlog
from telegram import Update
from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def handle_voice_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Transcribe voice/audio message via Groq Whisper, then forward as text to Claude."""
    message = update.effective_message
    if not message:
        return

    # Support both voice notes and audio files
    voice = message.voice or message.audio
    if not voice:
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        await message.reply_text(
            "Voice input not available: GROQ_API_KEY is not configured."
        )
        return

    await message.chat.send_action("typing")
    status_msg = await message.reply_text("Transcribing audio...")

    try:
        # Download audio from Telegram
        tg_file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name

        await tg_file.download_to_drive(tmp_path)

        # Transcribe via Groq Whisper
        from groq import Groq

        client = Groq(api_key=api_key)
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text",
            )

        transcript = str(transcription).strip()
        logger.info(
            "Voice transcription completed",
            user_id=update.effective_user.id if update.effective_user else None,
            length=len(transcript),
        )

        await status_msg.delete()

        if not transcript:
            await message.reply_text("Could not transcribe audio — no speech detected.")
            return

        # Echo transcript so the user sees what was understood
        await message.reply_text(f"Voice: {transcript}")

        # Route transcript to normal Claude handler via context override
        context.user_data["_voice_override"] = transcript
        try:
            if context.bot_data.get("settings") and getattr(
                context.bot_data["settings"], "agentic_mode", False
            ):
                orchestrator = context.bot_data.get("_orchestrator")
                if orchestrator:
                    await orchestrator.agentic_text(update, context)
                else:
                    from .message import handle_text_message
                    await handle_text_message(update, context)
            else:
                from .message import handle_text_message
                await handle_text_message(update, context)
        finally:
            context.user_data.pop("_voice_override", None)

    except Exception as exc:
        logger.error("Voice transcription failed", error=str(exc))
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.reply_text(f"Voice transcription failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
