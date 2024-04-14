from telegram.ext import ApplicationBuilder, ContextTypes
import os
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, CallbackContext
from telegram.ext import filters

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

wav_mimetypes = [
    "audio/vnd.wav",
    "Preferred",
    "audio/vnd.wave",
    "audio/wave",
    "audio/x-pn-wav",
    "audio/x-wav",
]


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Пожалуйста, отправь мне текстовый промпт и аудиозапись в формате wav.')


async def request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.caption
    audio_file = update.message.document
    if text is None:
        await update.message.reply_text('Текстовый промпт отсутствует.')
        return 0

    if audio_file is None or audio_file.mime_type not in wav_mimetypes:
        await update.message.reply_text('Неправильный формат файла. Поддерживаются только wav файлы.')
        return 0

    print(text, audio_file)


app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.ATTACHMENT, request))

app.run_polling()
