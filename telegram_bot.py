import tempfile
import uuid

import faiss
from pydub import AudioSegment
from telegram.ext import ApplicationBuilder, ContextTypes
import os
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, CallbackContext
from telegram.ext import filters
import requests

from constants.index_classes import class_checkpoint_combinations
from separator import Separator

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
max_file_size_in_mb = 20

wav_mimetypes = [
    "audio/vnd.wav",
    "audio/vnd.wave",
    "audio/wave",
    "audio/x-pn-wav",
    "audio/x-wav",
]

mp3_mimetypes = [
    "audio/mpeg3",
    "audio/x-mpeg-3",
    "audio/mpeg",
    "audio/x-mpeg",
]

supported_mimetypes = wav_mimetypes + mp3_mimetypes

index = faiss.read_index('research/combined.idx')
separator_instance = Separator(index, class_checkpoint_combinations, 0.85)


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Пожалуйста, отправь мне текстовый промпт и аудиозапись в формате wav или mp3.')


async def request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.caption
    audio_file = update.message.document or update.message.audio

    if text is None:
        await update.message.reply_text('Текстовый промпт отсутствует.')
        return 0

    if audio_file is None or audio_file.mime_type not in supported_mimetypes:
        await update.message.reply_text('Неправильный формат файла. Поддерживаются только wav и mp3 файлы.')
        return 0

    if audio_file.file_size / 1024 / 1024 > max_file_size_in_mb:
        await update.message.reply_text('Слишком большой файл, поддерживаются файлы до 20 мегабайт.')
        return 0

    file_info = await audio_file.get_file()

    file_url = file_info.file_path

    tmpdirname = '/tmp/tuned_audio_sep'

    input_filename = f'{uuid.uuid4()}.wav'
    output_filename = f'{uuid.uuid4()}.wav'

    input_file_path = os.path.join(tmpdirname, input_filename)
    output_file_path = os.path.join(tmpdirname, output_filename)

    if not os.path.exists(tmpdirname):
        os.mkdir(tmpdirname)

    response = requests.get(file_url, stream=True)

    if response.status_code == 200:
        with open(input_file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

            if audio_file.mime_type in mp3_mimetypes:
                sound = AudioSegment.from_mp3(input_file_path)
                sound.export(input_file_path, format="wav")
    else:
        await update.message.reply_text('Ошибка при скачивании файла.')


    print(text, input_file_path, output_file_path, input_filename)

    resulting_caption = separator_instance.separate(input_file_path, output_file_path, text.lower())

    await update.message.reply_document(output_file_path, f'Аудиозапись была отделена как {resulting_caption}.')


app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.ATTACHMENT, request))
app.run_polling()
