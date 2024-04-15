# Заточенные под fine-tuning модели на основе AudioSep 

Ссылка на репозиторий с AudioSep https://github.com/Audio-AGI/AudioSep

Статья с описанием работы AudioSep ["Separate Anything You Describe"](https://audio-agi.github.io/Separate-Anything-You-Describe/AudioSep_arXiv.pdf).

По сравнению с оригинальным репозиторием добавлена директория research с проверкой различных гипотез и валидации моделей.

Написан кастомный waveform mixer, обрабатывающий проблему нескольких одинаковых классов в песне одновременно.

Для кастомных моделей описан validation step, соответственно, при тренировке есть возможность отслеживать результат валидации.
Метрики записываются в weight and biases.

Для тренировки сети необходимо скачать чекпоинты `audiosep_base_4M_steps.ckpt` и `music_speech_audioset_epoch_15_esc_89.98.pt` 
и положить в папку chekpoint (предварительно ее создав). Конфигурационные файлы модели лежат в папке config.
Для того чтобы тюнить сеть на своем датасете необходимо создать отдельный конфигурационный файл, указав путь до json файла 
с caption и path до файла (примеры можно найти в папке datafiles). Для тренировки соответствующей модели нужно использовать
один из следующих файлов 
1. [train_audiosep_lora.py](train_audiosep_lora.py)
2. [train_audiosep_tuned_embeddings.py](train_audiosep_tuned_embeddings.py)
3. [train_audiosep_lora_and_tuned_embeddings.py](train_audiosep_lora_and_tuned_embeddings.py)

Запустить процесс тренировки можно с помощью команды 
`python train_audiosep_lora_and_tuned_embeddings.py --workspace ./ --config_yaml config/audiosep_lora_and_tuned_embeddings_desed.yaml --resume_checkpoint_path ''`

Чекпоинты с весами хранят информацию о текущей эпохе, поэтому есть возможность продолжать тренировку с определенного чекпоинта.

Также расширен `benchmark.py`, в него добавлена evaluation для датасета musdb18 и есть возможность выбирать модель. 
Запустить бенчмарк можно с помощью команды `python benchmark.py --model_name audiosep_lora_and_embeddings --datasets_to_test musdb18 --checkpoint_path "checkpoints/train_audiosep_lora_and_tuned_embeddings/audiosep_lora_and_tuned_embeddings_musdb18,args=logs_per_class=True, dropout=0.1,timestamp=1712871001.7698753/epoch=19.ckpt"`

Для сохранения своего faiss индекса необходимо заполнить мапу `constants/index_classes.py` и запустить `research/create_index.faiss.ipynb`

Запустить телеграм бота можно командой `export TELEGRAM_BOT_TOKEN='' && python telegram_bot.py`.
Бот будет выбирать необходимый чекпоинт в зависимости от промпта пользователя. 

Важно правильно настроить `distance_threshold`, тем самым регулировать количество false positive и false negative

Также важно понимать, что если бот находит подходящий fine-tuned index, в качестве промпта он использует
caption ближайшего класса, а не промпт пользователя, так как был настроен именно на нем. 
При mismatch между пользовательским промптом и caption класса это может сильно ухудшить 
качество аудиозаписи на выходе.