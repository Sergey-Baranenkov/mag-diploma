musdb_ckpt = 'checkpoints/train_audiosep_lora_and_tuned_embeddings/audiosep_lora_and_tuned_embeddings_musdb18,' \
             'args=logs_per_class=True, dropout=0.1,timestamp=1713021870.315059/epoch=19.ckpt'
desed_ckpt = 'checkpoints/train_audiosep_lora_and_tuned_embeddings/audiosep_lora_and_tuned_embeddings_desed,' \
             'args=logs_per_class=True, dropout=0.1,timestamp=1713043618.1325147/epoch=9.ckpt'

class_checkpoint_combinations = [
    ('vacuum cleaner', desed_ckpt),
    ('frying', desed_ckpt),
    ('dishes', desed_ckpt),
    ('blender', desed_ckpt),
    ('speech', desed_ckpt),
    ('dog', desed_ckpt),
    ('alarm bell ringing', desed_ckpt),
    ('running water', desed_ckpt),
    ('electric shaver toothbrush', desed_ckpt),
    ('cat', desed_ckpt),

    ('drums', musdb_ckpt),
    ('bass', musdb_ckpt),
    ('vocal', musdb_ckpt),
]
