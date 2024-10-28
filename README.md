# TetrisBot_AI-801
edit "common.py";

choose mode "human_player", "ai_player_training" and "ai_player_watching"

edit "tetromino.py" -> create_pool(cls): -> elif GAME_TYPE == 'extra':

add or delete tetromino.

run "tetris_ai.py".

training may take a significant amount of cpu usage.