# TetrisBot_AI-801
In the application root directory, create a folder for training data in the format 'tetris_regular_cnn_v2_nov_18'
In this directory create three additional directories 'checkpoints_dqn', 'dataset', and 'whole_model'

edit "common.py" to include this filepath;

choose mode "human_player", "ai_player_training" and "ai_player_watching"

edit "tetromino.py" -> create_pool(cls): -> elif GAME_TYPE == 'extra':

add or delete tetromino.

run "tetris_ai.py".

training may take a significant amount of cpu usage.


NOTE:
nov_13 - base model tested
nov_24 - base model with GPU acceleration
nov_25 - hyperparameter tuning added to incorporate encouraging exploration on early steps. Using coefficient plan and adjusted coefficients
nov_26.1 - Added loss function based on average score instead of MSE. Refined coefficient numbers to what seemed best on last test.
nov_26.2 - Added loss threshold and increased inner loops(epochs). Added lower exploration rate of .95