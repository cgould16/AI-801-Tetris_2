from multiprocessing import cpu_count
GAME_BOARD_WIDTH = 10
GAME_BOARD_HEIGHT = 20

STATE_INPUT = 'short'       # 'short', 'long', 'dense'
SEARCH = True
T_SPIN_MARK = True
OUTER_MAX = 50
CPU_MAX = int(cpu_count()*.75)                # num of cpu used to collect samples = min(multiprocessing.cpu_count(), CPU_MAX)

#   1.  choose what kind of Tetris you'd like to play.
#       If 'extra', it's custom Tetris. Go to tetromino.py and search "elif GAME_TYPE == 'extra'" to edit pieces.
# GAME_TYPE = 'extra'
GAME_TYPE = 'regular'


#   2.  folder name to store dataset and model. './anything_you_like/'
# FOLDER_NAME = './tetris_extra/'
FOLDER_NAME = "/Users/joshrackham/AI-801-Tetris_2/tetris_regular_cnn_v1_nov_24/"

#   3.  if > 0, then model {FOLDER_NAME}/whole_model/outer_{OUT_START} will be loaded to continue training or watch it play
#       if 0, then create a brand-new model.
OUT_START = 0

#   4.  choose the mode
# MODE = 'human_player'
MODE = 'ai_player_training'
# MODE = 'ai_player_watching'

#   5.  run tetris_ai.py
