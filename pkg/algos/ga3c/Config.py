
class Config:

    #########################################################################
    # Game configuration
    
    PLAY_MODE = False

    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = False
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0 

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNT = 0.99

    # Max size of the queue
    MAX_QUEUE_SIZE = 100

    # Total number of episodes and annealing frequency
    EPISODES = 400000
    ANNEALING_EPISODE_COUNT = 400000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.0001
    BETA_END = 0.0001

    # Learning rate
    LEARNING_RATE_START = 0.00025
    LEARNING_RATE_END = 0.00025

    # Optimizer (Adam or RMSProp)
    OPTIMIZER = 'Adam'

    # AdamOptimizer parameters
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPSILON = 1e-8

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1
    
    # Gradient clipping
    USE_GRAD_CLIP = True
    GRAD_CLIP_NORM = 40.0 

    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = False
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1000
    
    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 50
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 100

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here
    
    # Minimum policy
    MIN_POLICY = 0.01

    # Number of Trainers
    TRAINERS = 1
    # Number of Predictors
    PREDICTORS = 1

    # Master's device
    MASTER_DEVICE = '/gpu:0' 
    
    # train batch size
    TRAINING_MIN_BATCH_SIZE = 128
    # predict batch size
    PREDICTION_MAX_BATCH_SIZE = 8
    # policy lag in GA3C
    LOG_EPSILON = 1e-6
    # test step 
    TEST_STEP = 100
