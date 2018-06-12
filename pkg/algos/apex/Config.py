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
    PREDICTION_BATCH_SIZE = 128

    # Total number of episodes and annealing frequency
    EPISODES = 400000
    ANNEALING_EPISODE_COUNT = 400000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.0001
    BETA_END = 0.0001

    # Learning rate
    LEARNING_RATE_START = 0.00025
    LEARNING_RATE_END = 0.00025
    #LEARNING_RATE_START = 0.005
    #LEARNING_RATE_END = 0.005

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
    TRAINERS = 2
    # Number of replay buffers
    BUFFERS = 4

    # DEVICE
    DEVICE = '/cpu:0' 
    # MASTER_DEVICE
    MASTER_DEVICE = '/gpu:0'
    # Worker's data size
    WORKER_DATA_SIZE = 32

    # whether use prioritized replay
    USE_PRIORITY = True
    P_ALPHA = 0.6
    P_BETA_BASE = 0.5
    P_MAX_BETA = 1.0
    P_ANNEALED_STEP = 1e6
    #
    TRAINING_BATCH_SIZE = 256
    #
    MIN_BUFFER_SIZE = 1e4
    MAX_BUFFER_SIZE = 2.5*1e5
    #
    TEST_STEP = 1000
    #
    GREEDY_ANNEALING_STEP = 1000
    # 
    STATS_SHOW_STEP = 100
    # 
    PRIORITY_EPSILON = 1e-8
