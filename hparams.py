from text import symbols


class hparams:
    seed = 1234

    ################################
    # Data Parameters              #
    ################################
    text_cleaners=['basic_cleaners']

    ################################
    # Audio                        #
    ################################
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_shift = 256
    frame_length = 1024
    fmin = 0
    fmax = 8000
    power = 1.5
    gl_iters = 30

    ################################
    # Train                        #
    ################################
    is_cuda = True
    pin_mem = True
    n_workers = 1
    prep = True
    #pth = 'lj-22k.pkl'
    pth = None
    lr = 2e-3
    #lr = 2.9605e-4
    betas = (0.9, 0.999)
    eps = 1e-6
    sch = True
    sch_step = 4000
    #max_iter = 200e3
    max_iter = 20000
    batch_size = 32
    iters_per_log = 10
    iters_per_sample = 500
    iters_per_ckpt = 10000
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = ['s a m ɨ j ▁ u m n ɨ j ▁ t͡ɕ ɪ ɫ ɐ vʲ e k ▁ n ə ▁ p ɫ ɐ nʲ e tʲ e ▁ i ▁ s a m ɨ j ▁ k r ɐ sʲ i v ɨ j']

    ################################
    # Model Parameters             #
    ################################
    n_symbols = len(symbols)
    symbols_embedding_dim = 512

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_ratio = 10
    gate_threshold = 0.2
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 128

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

