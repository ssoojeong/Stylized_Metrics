from option.config import Config

def config_all():
    config = Config({
    # device
    "GPU_ID": "0",
    
    # model for PIPAL (NTIRE2021 Challenge)
    "n_enc_seq": 21*21,                 # feature map dimension (H x W) from backbone, this size is related to crop_size
    "n_dec_seq": 21*21,                 # feature map dimension (H x W) from backbone
    "n_layer": 1,                       # number of encoder/decoder layers
    "d_hidn": 128,                      # input channel (C) of encoder / decoder (input: C x N)
    "i_pad": 0,
    "d_ff": 1024,                       # feed forward hidden layer dimension
    "d_MLP_head": 128,                  # hidden layer of final MLP 
    "n_head": 4,                        # number of head (in multi-head attention)
    "d_head": 128,                      # input channel (C) of each head (input: C x N) -> same as d_hidn
    "dropout": 0.1,                     # dropout ratio of transformer
    "emb_dropout": 0.1,                 # dropout ratio of input embedding
    "layer_norm_epsilon": 1e-12,
    "n_output": 1,                      # dimension of final prediction
    "crop_size": 192,                   # input image crop size

    # data
    "ori_path":"./data/test/org",
    "exp_path":"./data/test/gen",
    'save_path':'./results/',
    "weight_file": "./weights/PIPAL/epoch40.pth", # "./weights/epoch240.pth",
    "result_file": "allinone.txt",

    # ensemble in test
    "test_ensemble": True,
    "n_ensemble": 20,
    
    # FID
    "save_stats": False,
    "dims":2048,
    "batch_size":1
    })
    return config
