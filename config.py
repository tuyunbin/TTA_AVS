import argparse
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type', type=str, choices=['seq2seq', 'TTA', 'TTA_AVS','AVS'], default='TTA_AVS')
net_arg.add_argument('--dropout', type=float, default=0.5)
net_arg.add_argument('--weight_init', type=float, default=None)
net_arg.add_argument('--cell_type', type=str, default='lstm', choices=['lstm','gru'])
net_arg.add_argument('--birnn', type=str2bool, default=True)
net_arg.add_argument('--embed', type=int, default=512) # MSVD: 468 # MSR:512
net_arg.add_argument('--hid', type=int, default=768) # MSVD #256,MSR:768;
net_arg.add_argument('--num_layers', type=int, default=1)
net_arg.add_argument('--vid_dim', type=int, default=3584) # con: 3584
net_arg.add_argument('--encoder_rnn_max_length', type=int, default=28)
net_arg.add_argument('--decoder_rnn_max_length', type=int, default=30)
net_arg.add_argument('--tag_max_length', type=int, default=25) # MSVD: 20 MSR:25
net_arg.add_argument('--max_vocab_size', type=int, default=25000)
net_arg.add_argument('--msvd_max_vocab_size', type=int, default=15000) # 15000
net_arg.add_argument('--max_snli_vocab_size', type=int, default=36000)
net_arg.add_argument('--beam_size', type=int, default=1)
net_arg.add_argument('--gcn', type=str2bool, default=False)
net_arg.add_argument('--sat', type=str2bool, default=False)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, choices=['msvd','msrvtt'],default='msvd')
data_arg.add_argument('--vid_feature_path', type=str, default= "data/msrvtt16/msrvtt_con.hdf5")#'data/msrvtt16/MSR-VTT_InceptionV4.hdf5')
data_arg.add_argument('--captions_path', type=str, default='data/msrvtt16/CAP_with_POS.pkl')
data_arg.add_argument('--vocab_file', type=str, default='data/msrvtt16/vocab.txt')
data_arg.add_argument('--tag_path', type=str, default='data/msrvtt16/sem_object.pkl')
data_arg.add_argument('--snli_vocab_file', type=str, default='data/msrvtt16/vocab_snli.txt')
data_arg.add_argument('--vg_path', type=str, default='data/msrvtt16/vg_list')
data_arg.add_argument('--pos_file', type=str, default='data/msrvtt16/msr_pos.pkl')
# msvd data
data_arg.add_argument('--msvd_vid_feature_path', type=str, default= "data/msvd/msvd_con.hdf5")#"data/msvd/msvd_inceptionv4.hdf5"
data_arg.add_argument('--msvd_captions_path', type=str, default='data/msvd/CAP_with_POS.pkl')
data_arg.add_argument('--msvd_tag_path', type=str, default='data/msvd/sem_object.pkl')
data_arg.add_argument('--msvd_vg_path', type=str, default='data/msvd/vg_list')
data_arg.add_argument('--msvd_vocab_file', type=str, default='data/msvd/worddict.pkl')
data_arg.add_argument('--msvd_pos_file', type=str, default='data/msvd/msvd_pos.pkl')
# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'])
learn_arg.add_argument('--batch_size', type=int, default=64)
learn_arg.add_argument('--gamma_ml_rl', type=float, default=0.9981)
learn_arg.add_argument('--loss_function', type=str, default='xe', choices=['xe','rl', 'xe+rl'])
learn_arg.add_argument('--max_epoch', type=int, default=5)
learn_arg.add_argument('--max_epoch_stage2', type=int, default=2) # MSR=2 is better
learn_arg.add_argument('--reward_type', type=str, default='CIDEr')
learn_arg.add_argument('--grad_clip', type=float, default=10.0)
learn_arg.add_argument('--optim', type=str, default='adam')
learn_arg.add_argument('--lr', type=float, default=2e-4) # msr:1e-4 # msvd:2e-4
learn_arg.add_argument('--lr2', type=float, default=5e-5) # msvd:5e-5 # MSR: 1e-5
learn_arg.add_argument('--lr3', type=float, default=5e-5) # msvd:5e-5 MSR:9e-6
learn_arg.add_argument('--use_decay_lr', type=str2bool, default=False)
learn_arg.add_argument('--decay', type=float, default=0.96)
learn_arg.add_argument('--lambda_threshold', type=float, default=0.5)
learn_arg.add_argument('--beta_threshold', type=float, default=0.333)
learn_arg.add_argument('--alpha_lambda', type=float, default=0) #seq2seq=0
learn_arg.add_argument('--c_lambda', type=float, default=1.0)
learn_arg.add_argument('--arr_lambda', type=float, default=0.2) # msvd 0.5 #MSR: 0.3
learn_arg.add_argument('--acr_lambda', type=float, default=0.4) # msvd 0.3, con 0.4 # MSR:0.1
learn_arg.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
learn_arg.add_argument('--learning_rate_decay_every', type=int, default=2,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
learn_arg.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')

learn_arg.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
learn_arg.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
learn_arg.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
learn_arg.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')



# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_name', type=str, default='')
# misc_arg.add_argument('--load_path', type=str, default="/home/zhouc/tyb/caption/video_captioning_rl/graph_new/msrvtt_base_con/model_epoch1_step4072.pth")
# misc_arg.add_argument('--load_path', type=str, default="/home/zhouc/tyb/caption/video_captioning_rl/graph+/msvd_con_base/model_epoch1_step1526.pth")# CONCA
misc_arg.add_argument('--load_path', type=str, default='')
# misc_arg.add_argument('--load_path', type=str, default="/home/zhouc/tyb/caption/video_captioning_rl/graph/msvd_base/model_epoch2_step2289.pth")
# misc_arg.add_argument('--load_path', type=str, default="/home/zhouc/tyb/caption/video_captioning_rl/graph_new/msvd_base1e/model_epoch1_step1526.pth")
# misc_arg.add_argument('--load_path', type=str, default="/home/zhouc/tyb/caption/video_captioning_rl/graph/msrvtt_base1e/model_epoch3_step8144.pth")
misc_arg.add_argument('--result_path', type=str, default='results')
misc_arg.add_argument('--load_entailment_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=1)
misc_arg.add_argument('--save_criteria', type=str, default='AVG', choices=['CIDEr', 'AVG'])
misc_arg.add_argument('--max_save_num', type=int, default=2)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='experiments_upload_test')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=1111)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)


def get_args():
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed
