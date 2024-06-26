import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    parser.add_argument('--num_dataset_train', type=int, default=4, help='quantity of train datasets')
    # training settings
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers')
    parser.add_argument('--img_size', type=int, default=256, help='img size')
    parser.add_argument('--protocol', type=str, default="all", help='protocal')
    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='base learning rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=120, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--step_size', type=int, default=50, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--trans', type=str, default="p", help="different pre-process")
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()


def str2bool(x):
    return x.lower() in ('true')
    