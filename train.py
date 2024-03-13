from torch import manual_seed, tensor, save, cat
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda import device_count
from torch.nn import DataParallel
from utils import create_model, create_dataset
from model import CRW
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt
import argparse
import time
import os
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Train', add_help=False)
    # Meta
    parser.add_argument('--tune', default = False, type=bool, help='True=Use Raytune, False=Train 1 run with default')
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 0, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(48,16), type=int)
    parser.add_argument('--seq_length', default=8, type=int)
    parser.add_argument('--overlap', default=(32,0), nargs = '+', type=int)
    # Train
    parser.add_argument('--batch_size', default = 32, type=int)
    parser.add_argument('--epochs', default = 10, type = int)
    parser.add_argument('--lr', default = 1E-3, type = float)
    parser.add_argument('--tau', default = 0.01, type = float)
    # Dev
    parser.add_argument('--pos_embed', default = False, type = bool)
    parser.add_argument('--dataset_full', default = True)
    return parser

def main(args):
    print(args)
    if args.tune:
        checkpoint = session.get_checkpoint()
    # Model
    encoder = create_model(args.model, args.pos_embed)
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    model = CRW(encoder, args.tau, args.pos_embed)
    model = model.to('cuda')

    # Dataset
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, full = args.dataset_full, overlap = args.overlap)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs

    # Train
    model.train(True)
    loss_tot = []
    for epoch in range(epochs):
        t0 = time.time()
        loss_epoch = []
        for batch, seq in enumerate(dataloader):
            seq = seq.to('cuda')
            loss, _ = model(seq)
            loss_epoch.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch = tensor(loss_epoch).mean()
        loss_tot.append(loss_epoch)
        print('Epoch:',epoch,'Loss:',loss_epoch.item(), 'Time:', time.time()-t0)

    if args.tune:
        checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        checkpoint = Checkpoint.from_dict(checkpoint_data)
        session.report(
            {"loss": loss_tot[-1]},
            checkpoint=checkpoint,
        )

    plt.plot(loss_tot)
    plt.savefig('/home/jordydalcorso/workspace/crw/output/_loss.png')
    plt.close()
    save(encoder.state_dict(), '/home/jordydalcorso/workspace/crw/latest.pt')
    print('Finished training.')

def train_crw(config):
    print(os.environ["TUNE_ORIG_WORKING_DIR"])
    args = argparse.Namespace(**config)
    main(args) 

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.overlap = tuple(args.overlap)
    if not args.tune:
        main(args)
    else:
        # Use Ray Tune for hyperparameter tuning
        config = {
                "tune":1,
                "model":1,
                "dataset":0,
                "seq_length":8,
                "epochs":1,
                "batch_size": tune.choice([16, 32]),
                "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5]),
                "tau": tune.choice([1e-1, 1e-2, 1e-3, 1e-4]),
                "patch_size": tune.choice([(16,16),(32,32),(24,24),(48,48),(24,32),(16,24),(32,48)]),
                "overlap": tune.choice([(0,0), (8,8), (12,12), (4,4), (4,12)]),
                "pos_embed":False,
                "dataset_full":True
            }
        
        scheduler = ASHAScheduler(
            max_t=3,
            mode = 'min',
            grace_period=1,
            reduction_factor=2,
        )

        analysis = tune.run(
            train_crw,
            config = config,
            metric = 'loss',
            resources_per_trial={"gpu": 1},
            scheduler=scheduler,
            local_dir ='/home/jordydalcorso/workspace/crw/output/ray_results',
            num_samples=50,
            checkpoint_score_attr=None
        )

        print(analysis.trials[0].metric_analysis.keys())
        for an in analysis.trials:
            an.metric_analysis['loss'] = an.metric_analysis.pop('time_this_iter_s')
        best_trial = analysis.get_best_trial(metric = "loss", mode = "min")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")