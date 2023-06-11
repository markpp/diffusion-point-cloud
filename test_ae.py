import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD

import pandas as pd
from pyntcloud import PyntCloud

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/AE_chair.pt')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_refs = []
all_recs = []
for i, batch in enumerate(tqdm(test_loader)):
    refs = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(refs)
        recs = model.decode(code, refs.size(1), flexibility=ckpt['args'].flexibility).detach()

    refs = refs * scale + shift
    recs = recs * scale + shift

    refs = refs.detach().cpu()
    recs = recs.detach().cpu()

    all_refs.append(refs)
    all_recs.append(recs)

    print(refs.shape)

    df = pd.DataFrame(refs[0].numpy(), columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(os.path.join(save_dir,f'ref_{i}.ply'))

    df = pd.DataFrame(recs[0].numpy(), columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(os.path.join(save_dir,f'rec_{i}.ply'))
    break


all_refs = torch.cat(all_refs, dim=0)
all_recs = torch.cat(all_recs, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_refs.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recs.numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recs.to(args.device), all_refs.to(args.device), batch_size=args.batch_size)
cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
logger.info('CD:  %.12f' % cd)
logger.info('EMD: %.12f' % emd)
