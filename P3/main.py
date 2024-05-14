from data import CelebADataset
from models import VariationalAutoEncoder as VAE, WGAN
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.framework.ops import disable_eager_execution


from argparse import ArgumentParser

if __name__ == '__main__':
    
    parser = ArgumentParser(description='CelebA generation')
    parser.add_argument('model', type=str, choices=['vae', 'wgan'], help='Model to execute')
    parser.add_argument('-p', '--path', default='results/model', type=str, help='Path to store the model and predictions')
    parser.add_argument('--hidden-size', type=int, default=200, help='Hidden size')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    train, val, test = map(CelebADataset, ('train', 'val', 'test'))
    if args.model == 'vae':
        disable_eager_execution()
        vae = VAE(CelebADataset.IMG_SIZE, hidden_size = args.hidden_size, filters = [16, 32, 32, 32], kernels = [3,3,3,3], strides = [2,2,2,2])
        vae.train(train, val, test, args.path, optimizer=Adam(1e-4), epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == 'wgan':
        wgan = WGAN(CelebADataset.IMG_SIZE, hidden_size = args.hidden_size, critic_steps=3, gp_weight=10)
        gopt = Adam(2e-4, beta_1=0.5, beta_2=0.999)
        copt = Adam(2e-4, beta_1=0.5, beta_2=0.999)
        wgan.train(train, test, args.path, g_optimizer=gopt, c_optimizer=copt, epochs=args.epochs, batch_size=args.batch_size)
    else:
        raise NotImplementedError
        
        
