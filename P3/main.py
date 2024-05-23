from argparse import ArgumentParser

from data import CelebADataset
from models import GAN
from models import VariationalAutoEncoder as VAE
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.framework.ops import disable_eager_execution

if __name__ == "__main__":

    parser = ArgumentParser(description="CelebA generation")
    parser.add_argument(
        "model", type=str, choices=["vae", "wgan", "aegan"], help="Model to execute"
    )
    parser.add_argument(
        "-p",
        "--path",
        default="results",
        type=str,
        help="Path to store the model and predictions",
    )
    parser.add_argument("--hidden-size", type=int, default=200, help="Hidden size")
    parser.add_argument("--dilation", type=bool, default=False, help="Boolean flag for dilation instead of stride in VAE")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=1500, help="Number of steps per epoch")   

    args = parser.parse_args()
    train, val, test = map(CelebADataset, ("train", "val", "test"))

    if args.dilation:
        path = f"{args.path}/{args.model}_{args.hidden_size}_dilation"
    else:
        path = f"{args.path}/{args.model}_{args.hidden_size}_strides"

    if args.model == "vae":
        disable_eager_execution()
        vae = VAE(
            CelebADataset.IMG_SIZE,
            hidden_size=args.hidden_size,
            filters=[16, 32, 32, 32],
            kernels=[3, 3, 3, 3],
            strides=[2, 2, 2, 2],
            dilation=args.dilation,
        )
        vae.train(
            train,
            val,
            test,
            path,
            optimizer=Adam(1e-4),
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps,
        )
    elif 'gan' in args.model:
        gan = GAN(
            CelebADataset.IMG_SIZE,
            hidden_size=args.hidden_size,
            critic_steps=3,
            gp_weight=10,
            autoencoder=('aegan' == args.model)
        )
        gopt = Adam(2e-4, beta_1=0.5, beta_2=0.999)
        copt = Adam(2e-4, beta_1=0.5, beta_2=0.999)
        gan.train(
            train,
            val,
            test,
            path,
            g_optimizer=gopt,
            c_optimizer=copt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps,
        )
    else:
        raise NotImplementedError
