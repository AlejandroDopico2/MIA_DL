from argparse import ArgumentParser

from data import CelebADataset
from models import WGAN
from models import VariationalAutoEncoder as VAE
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.framework.ops import disable_eager_execution

if __name__ == "__main__":

    parser = ArgumentParser(description="CelebA generation")
    parser.add_argument(
        "model", type=str, choices=["vae", "wgan"], help="Model to execute"
    )
    parser.add_argument(
        "-p",
        "--path",
        default="results",
        type=str,
        help="Path to store the model and predictions",
    )
    parser.add_argument("--hidden-size", type=int, default=200, help="Hidden size")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument('--steps', type=int, default=1500, help='Number of steps per epoch')

    args = parser.parse_args()
    train, val, test = map(CelebADataset, ("train", "val", "test"))

    path = f"{args.path}/{args.model}_{args.hidden_size}"

    if args.model == "vae":
        disable_eager_execution()
        vae = VAE(
            CelebADataset.IMG_SIZE,
            hidden_size=args.hidden_size,
            filters=[16, 32, 32, 32],
            kernels=[3, 3, 3, 3],
            strides=[2, 2, 2, 2],
        )
        vae.train(
            train,
            val,
            test,
            path,
            optimizer=Adam(1e-4),
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps
        )
    elif args.model == "wgan":
        wgan = WGAN(
            CelebADataset.IMG_SIZE,
            hidden_size=args.hidden_size,
            critic_steps=3,
            gp_weight=2,
        )
        gopt = Adam(2e-4, beta_1=0.5, beta_2=0.999)
        copt = Adam(2e-4, beta_1=0.5, beta_2=0.999)
        wgan.train(
            train,
            val,
            test,
            path,
            g_optimizer=gopt,
            c_optimizer=copt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps
        )
    else:
        raise NotImplementedError
