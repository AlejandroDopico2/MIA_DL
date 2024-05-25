from argparse import ArgumentParser

from data import CelebADataset
from models import GAN
from models import VariationalAutoEncoder as VAE
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
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

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
        vae.model.load_weights(f"{path}/model.h5")
        vae.predict(
            test,
            path,
            batch_size=args.batch_size,
        )
    elif args.model == "wgan":
        wgan = GAN(
            CelebADataset.IMG_SIZE,
            hidden_size=args.hidden_size,
            critic_steps=3,
            gp_weight=10,
        )
        wgan.model.load_weights(f"{path}/checkpoint/checkpoint.ckpt")
        wgan.predict(
            test,
            path,
            batch_size=args.batch_size,
        )
    else:
        raise NotImplementedError
