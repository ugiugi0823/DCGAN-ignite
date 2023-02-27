import argparse


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True,
        choices={"cifar10", "lsun", "imagenet", "folder", "lfw", "fake", "mnist"},
        help="Type of the dataset to be used.",
    )

    p.add_argument(
      "--dataroot", 
      required=True, 
      help="path to dataset"
    )

    p.add_argument(
      "--workers", 
      type=int, 
      default=2, 
      help="number of data loading workers"
    )

    p.add_argument(
      "--batch-size", 
      type=int, 
      default=64, 
      help="input batch size"
    )

    p.add_argument(
      "--z-dim", 
      type=int, 
      default=100, 
      help="size of the latent z vector"
    )

    p.add_argument(
        "--g-filters", 
        type=int, 
        default=64, 
        help="Number of filters in the second-to-last generator deconv layer"
    )

    p.add_argument(
      "--d-filters", 
      type=int, 
      default=64, 
      help="Number of filters in first discriminator conv layer"
    )

    p.add_argument(
      "--epochs", 
      type=int, 
      default=25, 
      help="number of epochs to train for"
    )

    p.add_argument(
      "--lr", 
      type=float, 
      default=0.0002, 
      help="learning rate"
    )

    p.add_argument(
      "--beta-1", 
      type=float, 
      default=0.5, 
      help="beta_1 for adam"
    )

    p.add_argument(
      "--no-cuda", 
      action="store_true", 
      help="disables cuda"
    )

    p.add_argument(
      "--saved-G", 
      default="", 
      help="path to pickled generator (to continue training)"
    )

    p.add_argument(
      "--saved-D", 
      default="", 
      help="path to pickled discriminator (to continue training)"
    )

    p.add_argument(
      "--output-dir", 
      default=".", 
      help="directory to output images and model checkpoints"
    )

    p.add_argument(
      "--seed", 
      type=int, 
      help="manual seed"
    )

    p.add_argument(
      "--alpha", 
      type=float, 
      default=0.98, 
      help="smoothing constant for exponential moving averages"
    )

    args = p.parse_args()
    
    # dev = "cpu" if (not torch.cuda.is_available() or args.no_cuda) else "cuda:0"

    # args.output_dir = Path(args.output_dir)
    # try:
    #     args.output_dir.mkdir(parents=True)
    # except FileExistsError:
    #     if (not args.output_dir.is_dir()) or (len(os.listdir(args.output_dir)) > 0):
    #         raise FileExistsError("Please provide a path to a non-existing or empty directory.")
    
    return args    
