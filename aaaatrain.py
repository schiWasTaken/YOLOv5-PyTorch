import os
import sys
import subprocess

def main():
    # directory structure
    root = "."

    # training parameters
    ngpu = 1  # ignored (we're not using torch.distributed)
    dataset = "coco"
    batch_size = 1
    print_freq = 100
    lr = 0.01
    epochs = 10
    period = 300
    img_size1, img_size2 = 640, 640
    ckpt_file = f"yolov5s_{dataset}.pth"
    iters = -1

    # dataset paths
    if dataset == "voc":
        data_dir = os.path.join(root, "input/data/voc2012/VOCdevkit/VOC2012/")
    elif dataset == "coco":
        data_dir = os.path.join(root, r"datasets\coco")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # train.py path
    train_py = os.path.join(root, "train.py")

    # args (same as in your bash script, minus torch.distributed)
    args = [
        sys.executable, train_py,
        "--use-cuda",
        "--epochs", str(epochs),
        "--period", str(period),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--img-sizes", str(img_size1), str(img_size2),
        "--dataset", dataset,
        "--data-dir", data_dir,
        "--iters", str(iters),
        "--root", root,
        "--mosaic",
        "--dali",
        "--ckpt-path", os.path.join(root, "input/ckpts", ckpt_file),
        "--print-freq", str(print_freq),
    ]

    # logging
    # log_path = os.path.join(root, "data/logs/log.txt")
    # with open(log_path, "w") as log_file:
    process = subprocess.Popen(args)
    process.wait()

if __name__ == "__main__":
    main()
