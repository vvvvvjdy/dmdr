import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Training")

    # deepspeed
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to deepspeed config file.")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--logging-dir", type=str, default="logs")

    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpoint-steps", type=int, default=20000)
    parser.add_argument("--max-train-steps", type=int, default=200000)

    # Gen model
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--xl-ode-pretrain-path", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--sample-num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--use-lora-gen",type=float, default=1, help="use if > 1")

    # Guidance model
    parser.add_argument("--use-lora", type=float, default=2, help="use if > 0")
    parser.add_argument("--lora-rank", type=int, default=64, help="Rank for LoRA units.")
    parser.add_argument("--lora-scale-f", type=float, default=1.0, help="Guidance scale for the fake estimator.")
    parser.add_argument("--lora-scale-r", type=float, default=0.25, help="Guidance scale for the real estimator.")

    # dataset
    parser.add_argument("--data-path-train", type=str, default="../data/x.txt")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution for training.")
    parser.add_argument("--batch-size", type=int, default=8, help="local batch size train.")
    parser.add_argument("--batch-size-test", type=int, default=1, help="local batch size train.")

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use-8bit-adam", action=argparse.BooleanOptionalAction, default=False,)

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate-gen", type=float, default=1e-6)
    parser.add_argument("--learning-rate-gui", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=30)

    # cpu
    parser.add_argument("--num-workers", type=int, default=16)

    # loss
    parser.add_argument("--cold-start-iter", type=int, default=0, help="Number of iterations to cold start the DMD loss.")
    parser.add_argument("--gen-a", type=float, default=1.0, help="Mean for log-normal distribution for Train.")
    parser.add_argument("--gen-b", type=float, default=1.0,
                        help="Standard deviation for log-normal distribution for Train.")
    parser.add_argument("--s-type-gen", type=str, default="normal", choices=["logit_normal", "uniform"],)
    parser.add_argument("--gui-a", type=float, default=1.0, help="Mean for log-normal distribution for DMD loss.")
    parser.add_argument("--gui-b", type=float, default=1.0,
                        help="Standard deviation for log-normal distribution for DMD loss.")
    parser.add_argument("--s-type-gui", type=str, default="normal", choices=["logit_normal", "uniform"],)
    parser.add_argument("--dynamic-step", type=int, default=0)
    parser.add_argument("--ratio-update", type=float, default=5.0,
                        help="Ratio of generator update to discriminator update.")
    parser.add_argument("--cfg-r", type=float, default=1.0, help="Classifier-free guidance scale for real estimator. (middle), set to 1.0 to disable.")
    parser.add_argument("--cfg-f", type=float, default=0.0, help="Classifier-free guidance scale for fake estimator (middle) , set to 1.0 to disable.")

    # reward ReFL
    parser.add_argument("--clip-reward-model", type=str, default=None, help="id of the reward model for clip rewards.")
    parser.add_argument("--hpsv2-reward-model", type=str, default=None, help="ckpt path of the reward model for hps rewards.")
    parser.add_argument("--reward-clip-weight",type=float, default=0.2,)
    parser.add_argument("--reward-hps-weight",type=float, default=0.5,)
    parser.add_argument("--magic-prompt", type=str, default=None, help="flowing SRPO to inject magic prompt for hpsv2 reward.")





    args = parser.parse_args()

    return args