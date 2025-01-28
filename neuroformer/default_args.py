import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--train", action="store_true", default=False, help="Train mode")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=True, help="Enable or disable Weights & Biases logging")
    parser.add_argument("--dist", action="store_true", default=False, help="Distributed mode")
    parser.add_argument("--seed", type=int, default=25, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--rand_perm", action="store_true", default=False, help="Randomly permute the ID column")
    parser.add_argument("--mconf", type=str, default=None, help="Path to model config file")
    parser.add_argument("--eos_loss", action="store_true", default=False, help="Use EOS loss")
    parser.add_argument("--no_eos_dt", action="store_true", default=False, help="No EOS dt token")
    parser.add_argument("--downstream", action="store_true", default=False, help="Downstream task")
    parser.add_argument("--freeze_model", action="store_true", default=False, help="Freeze model")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="Distance-Coding")
    parser.add_argument("--behavior", action="store_true", default=False, help="Behavior task")
    parser.add_argument("--pred_behavior", action="store_true", default=False, help="Predict behavior")
    parser.add_argument("--past_state", action="store_true", default=False, help="Input past state")
    parser.add_argument("--visual", action="store_true", default=False, help="Visualize")
    parser.add_argument("--contrastive", action="store_true", default=False, help="Contrastive")
    parser.add_argument("--clip_loss", action="store_true", default=False, help="Clip loss")
    parser.add_argument("--clip_vars", nargs="+", default=['id','frames'], help="Clip variables")
    parser.add_argument("--class_weights", action="store_true", default=False, help="Class weights")
    parser.add_argument("--resample", action="store_true", default=False, help="Resample")
    parser.add_argument("--loss_bprop", nargs="+", default=None, help="Loss type to backpropagate")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--true_past", action="store_true", default=False, help="True past")
    parser.add_argument("--predict_modes", nargs='+', default=None, help="List of modes to predict")
    parser.add_argument("--finetune", action="store_true", default=False, help="Finetune")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=250, help="Number of epochs")
    return parser.parse_args()

class DefaultArgs:
    def __init__(self):
        self.train = False
        self.dist = False
        self.seed = 69
        self.downstream = False
        self.title = None
        self.resume = None
        self.rand_perm = False
        self.mconf = None
        self.eos_loss = False
        self.no_eos_dt = False
        self.freeze_model = False
        self.dataset = "lateral"
        self.behavior = False
        self.pred_behavior = False
        self.visual = True
        self.past_state = True
        self.contrastive = False
        self.clip_loss = True
        self.clip_vars = ['id','frames']
        self.class_weights = False
        self.resample = False
        self.loss_bprop = None
        self.config = None
        self.sweep_id = None
        self.ckpt_path = None
        self.true_past = False
        self.predict_modes = None
        self.finetune = False
        self.batch_size = 32
        self.n_epochs = 250