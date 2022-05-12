from nnbla_model import BertModel, bert_for_pre_training
import argparse
import nnabla as nn
import numpy as np
import torch
import modeling
from nnabla.functions import softmax_cross_entropy, mean
from convert_weights import rename_params
import collections
import nnabla.communicators as Comm
from nnabla.ext_utils import get_extension_context
import sys
from nnabla.logger import logger
import random
import matplotlib.pyplot as plt
import nnabla.functions as F
from pathlib import Path
# TODO: compare GPU weights


def setting_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def nnabla_forward_hook(f):
    print(f"{f.name} | mean:{f.outputs[0].d.mean():>.6f} |"
          f" max:{f.outputs[0].d.max():>.6f} "
          f"| min:{f.outputs[0].d.min():>.6f} "
          f"| std:{f.outputs[0].d.std():>.6f}")


def nnabla_backward_hook(f):
    print(f"{f.name} | mean:{f.outputs[0].g.mean():>.6f} |"
          f" max:{f.outputs[0].g.max():>.6f} "
          f"| min:{f.outputs[0].g.min():>.6f} "
          f"| std:{f.outputs[0].g.std():>.6f}")


def pytorch_forward_hook(self, input_d, output_d):
    if isinstance(output_d, torch.Tensor):
        print(f"{self.__class__.__name__} | mean:{output_d.mean():>.6f}"
              f"| max:{output_d.max():>.6f} | min:{output_d.min():>.6f} "
              f"| std:{output_d.std():>.6f})")
    else:
        for x in output_d:
            if isinstance(x, torch.Tensor):
                print(f"{self.__class__.__name__} | mean:{x.mean():>.6f}"
                      f"| max:{x.max():>.6f} | min:{x.min():>.6f} "
                      f"| std:{x.std():>.6f})")
            else:
                for y in x:
                    if isinstance(x, torch.Tensor):
                        print(f"{self.__class__.__name__} "
                              f"| mean:{x.mean():>.6f}"
                              f"| max:{x.max():>.6f} | min:{x.min():>.6f} "
                              f"| std:{x.std():>.6f})")


def pytorch_backward_hook(self, grad_input, grad_output):
    if isinstance(grad_output, torch.Tensor):
        print(f"{self.__class__.__name__} | mean:{grad_output.mean():>.6f}"
              f"| max:{grad_output.max():>.6f} | min:{grad_output.min():>.6f} "
              f"| std:{grad_output.std():>.6f})")
    else:
        for x in grad_output:
            if isinstance(x, torch.Tensor):
                print(f"{self.__class__.__name__} | mean:{x.mean():>.6f}"
                      f"| max:{x.max():>.6f} | min:{x.min():>.6f} "
                      f"| std:{x.std():>.6f})")
            else:
                if x is not None:
                    for y in x:
                        if isinstance(x, torch.Tensor):
                            print(f"{self.__class__.__name__} "
                                  f"| mean:{x.mean():>.6f}"
                                  f"| max:{x.max():>.6f} | min:{x.min():>.6f} "
                                  f"| std:{x.std():>.6f})")


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(
            prediction_scores.view(-1, self.vocab_size),
            masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(
            seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


def bert_pretraining_criterion(MLM_predictions: nn.Variable,
                               NSP_predictions: nn.Variable,
                               MLM_labels: nn.Variable,
                               NSP_labels: nn.Variable,
                               vocab_size: int = 30_522):
    """BERT pre-training loss function

    Args:
        MLM_predictions (nn.Variable): Mask language model prediction scores.
                        of shape [batch_size, sequence_length , vocab_size]
        NSP_predictions (nn.Variable): Next sentence prediction scores.
                        of shape [batch_size, 2]
        MLM_labels (nn.Variable): True labels for MLM.
                        of shape [batch_size, sequence_length]
        NSP_labels (nn.Variable): True labels for NSP.
                        of shape [batch_size, 1]
        vocab_size (int, optional): Size of the Vocabulary.
                        Defaults to 30,522.

    Returns:
        total_loss (nn.Variable): Sum of MLM_loss and NSP_loss.
                    of shape [1]
    """

    MLM_loss = mean(softmax_cross_entropy(
        MLM_predictions,
        MLM_labels.reshape(MLM_labels.shape + (1,))))

    NSP_loss = mean(softmax_cross_entropy(
        NSP_predictions,
        NSP_labels.reshape(NSP_labels.shape + (1,))))

    total_loss = MLM_loss + NSP_loss
    return total_loss


def setup_context(args: argparse.ArgumentParser):

    # setting up the context
    try:
        extension_module = "cudnn"
        ctx = get_extension_context(extension_module)
        comm = Comm.MultiProcessCommunicator(ctx)
        comm.init()
        args.world_size = comm.size
        # args.local_rank = comm.rank
        ctx = get_extension_context(extension_module,
                                    device_id=args.local_rank)
        nn.set_default_context(ctx)
        args.ctx = ctx
        return args
    # QUESTION: check this way of exiting is ok?
    except Exception as e:
        print(e.with_traceback())
        print(f"args given {e.args}")
        sys.exit()


def create_sample_data(vocab_size=30528, batch_size=8,
                       sequence_length=128):
    input_ids_data = np.random.randint(0, vocab_size,
                                       size=(batch_size, sequence_length))
    token_type_ids_data = np.random.randint(0, 2,
                                            size=(batch_size, sequence_length))

    attention_mask_data = np.random.randint(0, 2,
                                            size=(batch_size, sequence_length))

    MLM_label_data = np.random.randint(0, vocab_size,
                                       size=(batch_size, sequence_length))
    NSP_label_data = np.random.randint(
        0, 2, size=batch_size)

    np.savez('test_data', input_ids=input_ids_data,
             token_type_ids=token_type_ids_data,
             attention_mask=attention_mask_data, MLM_label=MLM_label_data,
             NSP_label=NSP_label_data)


def get_sample_data(file):
    return np.load(file)


def get_nnabla_model(args):
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    vocab_size = args.vocab_size
    nn_input_ids = nn.Variable([batch_size, sequence_length])
    nn_token_type_ids = nn.Variable([batch_size, sequence_length])
    nn_attention_mask = nn.Variable([batch_size, sequence_length])
    MLM_labels = nn.Variable([batch_size, sequence_length])
    NSP_labels = nn.Variable([batch_size, ])

    nn.clear_parameters()
    model = BertModel()
    [encoder, pooled,
     en_layers, embedding_output] = model(args, nn_input_ids,
                                          nn_attention_mask,
                                          nn_token_type_ids,
                                          vocab_size=vocab_size)
    encoder.persistent = True
    pooled.persistent = True
    embedding_output.persistent = True
    mlm_pred, nsp_pred = bert_for_pre_training(encoder, pooled)
    mlm_pred.persistent = True
    nsp_pred.persistent = True
    for i in range(len(en_layers)):
        en_layers[i].persistent = True

    loss = bert_pretraining_criterion(
        mlm_pred, nsp_pred, MLM_labels, NSP_labels)
    loss.persistent = True
    return {"input_ids": nn_input_ids,
            "token_type_ids": nn_token_type_ids,
            "attention_mask": nn_attention_mask,
            "MLM_label": MLM_labels,
            "NSP_label": NSP_labels,
            "mlm_pred": mlm_pred,
            "nsp_pred": nsp_pred,
            "loss": loss,
            "encoder": encoder,
            "pooled": pooled,
            "en_layers": en_layers,
            "embedding_output": embedding_output}


def nnabla_assign_data(var_dict, data_dict, args):
    for key in data_dict:
        if key == "attention_mask":
            var_dict[key].d = (data_dict[key]-1) * -1
        else:
            var_dict[key].d = data_dict[key]
        var_dict[key].data.cast(np.int32, args.ctx)


def nnabla_zero_grad():
    for k, v in nn.get_parameters().items():
        try:
            v.grad.zero()
        except Exception as e:
            print(e.with_traceback())
            print(f"args given {e.args}")


def get_pytroch_model(args):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.bert_config)
    config.vocab_size = args.vocab_size
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)
    model = model.to(device=args.device)
    py_loss = BertPretrainingCriterion(args.vocab_size)
    return model, py_loss


def pytorch_assign_data(args, data_dict):
    py_input_ids = torch.tensor(
        data_dict['input_ids'], device=args.device, dtype=torch.int32)
    py_token_type_ids = torch.tensor(
        data_dict['token_type_ids'], device=args.device, dtype=torch.int32)
    py_attention_mask = torch.tensor(
        data_dict['attention_mask'], device=args.device, dtype=torch.int32)
    py_mlm_label = torch.tensor(
        data_dict['MLM_label'], device=args.device, dtype=torch.int64)
    py_nsp_label = torch.tensor(
        data_dict['NSP_label'], device=args.device, dtype=torch.int64)

    return [py_input_ids, py_token_type_ids,
            py_attention_mask, py_mlm_label, py_nsp_label]


def convert_pytorch_weights_to_nnabla(py_state_dict):
    new_weight_dict = collections.OrderedDict()
    for k, v in py_state_dict.items():
        key, value = rename_params(k, v.cpu().numpy())
        if key not in new_weight_dict:
            new_weight_dict[key] = value
        else:
            logger.error(f"{key} converted twice and "
                         f"is of shape {tuple(new_weight_dict[key].shape)}")
    return new_weight_dict


def compare_weights(nnabla_weights, pytorch_weights):
    assert len(nnabla_weights) == len(pytorch_weights), (
        "the weights length doesn't match. "
        f"nnabla weights len: {len(nnabla_weights)} "
        f"pytorch weights len : {len(pytorch_weights)}"
    )
    mismatch_weights = 0
    for key in pytorch_weights:
        if nnabla_weights[key].shape == pytorch_weights[key].shape:
            if not (nnabla_weights[key].d == pytorch_weights[key]).all():
                logger.error(f"{key} doesn't have matching weights")
                mismatch_weights += 1
        else:
            logger.error(f"for key: {key} weight shape don't match.")
            mismatch_weights += 1
    if mismatch_weights > 0:
        logger.info(f"number of mismatch weights {mismatch_weights}")


def compute_mean_max_min_std(diff: np.ndarray, encoder_layer=False):
    if encoder_layer:
        diff = np.transpose(diff, (1, 0, 2))
    d_mean = np.mean(diff)
    d_max = np.max(diff)
    d_min = np.min(diff)
    d_std = np.std(diff)
    return [d_mean, d_max, d_min, d_std]


def generate_histogram(name, arr, base_dir):
    plt.style.use("fivethirtyeight")
    plt.hist(arr, alpha=0.5, label=f"{name}", edgecolor='k')
    plt.axvline(arr.mean(), label=f"mean {arr.mean():<.5f}",
                linestyle="dashed",
                color='red', linewidth=1)
    plt.axvline(arr.max(), label=f"max {arr.max():<.5f}", linestyle="dashed",
                color='red', linewidth=1)
    plt.axvline(arr.min(), label=f"min {arr.min():<.5f}", linestyle="dashed",
                color='red', linewidth=1)
    plt.title(f"{name} histogram")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)
    base_dir = str(base_dir)
    plt.savefig(f"{base_dir}/{name}_hist.jpg")
    plt.close()


def print_mean_max_min_std(name, arr, type="diff", encoder_layer=False):
    print("{:<10} {:<12}\t {:<.6f}\t {:<.6f}\t {:<.6f}\t {:<.6f}".format(
        name, type,
        *compute_mean_max_min_std(arr, encoder_layer=encoder_layer)))


def print_and_plot_difference(name: str, nn_var: nn.Variable, py_var,
                              base_dir="plots", for_grad=False,
                              create_hist=False, encoder_layer=False):

    if not encoder_layer:
        assert nn_var.shape == py_var.shape, (
            f"the shape of the nnabla variable {nn_var.shape} doesn't match "
            f"with the shape of PyTorch variable { py_var.shape}")

    else:
        assert F.transpose(nn_var, (1, 0, 2)).shape == py_var.shape, (
            f"the shape of the nnabla variable"
            f" {F.transpose(nn_var, (1, 0, 2)).shape} doesn't match "
            f"with the shape of PyTorch variable { py_var.shape}")
    print_mean_max_min_std(f"{name}", nn_var.d,
                           type="NNabla_d", encoder_layer=encoder_layer)
    py_var_d = py_var.detach().cpu().numpy()
    print_mean_max_min_std(f"{name}", py_var_d, type="PyTorch_d")

    if encoder_layer:
        py_var_d = np.transpose(py_var_d, (1, 0, 2))
    diff_d = nn_var.d - py_var_d
    print_mean_max_min_std(name, diff_d)
    print("\n")
    if create_hist:
        generate_histogram(f"{name}", diff_d.reshape(-1), base_dir=base_dir)
    if for_grad:
        py_var_g = py_var.grad.cpu().numpy()
        if encoder_layer:
            py_var_g = np.transpose(py_var_g, (1, 0, 2))
        print_mean_max_min_std(f"{name}", nn_var.g,
                               type="NNabla_g", encoder_layer=encoder_layer)
        print_mean_max_min_std(f"{name}", py_var_g, type="PyTorch_g")
        diff_g = nn_var.g - py_var_g
        print_mean_max_min_std(name, diff_g, type="grad")
        print("\n")
        if create_hist:
            generate_histogram(
                f"{name}", diff_g.reshape(-1), base_dir=base_dir)


if __name__ == "__main__":

    plt.style.use("seaborn")
    logger.info('setting the seed.')
    setting_seed(12345)
    logger.info('done setting the seed.')
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_dropout",
                        default=0.1,
                        type=float)
    args = parser.parse_args()
    args.batch_size = 32
    args.vocab_size = 30528
    args.sequence_length = 128
    # create_sample_data(vocab_size=args.vocab_size,
    #                    batch_size=args.batch_size,
    #                    sequence_length=args.sequence_length)
    args.nnabla_weights = "nnabla_weights.h5"
    args.pytorch_weights = "pytorch_weights.pt"

    # pytorch device setting
    args.local_rank = "3"
    args.device = torch.device(f"cuda:{args.local_rank}")

    # create nnabla model and load weights
    args = setup_context(args)
    nn_model_dict = get_nnabla_model(args)
    nn.load_parameters(args.nnabla_weights)
    logger.info(f"length of nnabla weights: {len(nn.get_parameters())}")

    # create pytorch model
    args.bert_config = "bert_config_base.json"
    py_model, py_loss_criteria = get_pytroch_model(args)
    py_weights = torch.load(args.pytorch_weights, map_location=args.device)
    py_model.load_state_dict(py_weights)

    logger.info(f"the length of pytorch weights: {len(py_model.state_dict())}")

    # compare weights before forward
    logger.info('weight comparision before forward')
    nn_weight_dict = nn.get_parameters()
    py_weight_dict = convert_pytorch_weights_to_nnabla(py_model.state_dict())
    compare_weights(nn_weight_dict, py_weight_dict)
    logger.info('weight comparision done')

    # data loader
    data_dict = get_sample_data('test_data.npz')

    nnabla_zero_grad()
    # assign data to nnabla variable
    nnabla_assign_data(nn_model_dict, data_dict, args)
    # nnabla forward
    nn_model_dict["loss"].forward(clear_no_need_grad=True)

    # pytorch forward
    py_model.zero_grad()
    # create variable for pytorch
    ii, tti, am, mlm, nsp = pytorch_assign_data(args, data_dict)
    [mlm_pred, nsp_pred, py_encoder, py_pooled,
     encoder_layers, embedding_output] = py_model(ii, tti, am)
    py_loss = py_loss_criteria(mlm_pred, nsp_pred, mlm, nsp)
    py_encoder.retain_grad()
    py_pooled.retain_grad()
    mlm_pred.retain_grad()
    nsp_pred.retain_grad()
    py_loss.retain_grad()
    embedding_output.retain_grad()
    # for encoder_layers
    for i, _ in enumerate(encoder_layers):
        encoder_layers[i].retain_grad()

    # compare weight values after forward.
    logger.info('weight comparision after forward')
    nn_weight_dict = nn.get_parameters()
    py_weight_dict = convert_pytorch_weights_to_nnabla(py_model.state_dict())
    compare_weights(nn_weight_dict, py_weight_dict)
    logger.info('weight comparision done')

    # compare weight values after backward.
    nn_model_dict["loss"].backward(clear_buffer=True)
    py_loss.backward()
    logger.info('weight comparision after backward')
    nn_weight_dict = nn.get_parameters()
    py_weight_dict = convert_pytorch_weights_to_nnabla(py_model.state_dict())
    compare_weights(nn_weight_dict, py_weight_dict)
    logger.info('weight comparision done')

    # compute variables data difference.
    print("variable name \t\tmean\t\tmax\t\tmin\t\tstd")

    print_and_plot_difference(
        "input_ids", nn_model_dict["input_ids"], ii, for_grad=False)
    print_and_plot_difference(
        "token_type_ids", nn_model_dict["token_type_ids"], tti, for_grad=False)
    print_and_plot_difference(
        "attention_mask", nn_model_dict["attention_mask"], am, for_grad=False)
    print_and_plot_difference(
        "MLM_label", nn_model_dict["MLM_label"], mlm, for_grad=False)
    print_and_plot_difference(
        "NSP_label", nn_model_dict["NSP_label"], nsp, for_grad=False)

    print_and_plot_difference(
        "embedding_output", nn_model_dict["embedding_output"],
        embedding_output, for_grad=True)

    for i, ens in enumerate(zip(nn_model_dict["en_layers"], encoder_layers)):
        nn_en, py_en = ens
        inx = i + 1
        print_and_plot_difference(f"encoder_{inx}", nn_en, py_en,
                                  base_dir="plots/encoders",
                                  for_grad=True,
                                  encoder_layer=True,
                                  create_hist=False)

    print_and_plot_difference(
        "pooled", nn_model_dict["pooled"], py_pooled, for_grad=True)
    print_and_plot_difference(
        "mlm_pred", nn_model_dict["mlm_pred"], mlm_pred, for_grad=True)
    print_and_plot_difference(
        "nsp_pred", nn_model_dict["nsp_pred"], nsp_pred, for_grad=True)
    print_and_plot_difference(
        "loss", nn_model_dict["loss"], py_loss, for_grad=True)
