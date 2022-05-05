import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np
import nnabla as nn
import torch
from modeling import BertLayer, BertConfig
import argparse
from convert_weights_encoder import rename_params
import collections
# re-using weight comparison code
from weight_comparison import compare_weights
from weight_comparison import print_and_plot_difference, setup_context
from nnabla.logger import logger


def nnabla_forward_hook(f):
    print(f"{f.name} | mean:{f.outputs[0].d.mean():>.6f} |"
          f" max:{f.outputs[0].d.max():>.6f} | min:{f.outputs[0].d.min():>.6f} "
          f"| std:{f.outputs[0].d.std():>.6f}")


def nnabla_backward_hook(f):
    print(f"{f.name} | mean:{f.outputs[0].g.mean():>.6f} |"
          f" max:{f.outputs[0].g.max():>.6f} | min:{f.outputs[0].g.min():>.6f} "
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
                        print(f"{self.__class__.__name__} | mean:{x.mean():>.6f}"
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
                            print(f"{self.__class__.__name__} | mean:{x.mean():>.6f}"
                                  f"| max:{x.max():>.6f} | min:{x.min():>.6f} "
                                  f"| std:{x.std():>.6f})")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="bert_config_base.json",
                        type=str)
    args = parser.parse_args()
    args.local_rank = "0"
    args.device = torch.device(f"cuda:{args.local_rank}")
    args.batch_size = 32
    args.num_heads = 12
    args.embed_dim = 768
    args.dim_feedforward = 3072
    args.seq_len = 128
    return args


def create_sample_data(args):

    if args.create_data:
        # [32, 128, 768]
        hs = np.random.rand(args.batch_size, args.seq_len, args.embed_dim)
        # [32, 128]
        attention_mask = np.ones((args.batch_size, args.seq_len))
        np.savez('encoder_test_data', hs=hs, attention_mask=attention_mask)


def get_sample_data(file):
    old_dict = np.load(file)
    hs = old_dict["hs"]
    attn = old_dict["attention_mask"]
    return {
        "hs": hs,
        "attention_mask": attn
    }


def create_nnabla_model(args):

    # we are creating a transposed hs
    hs = nn.Variable([args.seq_len, args.batch_size, args.embed_dim])
    attention_mask = nn.Variable([args.batch_size, args.seq_len])
    # extended_attention_mask = (1.0 - attention_mask) * -10000.0
    encoder = PF.transformer_encode(hs, args.embed_dim, args.num_heads,
                                    dim_feedforward=args.dim_feedforward,
                                    dropout=0.0, activation=F.gelu,
                                    name='encoder01',
                                    src_additive_mask=None,
                                    src_key_padding_mask=attention_mask)

    model = {}
    model["model"] = encoder
    model["model"].persistent = True

    model['hs'] = hs
    model['attention_mask'] = attention_mask
    return model


def create_pytorch_model(args, save_weights=False):

    # bert encoder
    config = BertConfig(args.config)
    encoder = BertLayer(config)
    encoder = encoder.to(args.device)
    encoder.eval()
    if save_weights:
        torch.save(encoder.state_dict(), 'pyt_encoder.pt')
    return encoder


def create_pytorch_inbuilt_model(args):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=args.embed_dim,
        nhead=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        activation='gelu', batch_first=True)
    encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
    encoder = encoder.to(args.device)
    return encoder


def nnabla_assign_data(var_dict, data_dict, args):
    for key in data_dict:
        if key == "hs":
            var_dict[key].d = np.transpose(data_dict[key], (1, 0, 2))
            var_dict[key].data.cast(np.float32, args.ctx)
        else:
            # var_dict[key].d = (data_dict[key] - 1) * -1
            var_dict[key].data.cast(np.float32, args.ctx)
            print(
                f"nnabla {key} max {np.max(var_dict[key].d)} | "
                f"min {np.min(var_dict[key].d)}")


def pytorch_assign_data(data_dict):

    hs = torch.tensor(data_dict['hs'], dtype=torch.float32, device=args.device)

    attention_mask = torch.tensor(
        data_dict['attention_mask'][:, None, None, :],
        dtype=torch.float32, device=args.device)
    attention_mask = (1.0 - attention_mask) * -10000.0

    return hs, attention_mask


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


def print_keys(dictionary):
    for key in dictionary:
        print(key)


def main(args):

    # data creation
    args.create_data = False
    create_sample_data(args)
    data_dict = get_sample_data("encoder_test_data.npz")

    # load pytorch model
    torch.nn.modules.module.register_module_forward_hook(pytorch_forward_hook)
    torch.nn.modules.module.register_module_full_backward_hook(pytorch_backward_hook)
    pyt_model = create_pytorch_model(args, False)
    py_weights = torch.load("pyt_encoder.pt", map_location=args.device)
    pyt_model.load_state_dict(py_weights)

    # load nnabla model
    setup_context(args)
    nn.clear_parameters()
    nn_model_dict = create_nnabla_model(args)
    nn.load_parameters("nn_encoder.h5")
    # compare weights before data assignment
    print("weight comparision before data assignment")
    pyt_weight_dict = convert_pytorch_weights_to_nnabla(pyt_model.state_dict())
    compare_weights(nn.get_parameters(), pyt_weight_dict)
    print("weight comparision done")

    # data assignment
    nnabla_assign_data(nn_model_dict, data_dict, args)
    hs, attention_mask = pytorch_assign_data(data_dict)

    # compare weights after data assignment
    print("weight comparision after data assignment")
    pyt_weight_dict = convert_pytorch_weights_to_nnabla(pyt_model.state_dict())
    compare_weights(nn.get_parameters(), pyt_weight_dict)
    print("weight comparision done\n")

    # forward
    print("\n\n\n")
    pyt_output = pyt_model(hs, attention_mask)
    print("\n\n\n")
    nn_model_dict["model"].forward(
        clear_no_need_grad=True, function_post_hook=nnabla_forward_hook)
    # nn_model_dict["model"].forward(clear_no_need_grad=True)
    print("\n\n\n")
    print_and_plot_difference(
        "encoder_out", nn_model_dict["model"], pyt_output, encoder_layer=True)

    print("\n\n\n")
    pyt_output.mean().backward()
    print("\n\n\n")

    nn_model_dict["model"].backward(
        clear_buffer=True, function_post_hook=nnabla_backward_hook)


if __name__ == "__main__":

    args = get_args()
    main(args)
