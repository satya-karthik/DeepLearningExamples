import modeling
from pytorch_model_summary import summary
import torch


if __name__ == "__main__":
    input_ids = torch.randint(0,30522,(4,128),device='cuda')
    token_type_ids = torch.randint(0,1,(4,128),device='cuda')
    attention_mask = torch.randint(0,1,(4,128),device='cuda')

    # Prepare model
    config = modeling.BertConfig.from_json_file("bert_config_large.json")

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
        print(config.vocab_size)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)
    model.to('cuda')
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    # # for n,p in param_optimizer:
    # #     if not any(nd in n for nd in no_decay):
    # #         print(n)


    summary(model,input_ids,token_type_ids,attention_mask,
    print_summary=True, max_depth=7,show_hierarchical=False,
    show_parent_layers=True,show_input=False)

