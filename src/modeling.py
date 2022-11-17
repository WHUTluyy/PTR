import torch
import torch.nn as nn
from arguments import get_model_classes, get_args

class Model(torch.nn.Module):

    def __init__(self, args, tokenizer = None, prompt_label_idx = None):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type] #从参数args里获取robert/bert/albert
        """
        prompt_label_idx是下列5个list的token
        第一个mask对应3种可能的token: [person, organization, entity, …]
        第二个mask对应3种可能的token: ['s , is , was]
        第三个mask对应24种可能的token: [died, founded, ...]
        第四个mask对应9种可能的token: [of, by, in ...]
        第五个mask对应12中可能的token: [state, person, ...]
        """
        self.prompt_label_idx = prompt_label_idx
        # 从预训练语言模型里加载到self.model
        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.mlp = torch.nn.Sequential(
            # hidden_size--->hidden_size
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            # hidden_size--->hidden_size
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            )
        #额外的词嵌入向量 new_tokens=5  每个向量大小为hidden_size
        self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, input_flags, mlm_labels, labels):
        # 用初始的模型将input_ids进行嵌入
        raw_embeddings = self.model.embeddings.word_embeddings(input_ids) # [batch_size,512,1024]
        # 额外n个token嵌入层的权重
        '''
        new_token_embeddings [n,1024]
        tensor([[-0.2188,  0.1157, -0.1663,  ...,  0.3989, -0.2228,  0.1165],
        [-0.0849, -0.1754, -0.1862,  ...,  0.0822, -0.2826, -0.5738],
        [-0.0751,  0.0890, -0.1469,  ...,  0.2220, -0.1952, -0.0569],
        ...,
        [ 0.0172,  0.1000, -0.2583,  ...,  0.1256, -0.1615, -0.0834],
        [-0.0681,  0.2455, -0.0501,  ..., -0.0759, -0.3531, -0.0673],
        [ 0.0982,  0.0266, -0.1607,  ...,  0.1843, -0.1310, -0.2256]]
        '''
        new_token_embeddings = self.mlp(self.extra_token_embeddings.weight) # [n,1024]
        # 新的input的嵌入
        new_embeddings = new_token_embeddings[input_flags] # [batch_size,512,1024]
        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings, raw_embeddings) # [batch_size,512,1024]
        # 输入经过原始model 如robert
        hidden_states, _ = self.model(inputs_embeds=inputs_embeds, # [batch_size,512,1024]
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        # mlm_labels是掩码标签，去得到robert中每个掩码的预测值
        hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), len(self.prompt_label_idx), -1) #展开成[batch_size,5,1024]
        logits = [
            torch.mm(
                hidden_states[:,index,:], 
                self.model.embeddings.word_embeddings.weight[i].transpose(1,0)
            )
            for index, i in enumerate(self.prompt_label_idx)
        ]
        return logits

def get_model(tokenizer, prompt_label_idx):
    args = get_args()
    model = Model(args, tokenizer, prompt_label_idx)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model

def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer

