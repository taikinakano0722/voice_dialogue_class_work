from env import Env
from utils import *
from agent import PPDPP
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig

from itertools import count
import argparse
from fastchat.model import add_model_args

cfg = {'bert': BertConfig, 'roberta': RobertaConfig}
tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}

llm_cache_dir = "cache/huggingface"
os.makedirs(llm_cache_dir, exist_ok=True)  # もしディレクトリがなければ作成
os.environ["HF_HOME"] = llm_cache_dir



def chat(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='human') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer) # policy network init

    # load policy parameters
    if args.sft_dir is not None: #事前学習済みのモデルをロード
        print('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)

    if args.load_rl_epoch > 0:#args.load_rl_epoch > 0 の場合、過去の強化学習モデルをロード
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
        
    state=env.reset()
    done = False
        
    for t in count():
        # user  dialog
        action = policy.select_action(state)
        state, reward, done = env.step(action) #doneは対話が終了したかどうか。stepのところで対話履歴stateの更新も行われている
        reward = torch.tensor([reward], device=args.device, dtype=torch.float)
        policy.rewards.append(reward)

        if done:
            print("対話は終了しました。お疲れ様です。")
            break
        
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus.')
   
    parser.add_argument('--data_name', type=str, default='cb', choices=['esc','cima','cb'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--system', type=str, default='chatgpt', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--critic', type=str, default='chatgpt', choices=['vicuna','chatgpt','llama2'], #報酬/終了判定のモデル
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--sft_dir', default='sft', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=8, help='max conversation turn')
    parser.add_argument('--load_rl_epoch', type=int, default=10, help='load agent from epoch')


    #parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    #parser.add_argument("--model_path", type=str, default="/storage_fast/ydeng/llm/vicuna_hf/7B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    add_model_args(parser)
    args = parser.parse_args()
    args.cache_dir=os.environ["HF_HOME"]
    args.model_path=f"{os.environ['HF_HOME']}/model" #これあとで存在じないときつくるやつ実装

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu' 
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    
    args.user="human"
    
    dataset = load_dataset(args.data_name)
    filename = '{}-{}-{}-{}-{}'.format(args.data_name,args.sft_dir,args.system,args.system,args.critic)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    
    if args.sft_dir:
        args.sft_dir = os.path.join(args.sft_dir, args.data_name, args.model_name, 'best_checkpoint')
    if not os.path.exists(args.sft_dir):
        print("no sft model, randomly initialize policy model")
        args.sft_dir = None

    chat(args, config, dataset, filename, tokenizer)