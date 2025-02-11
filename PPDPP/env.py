import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template

import openai

from dotenv import load_dotenv
from utils import *
from prompt import *
#from unidecode import unidecode
import nltk
import re
import time

nltk.download('punkt_tab')

system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages, 'cima': CIMAMessages, 'cb': CBMessages}

load_dotenv()  # .env ファイルを読み込む


class Env(object):
    """
    対話環境を定義する
    """
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
            if mode == 'train':
                self.vicuna_model, self.vicuna_tokenizer = load_model(
                    args.model_path,
                    args.device,
                    args.num_gpus,
                    args.max_gpu_memory,
                    args.load_8bit,
                    args.cpu_offloading,
                    debug=args.debug,
                )
            else:
                self.vicuna_model = env_model
                self.vicuna_tokenizer = env_tokenizer
        
        api_key = os.getenv("OPENAI_API_KEY")  # 環境変数からAPIキーを取得
        self.client=openai.OpenAI(api_key=api_key)
        self.args = args
        if mode=="human":
            self.dataset=dataset["train"]+dataset["test"]
        else:
            self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode

        self.reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0,
            },
            'cima': {
                'incorrect': -1.0,
                'did not': -0.5,
                'part': 0.5,
                'whole': 1.0,
            },
        }

        set_random_seed(args.seed)

        
    def reset(self):
        self.cur_conver_step = 0
        if self.mode == 'train' or self.mode=="human":
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        if self.args.data_name == 'esc':
            self.conversation = [{"role":"Patient", "content":self.case['situation']}]
        elif self.args.data_name == 'cima':
            self.conversation = [{"role":"Teacher", "content":self.case['dialog'][0]['text']}, {"role":"Student", "content":self.case['dialog'][1]['text']}]
        elif self.args.data_name == 'cb':
            self.conversation = [{"role":"Buyer", "content":"Hi, how much is the %s?" % self.case['item_name']}, {"role":"Seller", "content":"Hi, this is a good %s and its price is %s." % (self.case['item_name'], self.case['seller_price'])}]
            if self.mode=="human":
                print(f"あなたは売り手(seller)です。相手に{self.case['item_name']}をできるだけ高く売ってください。目標金額は{self.case['seller_price']}です。(英語のみ対応)\n")
        if self.mode=="human":
            print(f"買い手(相手):{self.conversation[0]['content']}\n売りて(あなた):{self.conversation[1]['content']}")
        else:
            print(self.conversation)
        return self.conversation


    def step(self, action):
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        
        if self.mode!="human":
            print(action)
        #システムの発話を生成
        messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action)
        response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
        response = self.postprocess_response(response, user_role[self.args.data_name])
        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        if self.mode=="human":
            print(f"買い手(相手):{response}\n")
        else:
            print(self.conversation[-1])

        #ユーザーの発話を生成(強化学習のときの仮想的なユーザー側の発話)
        if self.args.user=="human":
            user_response=input("あなたの発話ターンです")
        else:
            messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
            user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
            user_response = self.postprocess_response(user_response, system_role[self.args.data_name])
        self.conversation.append({"role":user_role[self.args.data_name], "content":user_response})
        if self.mode=="human":
            print(f"売りて(自分):{user_response}\n")
        else:
            print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        reward = self.compute_reward(self.args.critic, messages, self.case)

        if self.args.data_name == 'esc':
            if reward > 0.5:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cima':
            if reward == 1:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cb':
            if reward >= 0:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
                
        self.cur_conver_step += 1
        return self.conversation, reward, done
    
    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response(self, model, messages, role):
        if self.mode == 'test':
            temperature = 0
        else:
            temperature = 0.7
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, role)
            #print(messages)
            output = query_openai_model(
                client=self.client,
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=self.args.max_new_tokens,
                temperature=temperature
            )
        return output
    
    def compute_reward(self, model, messages, case):
        """
        報酬を計算する関数。
        論文の中にあったRewardLLMに対応する概念がここに登場する
        """
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_openai_model(
                client=self.client,
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=10
            )
        
        if self.args.data_name in ['esc','cima']:
            rewards = []
            if self.mode!="human":
                print(outputs)
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
            print(reward)
        elif self.args.data_name == 'cb':
            deals = []
            rewards = []
            human_score=0
            if self.mode!="human":
                print(outputs)
            for output in outputs:
                if 'have not' in output.lower():
                    deals.append(-1)
                elif 'have reached' in output.lower():
                    deals.append(1)
                
                prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
                if len(prices) > 0:
                    deal_price = float(prices[0])
                    reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price']) #cbデータセットでは売った値段によって報酬が変わる
                    rewards.append(reward)
                    if self.mode=="human":
                        human_score=(deal_price - case['buyer_price']) / (case['seller_price'] - case['buyer_price'])
                

            if -1 in deals:
                reward = -0.1
            else:
                if len(rewards) == 0:
                    reward = 0
                else:
                    reward = max(set(rewards), key = rewards.count)
            if self.mode=="human":
                print(f"あなたの点数:{human_score}\n")   
            else:    
                print(reward)
            

        return reward



def query_openai_model(client, messages: str, model: str = "gpt-3.5-turbo", max_tokens: int = 128, temperature: float = 0, n: int = 1):
    flag = True
    while flag:
        """
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=n,
                stop=None,
                temperature=temperature,
                request_timeout=10,
            )

            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
        """
        completions = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
        )

        if n == 1:
            output = completions.choices[0].message.content.strip()
        else:
            output = []
            for choice in completions.choices:
                output.append(choice.message.content.strip())

        flag = False
    return output