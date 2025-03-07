import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import SpongeBob
from Config import LLMConfig


def init_model(args):
     tokenizer = AutoTokenizer.from_pretrained('./spongebob_tokenizer')
     if args.model_mode == 1:
          ckp = f'./{args.save_dir}/SFT.pth'
     if args.model_mode == 0:
          ckp = f'./{args.save_dir}/pretrain.pth'
     if args.model_mode ==2:
          ckp = f'./{args.save_dir}/distill.pth'
     if args.model_mode ==3:
          ckp = f'./{args.save_dir}/SFT_final_eval_mini512.pth'
     if args.model_mode ==4:
          ckp = f'./{args.save_dir}/pretrain_step15999_eval_ep2_bs84_acu2.pth'
     if args.model_mode ==5:
          ckp = f'./{args.save_dir}/distill_long.pth'

     print(f'加载模型: {ckp}')

     model = SpongeBob(LLMConfig(
          max_seq_len=args.max_seq_len,
     ))

     state_dict = torch.load(ckp, map_location=args.device)
     model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
     # 排除mask，如果k中没有mask，则加载，否则不加载

     print(f'模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
     return model.eval().to(args.device), tokenizer




def main():
     parser=argparse.ArgumentParser()
     parser.add_argument('--save_dir', default='results', type=str)
     parser.add_argument('--temperature', default=0.7, type=float)
     parser.add_argument('--top_p', default=0.8, type=float)
     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
     parser.add_argument('--max_seq_len', default=8192, type=int)
     parser.add_argument('--history_cnt', default=0, type=int)
     parser.add_argument('--stream', default=True, type=bool)
     parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型")
     
     args = parser.parse_args()
     model, tokenizer = init_model(args)
     messages=[]
     while True:
          # 获取用户输入
          prompt = input('👶: ')  # 手动输入对话内容 👶 符号提示用户输入 input()函数接收用户输入
          # 保留最近N条对话历史（N=0时清空历史）
          messages = messages[-args.history_cnt:] if args.history_cnt else []   # -args.history_cnt: 从后往前取N条历史记录
          # [-args.history_cnt:]表示从后往前取N条历史记录 从前往后取的话是[:args.history_cnt]
          # 添加用户消息到对话历史
          messages.append({"role": "user", "content": prompt})

          # 准备模型输入（两种模式）
          # print('messages:', messages)
          new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True    # SFT模式                     # 预训练模式
          )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)
          # print('new_prompt:', new_prompt)

          # 模型推理
          with torch.no_grad():   # 关闭梯度计算
               x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)  # 添加batch维度
               outputs = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,   # 终止符号ID
                    max_new_tokens=args.max_seq_len,    # 最大生成长度
                    temperature=args.temperature,    # 温度参数
                    top_p=args.top_p,   # top-p采样参数 核心采样策略
                    stream=True,   # 流式生成
                    pad_token_id=tokenizer.pad_token_id,    # 填充符号ID
                    rp=1.3    # 重复惩罚参数
               )

               # 流式输出处理
               print('🤖️: ', end='')    # 🤖️ 表示机器人回复开始
               # end=''表示不换行
               try:
                    history_idx = 0     # 已输出文本长度标记
                    for y in outputs:
                         answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=False)   # 解码当前输出
                         if (answer and answer[-1] == '�') or not answer:  # 过滤无效字符
                              continue
                         print(answer[history_idx:], end='', flush=True)   # 逐字输出 在Python中， flush=True 参数用于强制立即清空输出缓冲区。
                         history_idx = len(answer)     # 更新已输出长度
               except StopIteration:
                    print("No answer")  # 异常处理
               print('\n')

          messages.append({"role": "assistant", "content": answer})   # 更新对话历史


if __name__ == "__main__":
    main()