# [文件名]: SFT.py
# [功能]: 监督微调(SFT)主训练脚本，用于对大语言模型进行指令微调

import os
import argparse
import time
import math
import torch
from torch import optim, nn
from contextlib import nullcontext  # 用于创建空上下文管理器
from transformers import AutoTokenizer  # 自动加载分词器
from torch.utils.data import DataLoader
from model import SpongeBob  # 自定义模型类
from Config import LLMConfig  # 模型配置类
from dataset import SFTDataset  # 监督微调数据集类


def get_lr(current_step, total_steps, lr, warmup_iters=0):
     """
     动态学习率计算函数（余弦退火 + 预热）

     参数:
         current_step: 当前训练步数
         total_steps: 总训练步数（epochs * 每epoch迭代次数）
         lr: 基础学习率
         warmup_iters: 学习率预热步数

     返回:
         当前步骤的学习率值
     """
     # 预热阶段：线性增加学习率
     if current_step < warmup_iters:
          return lr * current_step / warmup_iters
     # 余弦退火阶段
     return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
     """
     单个训练epoch的执行逻辑

     参数:
         epoch: 当前epoch序号
         wandb: Weights & Biases日志对象
     """
     model.train()  # 确保模型处于训练模式
     loss_fct = nn.CrossEntropyLoss(reduction='none')  # 使用none reduction以便应用loss mask
     start_time = time.time()

     for step, (X, Y, loss_mask) in enumerate(train_loader):
          # 数据转移到指定设备
          X = X.to(args.device)
          Y = Y.to(args.device)
          loss_mask = loss_mask.to(args.device)

          # 动态调整学习率
          lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                      args.learning_rate, args.warmup_iters)
          for param_group in optimizer.param_groups:
               param_group['lr'] = lr

          # 混合精度训练上下文（如果启用）
          with ctx:
               # 前向传播
               res = model(X)
               # 计算原始损失（未应用mask）
               loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),  # res.logits形状是[batch_size, seq_len, vocab_size]
                    Y.view(-1)
               ).view(Y.size())
               # 应用loss mask并归一化
               loss = (loss * loss_mask).sum() / loss_mask.sum()
               # 梯度累积：按累积步数缩放损失
               loss = loss / args.accumulation_steps

          # 反向传播（自动处理混合精度缩放）
          scaler.scale(loss).backward()

          # 梯度累积达到指定步数时执行参数更新
          if (step + 1) % args.accumulation_steps == 0:
               # 取消梯度缩放以进行梯度裁剪
               scaler.unscale_(optimizer)
               # 梯度裁剪防止爆炸
               torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

               # 执行参数更新
               scaler.step(optimizer)
               scaler.update()
               optimizer.zero_grad(set_to_none=True)  # 更高效地清空梯度

          # 定期打印训练日志
          if step % args.log_step == 0:
               spend_time = time.time() - start_time
               print(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                         epoch + 1,
                         args.epochs,
                         step,
                         iter_per_epoch,
                         loss.item() * args.accumulation_steps,  # 还原真实损失值
                         optimizer.param_groups[-1]['lr'],
                         spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60)
               )
               # 记录到WandB
               if (wandb is not None):
                    wandb.log({
                         "loss": loss.item() * args.accumulation_steps,
                         "lr": optimizer.param_groups[-1]['lr'],
                         "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                    })

          # 定期保存模型检查点
          if (step + 1) % args.save_step == 0:
               model.eval()
               ckp = f'{args.save_dir}/SFT_512full.pth'

               # 处理分布式训练的情况
               if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
               else:
                    state_dict = model.state_dict()

               torch.save(state_dict, ckp)
               model.train()


def init_model(lm_config):
     """
     初始化模型和分词器

     参数:
         lm_config: 模型配置对象

     返回:
         model: 初始化后的模型
         tokenizer: 分词器对象
     """
     # 加载预训练分词器
     tokenizer = AutoTokenizer.from_pretrained('spongebob_tokenizer')
     # 初始化模型结构
     model = SpongeBob(lm_config)
     # 加载预训练权重
     ckp = f'./results/pretrain_final_train_ep2_bs84_acus2.pth'  # 预训练模型路径 pretrain.pth没有用GQA，使用的是传统MHA
     state_dict = torch.load(ckp, map_location=args.device)
     model.load_state_dict(state_dict, strict=False)  # 允许部分权重不匹配
     # 打印模型参数量
     print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
     model = model.to(args.device)
     return model, tokenizer


if __name__ == "__main__":
     # ------------------ 参数解析 ------------------
     parser = argparse.ArgumentParser(description="监督微调训练脚本")
     # 训练配置
     parser.add_argument("--save_dir", type=str, default="results", help="模型保存目录")
     parser.add_argument("--epochs", type=int, default=1, help="训练总轮次")
     parser.add_argument("--batch_size", type=int, default=84, help="批次大小")
     parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
     parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                         help="训练设备，默认使用GPU")
     # 日志与监控
     parser.add_argument("--use_wandb", type=bool, default=True, help="是否使用WandB记录日志")
     parser.add_argument("--wandb_project", type=str, default="SpongeBob-SFT", help="WandB项目名称")
     # 训练优化
     parser.add_argument("--dtype", type=str, default="bfloat16",
                         help="混合精度类型，可选float16或bfloat16")
     parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
     parser.add_argument("--accumulation_steps", type=int, default=2,
                         help="梯度累积步数")
     parser.add_argument("--grad_clip", type=float, default=1.0,
                         help="梯度裁剪阈值")
     parser.add_argument("--warmup_iters", type=int, default=1700,
                         help="学习率预热步数")
     # 日志与保存间隔
     parser.add_argument("--log_step", type=int, default=10,
                         help="每隔多少步打印日志")
     parser.add_argument("--save_step", type=int, default=1000,
                         help="每隔多少步保存模型")
     # 数据配置
     parser.add_argument('--max_seq_len', default=512, type=int,
                         help="输入序列最大长度")
     parser.add_argument("--data_path", type=str, default="sft_512.jsonl",
                         help="微调数据路径")
     args = parser.parse_args()

     # ------------------ 初始化配置 ------------------
     # 模型配置
     lm_config = LLMConfig(max_seq_len=args.max_seq_len)
     # 创建保存目录
     args.save_dir = os.path.join(args.save_dir)
     os.makedirs(args.save_dir, exist_ok=True)
     # 计算每迭代处理的token数（用于日志）
     tokens_per_iter = args.batch_size * lm_config.max_seq_len
     # 设置随机种子
     torch.manual_seed(1337)
     # 设备类型判断（CUDA/CPU）
     device_type = "cuda" if "cuda" in args.device else "cpu"

     # ------------------ 训练准备 ------------------
     # 初始化WandB运行（如果启用）
     args.wandb_run_name = f"SpongeBob-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
     # 混合精度训练上下文
     ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

     # 初始化WandB
     if args.use_wandb:
          import wandb

          wandb.init(project=args.wandb_project, name=args.wandb_run_name)
     else:
          wandb = None

     # ------------------ 模型与数据加载 ------------------
     # 初始化模型和分词器
     model, tokenizer = init_model(lm_config)
     # 创建数据集和数据加载器
     train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
     train_loader = DataLoader(
          train_ds,
          batch_size=args.batch_size,
          pin_memory=True,  # 加速GPU数据传输
          drop_last=False,  # 保留不完整批次
          shuffle=False,  # 数据是否打乱（可改为True）
          num_workers=args.num_workers,
     )

     # ------------------ 优化器配置 ------------------
     # 混合精度梯度缩放器
     scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
     # 使用AdamW优化器
     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

     # ------------------ 训练循环 ------------------
     iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
     for epoch in range(args.epochs):
          train_epoch(epoch, wandb)  # 执行单个epoch训练


     ########### 新增代码 ###########
     # 训练结束后将模型设置为评估模式
     model.eval()
     # 训练结束后保存最终模型
     final_ckp = os.path.join(args.save_dir, "SFT_final.pth")
     torch.save(model.state_dict(), final_ckp)
     print(f"最终模型已保存至：{final_ckp}")

     # 关闭wandb运行
     if args.use_wandb:
          wandb.finish()
     ##############################