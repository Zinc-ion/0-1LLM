import os
import argparse
import time
import math
import torch
from torch import optim, nn
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model1111 import SpongeBob
from Config import LLMConfig
from dataset import PretrainDataset

# 获取学习率，根据当前步数和总步数动态调整
def get_lr(current_step, total_steps, lr, warmup_iters=0):
     if warmup_iters > 0 and current_step < warmup_iters:
          # warmup阶段：从 lr/10 线性上升到 1.1*lr
          warmup_lr = lr / 10 + lr * (current_step / warmup_iters)
          return warmup_lr
     else:
          # cosine退火阶段：调整步数减去warmup_iters
          adjusted_step = current_step - warmup_iters
          adjusted_total = total_steps - warmup_iters
          return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * adjusted_step / adjusted_total))
    # 最初current_step=0 返回的是1.1*lr，最后current_step=total_steps返回的是0.1*lr，实现学习率的cosine——decay


# 训练一个epoch
def train_epoch(epoch, wandb):
     loss_fct = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数，使用交叉熵损失
     # reduction='none' 表示不对损失进行求和，返回每个样本的损失值

     start_time = time.time()  # 记录训练开始时间

     # 遍历训练数据集
     for step, (X, Y, loss_mask) in enumerate(train_loader):
          X = X.to(args.device)  # 将输入数据移动到设备上（GPU/CPU）
          Y = Y.to(args.device)  # 将标签数据移动到设备上
          loss_mask = loss_mask.to(args.device)  # 将损失mask数据移动到设备上

          # 计算当前学习率
          lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                      args.learning_rate, args.warmup_iters)
          # 更新优化器的学习率
          for param_group in optimizer.param_groups:
               param_group['lr'] = lr

          # 在上下文管理器中进行前向传播和反向传播
          with ctx:
               res = model(X)  # 模型的前向传播
               loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),  # 计算损失
                    # view(-1, res.logits.size(-1))将logits展平，即将二维张量展平为一维张量，
                    # .logits表示
                    Y.view(-1)
               ).view(Y.size())
               loss = (loss * loss_mask).sum() / loss_mask.sum()  # 应用mask并计算平均损失
               loss = loss / args.accumulation_steps  # 进行梯度累积

          # 执行反向传播
          scaler.scale(loss).backward()
          # 每累积一定步数后进行优化器更新
          if (step + 1) % args.accumulation_steps == 0:
               scaler.unscale_(optimizer)  # 解除梯度缩放
               torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

               scaler.step(optimizer)  # 执行优化步骤
               scaler.update()  # 更新缩放器
               optimizer.zero_grad(set_to_none=True)  # 清空梯度

          # 每隔一定步数记录一次日志
          if step % args.log_step == 0:
               spend_time = time.time() - start_time  # 计算当前的时间
               print(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                         epoch + 1,  # 当前epoch
                         args.epochs,  # 总的epoch数
                         step,  # 当前步数
                         iter_per_epoch,  # 每个epoch的迭代次数
                         loss.item() * args.accumulation_steps,  # 损失值，乘以累积步数
                         optimizer.param_groups[-1]['lr'],  # 当前学习率
                         spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 估计剩余时间
                         )
               )
               if (wandb is not None):
                    wandb.log({"loss": loss.item() * args.accumulation_steps,
                               "lr": optimizer.param_groups[-1]['lr'],
                               "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

          # 每隔一定步数保存模型
          if (step + 1) % args.save_step == 0:
               model.eval()  # 设置模型为评估模式
               ckp = f'{args.save_dir}/pretrain_step{step}.pth'  # 保存模型的路径

               state_dict = model.state_dict()

               torch.save(state_dict, ckp)  # 保存模型
               model.train()  # 恢复为训练模式


# 初始化模型和分词器
def init_model(llm_config):
     tokenizer = AutoTokenizer.from_pretrained('spongebob_tokenizer')  # 加载分词器
     model = SpongeBob(llm_config).to(args.device)  # 初始化模型并移动到设备
     print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万 = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f} B')  # 打印模型参数量
     # p.numel()返回张量中元素的数量，p.requires_grad返回张量是否需要梯度
     # 1e6表示百万
     # p.numel() for p in model.parameters() if p.requires_grad 计算模型中需要梯度的参数数量
     return model, tokenizer


if __name__=='__main__':
     # 创建命令行参数解析器
     parser = argparse.ArgumentParser()
     # 这样用户就可以通过命令行参数来配置训练过程，而不需要修改代码，减少代码入侵

     # 添加各类参数，用于配置训练过程
     parser.add_argument("--save_dir", type=str, default="results")  # 保存结果的目录
     parser.add_argument("--epochs", type=int, default=1)  # 训练的轮数
     parser.add_argument("--batch_size", type=int, default=80)  # 每批次的样本数量
     parser.add_argument("--learning_rate", type=float, default=5e-4)  # 学习率
     parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
     # 设备类型，支持cuda或cpu
     parser.add_argument("--use_wandb", type=bool,default=True)  # 是否使用wandb进行日志记录
     parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型，默认使用bfloat16
     parser.add_argument("--wandb_project", type=str, default="SpongeBob-Pretrain")  # wandb项目名称
     parser.add_argument("--num_workers", type=int, default=1)  # 数据加载时的工作线程数
     parser.add_argument("--accumulation_steps", type=int, default=2)  # 梯度累积步数
     parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
     parser.add_argument("--warmup_iters", type=int, default=0)  # 学习率预热的迭代次数
     parser.add_argument("--log_step", type=int, default=10)  # 每多少步记录一次日志
     parser.add_argument("--save_step", type=int, default=1000)  # 每多少步保存一次模型
     parser.add_argument('--max_seq_len', default=512, type=int)  # 输入的最大序列长度
     parser.add_argument("--data_path", type=str, default="pretrain.jsonl")  # 训练数据的路径
     
     # 解析命令行参数
     args = parser.parse_args()  # parse_args()方法解析命令行参数并返回一个命名空间

     # 配置语言模型的参数
     lm_config = LLMConfig(max_seq_len=args.max_seq_len)
     # 只设置最大序列长度，其他参数使用默认值
     
     # 创建保存目录
     args.save_dir = os.path.join(args.save_dir)
     os.makedirs(args.save_dir, exist_ok=True)

     # 每次迭代处理的token数量
     tokens_per_iter = args.batch_size * lm_config.max_seq_len

     # 设置随机种子，保证训练可复现
     torch.manual_seed(1337)

     # 判断设备类型（cuda或cpu）
     device_type = "cuda" if "cuda" in args.device else "cpu"

     # 设置wandb运行名称
     args.wandb_run_name = f"SponseBob-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
     # 记录wandb运行名称，包括epoch、batch size和学习率
     
     # 根据设备类型选择自动混合精度（AMP）或不使用AMP
     ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

     # 如果使用wandb进行实验跟踪
     if args.use_wandb:
          import wandb

          # 初始化wandb项目
          wandb.init(project=args.wandb_project, name=args.wandb_run_name)
     else:
          wandb = None  # 如果不使用wandb，设置为None
     
     # 初始化模型和分词器
     model, tokenizer = init_model(lm_config)

     # 加载训练数据集
     train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)

     # 创建数据加载器
     # DataLoader是一个用于批量加载数据的类，可以自动对数据进行分批、打乱和多线程加载
     # dataset 和 dataloader的区别是，dataset是一个数据集，dataloader是一个数据加载器
     train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 是否将数据复制到CUDA内存
        drop_last=False,  # 不丢弃最后一批数据，如果为True，最后一批数据不足batch_size会被丢弃
        shuffle=False,  # 不对数据进行乱序，这样在训练时发现问题更容易定位
        num_workers=args.num_workers,  # 数据加载时使用的子线程数
     )

     # 初始化梯度缩放器（用于混合精度训练）
     scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))


     # 设置优化器
     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

     # 计算每个epoch中的迭代次数
     iter_per_epoch = len(train_loader)

     # 开始训练
     for epoch in range(args.epochs):
        # 调用训练函数开始训练每个epoch
        train_epoch(epoch, wandb)


     ########### 新增代码 ###########
     # 训练结束后将模型设置为评估模式
     model.eval()
     # 训练结束后保存最终模型
     final_ckp = os.path.join(args.save_dir, "pretrain_final.pth")
     torch.save(model.state_dict(), final_ckp)
     print(f"最终模型已保存至：{final_ckp}")

     # 关闭wandb运行
     if args.use_wandb:
          wandb.finish()
     ##############################


