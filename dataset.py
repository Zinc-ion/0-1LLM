import json
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

class PretrainDataset(Dataset):
    #  用于预训练的数据集类，继承自 Dataset 类
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        预训练数据集初始化
        
        参数:
          data_path: 数据文件路径，每行是一个JSON格式的样本
          tokenizer: 分词器，用于将文本转为 token ID
          max_length: 每个样本的最大长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据，返回一个样本列表，每个样本为字典格式
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        从文件中逐行读取数据，并解析为 JSON 对象
        
        参数:
          path: 数据文件路径
        
        返回:
          samples: 样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                # line.strip() 用于去除字符串首尾的空格和换行符 json.loads(line.strip()) 用于将字符串转为字典
                samples.append(data)
        return samples

    def __len__(self):
        # 返回样本总数
        return len(self.samples)

    def __getitem__(self, index):
        # 根据索引获取单个样本数据
        sample = self.samples[index]
        # 构建输入文本，加上起始符和结束符
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        #  f"  " 用于格式化字符串，将变量插入字符串中
        # str(sample['text']) 将文本转为字符串  sample['text']表示样本中的文本部分
        # sample格式为{'text': '文本内容'}
        # 利用 tokenizer 将文本编码为 token IDs，固定最大长度，进行 padding 和截断
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # 返回 PyTorch 张量
        )
        # 获取输入 token IDs，去除多余的维度
        input_ids = encoding.input_ids.squeeze()
        # 构造损失掩码，标记非填充位置
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        # (input_ids != self.tokenizer.pad_token_id) 返回一个布尔值张量，标记非填充位置，True 表示非填充位置

        # X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # 去掉最后一个token，即去掉结束符EOS，X是输入序列，即模型的输入内容，并行预测下一个token，所以去掉最后一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # 去掉第一个token，即去掉开始符BOS，Y是目标序列，即模型要预测的所有内容
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        # 对齐预测位置，去掉第一个token，即去掉开始符BOS
        return X, Y, loss_mask

class SFTDataset(Dataset):
    # 用于微调的数据集类，继承自 Dataset 类
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        微调数据集初始化
        
        参数:
          jsonl_path: 数据文件路径，每行是一个 JSON 格式的对话样本
          tokenizer: 分词器，用于将文本转为 token ID
          max_length: 每个样本的最大长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据，返回样本列表
        self.samples = self.load_data(jsonl_path)
        # 定义开始和结束标记的 token ID（不添加特殊 token）
        # <s>assistant\n 标记对话中AI回复的开始
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        # </s>\n 标记对话的结束
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        # 返回样本总数
        return len(self.samples)

    def load_data(self, path):
        """
        从文件中逐行读取数据，并解析为 JSON 对象
        
        参数:
          path: 数据文件路径
        
        返回:
          samples: 样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            # with open() as f: 用于打开文件，自动关闭文件
            for line_num, line in enumerate(f, 1):
                # 逐行解析JSON数据
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        构建符合 ChatML 格式的对话提示
        
        参数:
          conversations: 对话轮次列表，每个元素为包含 'content' 的字典
        
        返回:
          prompt: 拼接后的对话文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            # 偶数轮为用户，奇数轮为助手
            # 根据轮次交替设置用户和助手角色
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 使用分词器提供的模板方法构建对话提示
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 返回文本而不是token IDs
            add_generation_prompt=False  # 不添加生成提示
        )

    def _generate_loss_mask(self, input_ids):
        """
        根据输入 token IDs 生成损失掩码。
        只有位于 <s>assistant\n 与 </s>\n 之间的 token 被标记为 1，其余为 0
        
        参数:
          input_ids: 一个整数列表，表示输入 token IDs
        
        返回:
          loss_mask: 一个与 input_ids 长度相同的列表，1 表示计算损失的位置，0 表示忽略
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        # 遍历整个输入序列
        while i < len(input_ids):
            # 检查当前位置是否匹配开始标记
            # 检查当前位置是否是assistant回复的开始（匹配bos_id）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 记录开始位置，排除 bos 部分
                # 计算回复内容的起始位置（跳过bos标记）
                start = i + len(self.bos_id)
                end = start
                # 从开始位置向后查找结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                    # +1是为了包含当前token的下一个预测位置
                # 将开始标记之后到结束标记位置之间的 token 标记为 1（参与损失计算）
                # 设置回复内容部分的loss_mask为1（包括eos标记）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 更新索引：跳过整个对话部分（包括结束标记）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
                获取单个训练样本，格式化为模型输入

                返回:
                    X: tensor - 输入token序列（前n-1个token）
                    Y: tensor - 目标token序列（后n-1个token）
                    loss_mask: tensor - 指示哪些位置需要计算loss
                """
        # 获取对应索引的样本
        sample = self.samples[index]
        # 利用对话轮次构建对话提示
        # 1. 构建对话格式文本
        prompt = self._create_chat_prompt(sample['conversations'])
        # 对对话提示进行编码，并限制最大长度
        # 2. 将文本转换为token IDs
        # 注意：不同分词器的返回值可能不同，这里假设返回包含input_ids的字典
        # input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        encoded = self.tokenizer(prompt)
        input_ids = encoded['input_ids'][:self.max_length]  # 截断到最大长度

        # 若不足最大长度则补齐 pad_token_id
        # input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 3. 填充到最大长度
        pad_num = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_num

        # 根据输入 token IDs 生成动态的损失掩码
        # 4. 生成动态loss mask（只对assistant回复部分计算loss）
        loss_mask = self._generate_loss_mask(input_ids)

        # 5. 构造训练数据（预测下一个token）
        # 构建训练数据：X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        # 输入序列：前n-1个token [0,1,...,n-1]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # 目标序列：后n-1个token [1,2,...,n]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # 损失掩码对齐目标序列长度
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask
