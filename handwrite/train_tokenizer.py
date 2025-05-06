import json
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders
)

import os

def train_tokenizer(): # 训练分词器
    def read_texts_from_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]

    data_path = "pretrain.jsonl"  # 数据文件路径
    tokenizer = Tokenizer(models.BPE())  # 创建BPE分词器
    tokenizer.per_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 设置预分词器
    # add_prefix_space=False表示不添加前缀空格

    special_tokens = ["<ukn>", "<s>", "</s>"]  # 特殊符号

    trainer = trainers.BpeTrainer(
        vocab_size=6400,  # 词表大小
        special_tokens=special_tokens,  # 特殊符号
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 初始化字母表
    )

    texts = read_texts_from_jsonl(data_path)  # 读取文本数据

    tokenizer.train_from_iterator(texts, trainer=trainer)  # 训练分词器

    tokenizer.decoder = decoders.ByteLevel()  # 设置解码器

    # 确保特殊符号的token_id设置正确
    assert tokenizer.token_to_id("<ukn>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    tokenizer_dir = "spongebob_tokenizer"  # 保存分词器的目录
    os.makedirs(tokenizer_dir, exist_ok=True)  # 创建目录
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))  # 保存分词器
    # os.path.join() 方法用于拼接文件路径
    # tokenizer.save() 方法用于保存分词器到指定路径
    tokenizer.model.save("./spongebob_tokenizer")  # 保存分词器模型
    # tokenizer.model.save() 方法用于保存分词器模型到指定路径

    config = {
        "add_prefix_space": False,  # 是否添加前缀空格
        "add_eos_token": False,
        "add_bos_token": False,
        "added_tokens_decoder" : {
            "0": {
                "content": "<ukn>",
                "special": True,
                "lstrip": False, # 是否去除前缀空格
                "rstrip": False, # 是否去除后缀空格
                "single_word": False, # 是否单词
            },
            "1": {
                "content": "<s>",  # 句子开始符号
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",  # 句子结束符号
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "addtional_special_tokens": [],
        "bos_token": "<s>",  # 句子开始符号
        "eos_token": "</s>",  # 句子结束符号
        "clean_up_tokenization_spaces": False,  # 是否清除分词器的空格
        "legacy": True,  # 是否使用旧版分词器
        "model_max_length": 32768,  # 模型最大长度
        "pad_token": "<ukn>",  # 填充符号
        "sp_model+kwargs": {},
        "spaces_between_special_tokens": False,  # 是否在特殊符号之间添加空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # 分词器类
        "unk_token": "<unk>",  # 未知符号
        "chat_template": "{% if messages[0]['role'] == 'system' %}"
                         "{% set system_message = messages[0]['content'] %}"
                         "{{ '<s>system\\n' + system_message + '</s>\\n' }}"
                         "{% else %}"
                         "{{ '<s>system\\n你是 SpongeBob，是一个有用的人工智能助手。</s>\\n' }}"
                         "{% endif %}"
                         "{% for message in messages %}"
                         "{% set content = message['content'] %}"
                         "{% if message['role'] == 'user' %}"
                         "{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}"
                         "{% elif message['role'] == 'assistant' %}"
                         "{{ content + '</s>' + '\\n' }}"
                         "{% endif %}"
                         "{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)  # 保存配置文件
        # ensure_ascii=False表示不转义非ASCII字符
        # indent=4表示缩进4个空格

if __name__ == "__main__":
    train_tokenizer()  # 调用训练分词器函数