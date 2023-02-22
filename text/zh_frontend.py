
from text.frontend.zh_frontend import Frontend
from text.symbols import word_boundary_symbol

frontend = Frontend()


pu_symbols = ['!', '?', '…', ",", "."]

# print(_symbol_to_id)
pinyin2phones = {}

for line in open("text/zh_dict.dict").readlines():
    pinyin, phones = line.strip().split("\t")
    phones = phones.split(" ")
    pinyin2phones[pinyin] = phones

def pu_symbol_replace(data):
    chinaTab = ['！', '？', "…", "，", "。",'、', "..."]
    englishTab = ['!', '?', "…", ",", ".",",", "…"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

# def del_special_pu(data):
#     ret = ''
#     to_del = ["'", "\"", "“","", '‘', "’", "”"]
#     for i in data:
#         if i not in to_del:
#             ret+=i
#     return ret


def zh_to_phonemes(text, add_word_boundary=False):
    # 替换标点为英文标点
    text = pu_symbol_replace(text)
    phones = frontend.get_phonemes(text, add_word_boundary=add_word_boundary)[0]
    return phones


def pinyin_to_phonemes(text, add_word_boundary=False):
    phones = []
    for pinyin in text.split(" "):
        try:
            if add_word_boundary:
                phones.append(word_boundary_symbol)
            phones += pinyin2phones[pinyin]
        except:
            print("词典中无此拼音：", pinyin)
    return phones