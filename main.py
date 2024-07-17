# -*- coding: utf-8 -*-

!pip install git+https://github.com/openai/CLIP.git

#Kaggle用にパスを変更
root_path_s = '/kaggle/input/vqacompe'
train_df_path_s = root_path_s + '/data/train.json'
train_image_dir_s = root_path_s + '/data/train/train'
test_df_path_s =  root_path_s + '/data/valid.json'
test_image_dir_s = root_path_s + '/data/valid/valid'

# CLIP モデルの設定
CLIP_MODEL_NAME = "ViT-B/32"

#     train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
#     test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)

import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import clip
from tqdm import tqdm
import torch.nn.functional as F


import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 損失関数の定義
def KLDivLoss(output, target):
    log_output = torch.log(torch.clamp(output, min=1e-9))  # ゼロで割らないように
    loss = F.kl_div(log_output, target, reduction='batchmean')
    return loss

def new_process_text(inText):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                    "youll": "you'll", "youre": "you're", "youve": "you've"}

    manualMap    = { 'none': '0',
                     'zero': '0',
                     'one': '1',
                     'two': '2',
                     'three': '3',
                     'four': '4',
                     'five': '5',
                     'six': '6',
                     'seven': '7',
                     'eight': '8',
                     'nine': '9',
                     'ten': '10'
                   }

    articles     = ['a',
                    'an',
                    'the'
                   ]

    periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip   = re.compile("(\d)(\,)(\d)")
    punct        = [';', r"/", '[', ']', '"', '{', '}',
                    '(', ')', '=', '+', '\\', '_', '-',
                    '>', '<', '@', '`', ',', '?', '!']

    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)

    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        #ansewrがTrueかどうかで、ansewrを読み込むかを分岐させている
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer #答えに関する処理を行うかどうか。True/False

        #答えとなるマッピングの読み込み
        #map_csv_path = root_path_s + '/data_annotations_class_mapping.csv'
        #new_answer4は、mappingにオリジナルのmode answerをプラス（単なるnew_answerはオリジナルの全ての単語を追加したものだが多すぎた？）
        #new_answer9は、mappingにオリジナルのコンペの短縮系や数字の除去処理を加えて、作成しなおしたもの。
        #オリジナルは、https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py
        map_csv_path = root_path_s + '/new_answer9.csv'

        answer_map = pandas.read_csv(map_csv_path)
        self.answer2idx = dict(zip(answer_map["answer"], answer_map["class_id"]))
        self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = new_process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
                        self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        print('now answer2idx の length is'+str(len(self.answer2idx)))
        print("Answer mapping:", list(self.answer2idx.items())[:10])

        # CLIP のテキスト変換用のトークナイザーを作成これは学習される…？
        self.clip_model, _ = clip.load(CLIP_MODEL_NAME, device=torch.device("cuda"))

    #datasetで与えらえた辞書で、自分の辞書を更新す．
    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    # get_item--対応するidxのデータ（画像，質問，回答）を取得．
    #    Parameters
    #    ----------
    #    idx : int
    #        取得するデータのインデックス
        """
        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : 質問文のそのもの str
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答の（辞書の？）id
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のインデックス
        """

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image) # 画像の前処理の実行
        #question = self.df["question"][idx] #questionはそのままstrで渡す。後でBERT処理される
        #CLIPはここでトークナイザ処理を行う。（BERTとちょっと違う）
        question = clip.tokenize([self.df["question"][idx]], truncate=True).squeeze(0)

        if self.answer: #mode_answer_idx（最頻値の答え…のインデックス）を作成
            answers = [self.answer2idx[new_process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, question, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, question

    def __len__(self):
        return len(self.df)




# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)




# 3. モデルのの実装
# ResNetを利用できるようにしておく
# ResNetよりCLIPの変換の方が、親和性が高いはず！…ということでResnet関係は削除。

from transformers import BertModel, BertTokenizer

class VQAModel(nn.Module):
#    def __init__(self, vocab_size, num_answers: int):
        #vocab_sizeは不要っぽい
    def __init__(self,  num_answers: int):
        super().__init__()

        #ここはBERTと同じ。モデルを作って、パラメタは学習させない。
        self.clip_model, _ = clip.load(CLIP_MODEL_NAME, device=torch.device("cuda"))

        for param in self.clip_model.parameters():
            param.requires_grad = False

        #画像の変換器を作る
        image_feature_dim = self.clip_model.visual.output_dim
        text_feature_dim = self.clip_model.text_projection.shape[1]
        text_out_dim = 1024
        dim2 = 2048
        dim3 = 1024
        #fc:full connect（全結合）
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + text_out_dim, dim2),  # 画像特徴量とテキスト特徴量の結合, #BERTのときは512(画像)+768（BERT）
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),  #0.5->0.3->0.5
            nn.Linear(dim2,dim3),
            nn.BatchNorm1d(dim3),
            nn.Dropout(0.1),  #0.3->0.1->0.2
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim3, num_answers),  # 正答は埋め込み処理されたものとなるよう、回答の次元数#BERTの出力次元数７６８にする。これと同じ次元数に出力
        )

        self.text_processor = nn.Sequential(
            nn.Linear(text_feature_dim, text_out_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, image, question):
        image = image.to(torch.float32)
        #question = question.to(torch.int32)
        question = question.to(torch.long)
        image_features = self.clip_model.encode_image(image).to(torch.float32)

        text_features = self.clip_model.encode_text(question).to(torch.float32)
        text_features = self.text_processor(text_features)

        combined_features = torch.cat([image_features, text_features], dim=1)

        output = self.fc(combined_features)

        return output
        #return torch.nn.functional.softmax(output, dim=1)  # 負の数にならないようにsoftmaxを入れてみる



# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0
    #動作検査のための変数temp
    tempkey =0
    tempc=0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        #print(f"image: {image.shape}, question: {question}, answers: {answers}, mode_answer: {mode_answer}")

        tempc = tempc +1
        if tempc % 100 == 1:
            print('train関数で処理した回数: ',tempc)
        image = image.to(device)
        question = question.to(device)
        answers = answers.to(device)
        mode_answer = mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer)
        #loss = criterion(pred, F.one_hot(mode_answer, num_classes=pred.size(1)).float()).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tempc % 200 == 1:
            print('train関数で処理した回数: ',tempc)
            print(f"pred: {pred}, loss: {loss.item()}")

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        #simple_acc += (pred.argmax(1) == mode_answer.argmax(1)).float().mean().item()  # simple accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()


    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start



def eval(model, dataloader, criterion, device):
    model.eval()
    print('now eval is performed')
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)


        with torch.no_grad():
            pred = model(image, question) #ここに、to(device)がなくても、引数やmodelがto(device)なので、同じ？
            loss = criterion(pred, mode_answer)
       #     loss = criterion(pred, F.one_hot(mode_answer, num_classes=pred.size(1)).float()).to(device)
            print(f"pred: {pred}, loss: {loss.item()}, answers: {answers}, mode_answer: {mode_answer}") #デバッグ用コードの追加
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()    # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start



import datetime
import pytz

def main():
    # deviceの設定
    set_seed(3407)
    device = torch.device("cuda")
    print('cuda is available!!')if torch.cuda.is_available() else print("cpu only....")

    # dataloader / model
    #左右反転・上下反転は文字が読めなくなるので外す。色の変化も、色を回答する可能性があるので、弱めにする。
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # ランダムに最大１０度回転（ちょいよわ）
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 色変換の範囲を制限->スコア少し悪くなる
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), #ImageNetの平均値で正規化
    ])

    train_dataset = VQADataset(df_path=train_df_path_s, image_dir=train_image_dir_s, transform=transform, answer = True)
    test_dataset = VQADataset(df_path=test_df_path_s, image_dir=test_image_dir_s, transform=transform, answer=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=40,
        num_workers = 4,
        pin_memory = True,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers = 4,
        pin_memory = True,
        shuffle=False
    )

    #model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    #model = VQAModel().to(device)

    model = VQAModel(num_answers=len(train_dataset.answer2idx)).to(device)


# 以前の学習で保存されたモデルをロードする
#    model.load_state_dict(torch.load("/kaggle/working/model.pth"))
#    model.to(device)


# optimizer / criterion
    num_epoch = 10

#評価関数
    #criterion = nn.MSELoss().to(device) #VQAタスクの出力形式やモデルの設計を変更して、連続的な予測スコアになったときは、平均二乗誤差（MSE）がより適している
    criterion = nn.CrossEntropyLoss() #離散値ならクロスエントロピー
    #criterion = KLDivLoss #別に定義した関数KLダイバージェンス

#    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 , weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 , weight_decay=1e-5)
    # train model
    for epoch in range(num_epoch):
        t_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        print('epoch_'+str(epoch)+' is starting on '+str(t_now))

        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)

        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        print('-----------------------')
        # 中間作成提出用ファイルの作成
        if epoch % 1 == 0:
            model.eval()
            submission = []
            print('submission.csv will be made　in the middle.')
            for image, question in test_loader:
                image, question = image.to(device), question.to(device)
                with torch.no_grad():
                    pred = model(image, question)  #predにはBERTの分散表現が入っている？→とりあえずマッピング
                    pred = pred.argmax(1).cpu().item()
                submission.append(pred)

            submission = [train_dataset.idx2answer[id] for id in submission]
            submission = np.array(submission)
            submission_file_name = "submission_ep"+str(epoch)+".npy"
            np.save(submission_file_name, submission)
            print('submission.npy have been made in the middle.')


    # 提出用ファイルの作成
#     model.eval()
#     submission = []
#     print('submission.csv will be made.')
#     for image, question in test_loader:
#         image, question = image.to(device), question.to(device)
#         with torch.no_grad():
#             pred = model(image, question)  #predにはBERTの分散表現が入っている？→とりあえずマッピング
#             pred = pred.argmax(1).cpu().item()
#         submission.append(pred)

#     submission = [train_dataset.idx2answer[id] for id in submission]
#     submission = np.array(submission)
#     np.save("submission.npy", submission)
#     print('submission.npy have been made!')

    print('save the model.state.dict()')
    torch.save(model.state_dict(), "model.pth")
#    torch.save(model.state_dict(), "model2.pth")

main()
print('finished!')