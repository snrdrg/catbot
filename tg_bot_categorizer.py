#default telegram daemon bot
import numpy as np
import json
import sqlite3
import asyncio
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import BotCommand
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup
from aiogram.types import KeyboardButton as kb
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from multiprocessing import Process
from functools import wraps, partial

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from evaluate import load

print('Loading API keys')
with open('api_key.json', 'r') as f:
   API_TOKEN = json.load(f)["key"]

print('Loading configurations')
with open('bot_messages.json', 'r', encoding='utf-8') as f:
   bot_messages = json.load(f)
print('Connecting to database')
conn = sqlite3.connect('cats.db')
print('Complete')

WELCOME_STR = bot_messages['welcome']
HELP_STR = bot_messages['help']#'Go help yoursef'

 
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

labels = ['Мошенничество','Защита периметра','Нормативная документация','Общие вопросы по кибербезопасности']
keys_row1 = [
    InlineKeyboardButton(labels[0], callback_data='btn_0'),
    InlineKeyboardButton(labels[1], callback_data='btn_1')]
keys_row2 = [
    InlineKeyboardButton(labels[2], callback_data='btn_2'),
    InlineKeyboardButton(labels[3], callback_data='btn_3')]

inline_kb = InlineKeyboardMarkup().row(*keys_row1).row(*keys_row2)

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational')


print('Loading model')

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained('./model_weights_bert')
model.to(device)
print('Complete')

async def setup_bot_commands(dp):
    await dp.bot.set_my_commands([
        BotCommand(command="/start", description="Start bot"),
        BotCommand(command="/help", description="Show help")
    ])


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
   await message.reply(WELCOME_STR)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
   await message.reply(HELP_STR)
   

@dp.message_handler()
async def echo(message: types.Message):
   res = predict_class([message.text], model, tokenizer)
   message.bot['user_input'] = message.text
   message.bot['predicted_class'] = int(res[0])
   
   result = 'Я считаю что это вопрос по теме ' + labels[res[0]] + '; Класс ' + str(res[0])
   result = result + '\nВыберите к какой категории относится этот вопрос на самом деле'
   await message.reply(result, reply_markup=inline_kb)


def wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('btn_'))
async def process_callback_kb1btn1(callback_query: types.CallbackQuery):
    user_message = callback_query.message.bot['user_input']
    predicted_class = callback_query.message.bot['predicted_class']
    #try:
    idx = int(callback_query.data[-1])
    callback_query.message.bot['users_class'] = idx
    conn = sqlite3.connect('cats.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_interactions (user_message, predicted_class, ground_truth) VALUES (?, ?, ?)', (user_message, predicted_class, idx))
    conn.commit()
    conn.close()
    #except Exception:
    #    print('Unknown button!')
    await callback_query.message.edit_reply_markup(reply_markup=None)
    await bot.send_message(callback_query.from_user.id, 'Запомнил, спасибо!')


@wrap
def predict_class_wrap(texts, model, tokenizer, max_len=128):
    return predict_class(texts, model, tokenizer, max_len)


def predict_class(texts, model, tokenizer, max_len=128):
    print('starting classification')
    model.eval()  
    inputs = tokenizer(
        texts, 
        add_special_tokens=True, 
        max_length=max_len,  
        padding='max_length', 
        truncation=True, 
        return_tensors='pt' 
    )
    
    # inputs to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device) #check
    
    # predictions
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # predicted class (logits -> class probabilities -> predicted class)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
    return predicted_class.cpu().numpy()  # predictions as a numpy array


if __name__ == '__main__':
    print('Self checking')
    new_texts = ["На пользователей Сбера были совершены попытки взломать почту",
                 "Получены письма с домена @sbersec, который был ненастоящий",
                 "Расскажи общий статус кибербезопасности банка",
                 "Какая ситуация с DDoS атаками?",
                 "Какой объем мошенничества был за последние 4 часа"]
    predicted_classes = predict_class(new_texts, model, tokenizer)

    for text, pred in zip(new_texts, predicted_classes):
        print(f"Text: {text} -> Predicted Class: {pred}")
    print('Starting bot')
    executor.start_polling(dp, skip_updates=True)
