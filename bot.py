import logging
import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler, ConversationHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
import pandas as pd
from telegram.inline.inlinekeyboardbutton import InlineKeyboardButton
from telegram.inline.inlinekeyboardmarkup import InlineKeyboardMarkup
from telegram.update import Update
import sys
from simpletransformers.classification import ClassificationModel
import spacy
import time
from dotenv import load_dotenv
load_dotenv()
import os


nlp = spacy.load("en_core_web_lg")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)
LANGS = {
    "en": "English",
    "ceb": "Cebuano",
    "sv": "Swedish",
    "de": "German",
    "fr": "French",
    "nl": "Dutch",
    "ru": "Russian",
    "it": "Italian",
    "es": "Spanish",
    "pl": "Polish",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "uk": "Ukrainian",
    "pt": "Portuguese",
    "fa": "Farsi",
    "ca": "Catalan",
    "sr": "Serbian",
    "id": "Indonesian",
    "no": "Norwegian",
    "ko": "Korean",
    "fi": "Finnish",
    "hu": "Hungarian",
    "cs": "Czech",
    "sh": "Serbo-Croation"
}


LANG_KBD = []
for i in range(0, len(LANGS.keys()), 4):
    LANG_KBD.append([telegram.InlineKeyboardButton(v, callback_data=k) for k, v in list(LANGS.items())[i:i+4]])


# Question IDs in this list will not be asked (we have some strange duplicates in the data)
QUESTION_BLACKLIST = [
    12.5
]

SIMILARITY_THRESHOLD = 0.5

data = pd.read_csv("./texas-test-set-translated.csv")
bert_model = ClassificationModel('bert', './models/asag-ml-6/', num_labels=1, use_cuda=False, args={ "regression": True })

LANGUAGE_SELECT, QUESTION, FEEDBACK = range(3)

def start(update: Update, context: CallbackContext):
    update.message.reply_photo("https://images.pexels.com/photos/2166/flight-sky-earth-space.jpg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260")
    context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="""\U0001F44B Hello astronaut! 
Welcome to the first manned mission to Mars! \U0001FA90

We have <strong>56 million km</strong> to reach the red planet. 
As our Chief Engineer you are responsible that the engine runs smoothly by <strong>answering Computer Science questions</strong> and help other astronauts with your <strong>feedback</strong>.

Are you ready for departure?    

Please select your <strong>language</strong>:""",
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=telegram.InlineKeyboardMarkup(LANG_KBD),
    )
    return LANGUAGE_SELECT

def set_language(update: Update, context: CallbackContext):
    #lang = update.message.text
    lang = update.callback_query.data
    
    # initialize the user data
    context.user_data["lang"] = lang
    context.user_data["distance"] = 56
    context.user_data["asked_questions"] = set()

    update.callback_query.message.reply_text(f"""Thank you! \u2764
Language set to <strong>{LANGS[lang]}</strong>.

You accidently selected the wrong language? 
No problem, just type <code>/start</code> to restart!

Ready? Let's start the engine! Here is the first question:
""", parse_mode=telegram.ParseMode.HTML)
    return ask_question(update, context)

def feedback(update: Update, context: CallbackContext):
    update.callback_query.message.reply_html("""Thank you for your feedback. 
Let's tackle the <strong>next question</strong>!""")
    return ask_question(update, context)


def ask_feedback(update: Update, context: CallbackContext):
    result = data.query(f"""questionid == {context.user_data["questionid"]}""").to_dict("records")
    question = result[0]

    

    lang = context.user_data["lang"]
    suffix = "_" + lang if lang != "en" else ""
    model = question["model"+suffix]
    question_text = question["question"+suffix]
    answer = question["student"]

    update.message.reply_text(f"""<strong>Wait!</strong> To use the new fuel, you need to activate the fuel pump! To do this, please <strong>grade the following answer</strong> of an English speaking fellow astronaut:

\u2753 <strong>The question</strong>:
{question_text}

\N{pencil} <strong>The astronaut's answer</strong>:
<i>{answer}</i>

\N{green book} <strong>Model answer</strong>: 
{model}

<strong>How many points would you give?</strong> Please be fair and honest!""", 
    reply_markup=telegram.InlineKeyboardMarkup([list([telegram.InlineKeyboardButton(i, callback_data=i) for i in range(6)])], one_time_keyboard=True), parse_mode=telegram.ParseMode.HTML)
    return FEEDBACK

def ask_question(update: Update, context: CallbackContext):
    lang = context.user_data["lang"]
    suffix = "_" + lang if lang != "en" else ""
    asked_questions = context.user_data["asked_questions"]

    new_question_found = False
    while not new_question_found:
        questions = data.sample().to_dict("records")
        question = questions[0]

        questionid = question["questionid"]
        if not questionid in QUESTION_BLACKLIST and not questionid in asked_questions:
            asked_questions.add(questionid)
            new_question_found = True

    question_text = question["question"+suffix]
    questionid = question["questionid"]
    print("QUESTION ID:", questionid)

    context.user_data["questionid"] = questionid

    message = update.message if update.message else update.callback_query.message
    message.reply_html(f"\u2753 <strong>{question_text}</strong>")

    return QUESTION

def tokens2mdown(doc, highlights):
    output = []
    print(highlights)
    for token in doc:
        if token.orth in highlights:
            output.append(f"<strong>{token.text}</strong>{token.whitespace_}")
        else:
            output.append(token.text_with_ws)
    return "".join(output)

def grade_answer(update: Update, context: CallbackContext):
    result = data.query(f"""questionid == {context.user_data["questionid"]}""").to_dict("records")
    question = result[0]

    lang = context.user_data["lang"]
    suffix = "_" + lang if lang != "en" else ""
    model = question["model"+suffix]
    question_text = question["question"+suffix]

    answer = update.message.text

    score_bert, raw_outputs = bert_model.predict([[model, answer]])
    score = round(score_bert.item())

    distance = context.user_data["distance"] = max(context.user_data["distance"] - 2*score, 0)

    if lang == "en":
        model_doc = nlp(model)
        answer_doc = nlp(answer)
        highlights = [] # list of highlighted tokens

        for token in model_doc:
            if not token.is_stop and token.is_alpha:
                for answer_token in answer_doc:
                    if not answer_token.is_stop and token.is_alpha and token.similarity(answer_token) > SIMILARITY_THRESHOLD:
                        highlights.append(token.orth)
                        highlights.append(answer_token.orth)

        model = tokens2mdown(model_doc, highlights)
        answer = tokens2mdown(answer_doc, highlights)


    stars = "".join(list(["\u2B50" for x in range(score)]))
    distance_points = "".join(list(["." for x in range(distance)]))
    travelled_points = "".join(list(["." for x in range(56-distance)]))

    update.message.reply_text(f"""Thank you for your answer! \U0001f44d

\u2753 <strong>The question</strong>:
{question_text}

\N{pencil} <strong>Your answer</strong>:
{answer}

\N{green book} <strong>Model answer</strong>: 
{model}

\U0001F393 <strong>Your grade</strong>:
{score} of 5 points {stars}

Awesome, that gives us fuel for {score*2} million more kilometers! \U0001F44F

Only <strong>{distance} million</strong> km away from Mars!
\U0001F30D {travelled_points} \N{rocket} {distance_points} \U0001FA90""", parse_mode=telegram.ParseMode.HTML)

    time.sleep(5)

    # we have reached Mars, end the game
    if distance == 0:
        update.message.reply_photo("https://cdn.pixabay.com/photo/2012/11/28/09/08/mars-67522_960_720.jpg")
        update.message.reply_text(f"""<strong>Congratulations</strong>, you successfully landed on Mars! \U0001F4AF

Now, the researchers on Earth need your help! 
<strong>Please help us by filling out our short survey:</strong>""", 
                parse_mode=telegram.ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Open Survey", url="https://docs.google.com/forms/d/e/1FAIpQLSdpI1PDu1OCk5qrFv1n-BrJOfFIJrhFoba0GIaEYxeiGNvj5g/viewform")]]
                )
            )

        update.message.reply_html("""<strong>Thank you!</strong>

PS: If you would like to play again, just type:
<code>/start</code>""",
        )
        return ConversationHandler.END
    
    return ask_feedback(update, context)

def done(update: Update, context: CallbackContext):
    print("done")
    return ConversationHandler.END

def main() -> None:

    updater = Updater(token=os.getenv("BOT_TOKEN"))
    dispatcher = updater.dispatcher

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            LANGUAGE_SELECT: [
                CallbackQueryHandler(
                    set_language
                )
            ],
            QUESTION: {
                MessageHandler(
                    Filters.text & (~Filters.command), grade_answer
                )
            },
            FEEDBACK: {
                CallbackQueryHandler(
                    feedback
                )
            }

        },
        fallbacks=[CommandHandler("start", start)],
    )

    dispatcher.add_handler(conv_handler)

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()