import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import transformers
from tokenizers import BertWordPieceTokenizer
from annotated_text import annotated_text
import nltk.data

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# @st.cache
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding()
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)

#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE

# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192

# @st.cache
def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

model.load_weights('model_weights.h5')


####################
## STREAMLIT SIDEBAR
####################

def execute_streamlit():

    add_selectbutton_phrase_evaluate = True

    add_selectbutton_phrase_evaluate = st.sidebar.checkbox("phrase evaluate", key=None)
    # add_selectbutton_scrape_evaluate = st.sidebar.button("scrape evaluate", key=None)


    add_selectbox_scrape = st.sidebar.selectbox(
            "Which data would like to evaluate?",
            ("-", "pik.bg", "sega.bg")
        )

    st.markdown("# Multilingual toxicity phrase test")
    if add_selectbutton_phrase_evaluate:

        single_phrase_lst = "Initial phrase"
        single_comment = st.text_input('Please enter phrase below:', 'Initial phrase')
        st.markdown('-------------')
        single_phrase_lst = [single_comment]
        single_phrase_df = pd.DataFrame(single_phrase_lst, columns=['content'])

        x_single_phrase_df = fast_encode(single_phrase_df.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
        single_phrase_df_dataset = (
            tf.data.Dataset
                .from_tensor_slices(x_single_phrase_df)
                .batch(BATCH_SIZE)
        )
        single_phrase_df['toxic'] = model.predict(x_single_phrase_df, verbose=1)
        # st.write(single_phrase_df.iloc[0]['content'])
        st.write(round(single_phrase_df.iloc[0]['toxic'] * 100, 2), '% toxicity')

        result = float(single_phrase_df.iloc[0]['toxic'])

        text_result = ''

        if result < 0.2:
            text_result = ':angel:'
        else:
            text_result = int((result - 0.1) / 0.1) * ':hot_pepper:'
        st.write(single_phrase_df.iloc[0]['content'],'  ', text_result)

        st.markdown('-------------')

        # Starts annotating of a whole phrase
        annotated_sentence = []
        single_comment_words = single_comment.split(' ')
        print(single_comment_words)
        for word in single_comment_words:
            single_phrase_lst = [word]
            single_phrase_df = pd.DataFrame(single_phrase_lst, columns=['content'])

            x_single_phrase_df = fast_encode(single_phrase_df.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
            single_phrase_df_dataset = (
                tf.data.Dataset
                    .from_tensor_slices(x_single_phrase_df)
                    .batch(BATCH_SIZE)
            )
            single_phrase_df['toxic'] = model.predict(x_single_phrase_df, verbose=1)
            toxicity =  round(single_phrase_df.iloc[0]['toxic'] * 100, 2)
            annotation_color = '' #'#FFFFFF' # white
            if 33 < toxicity < 66:
                annotation_color = '#fea'   # red
            elif toxicity >= 66:
                annotation_color = '#faa'   # pink

            print(word, str(toxicity) +'%', annotation_color)
            annotated_sentence.append((word, str(toxicity) +'%', annotation_color))
            print(annotated_sentence)

        annotated_text(*annotated_sentence)


    def scrape():
        print(add_selectbox_scrape)
        if add_selectbox_scrape == 'pik.bg':
            scrape_data = pd.read_csv("scrapers/pik_results.csv",)
        else:
            scrape_data = pd.read_csv("scrapers/sega_results.csv")
        scrape_data['content_list'].replace('', np.nan, inplace=True)
        scrape_data.dropna(subset=['content_list'], inplace=True)
        to_select_from = scrape_data['title']
        selection_title = st.selectbox("Select title:",to_select_from)
        print(selection_title)
        is_selected = scrape_data['title'] == selection_title
        selected = scrape_data[is_selected]
        is_selected = scrape_data['id'] == min(selected['id'])
        selected2 = scrape_data[is_selected]
        # st.markdown("---------------")
        # st.dataframe(selected2)
        list_selected = selected2.values.tolist()
        list_selected = list_selected[0]
        # PRINT SELECTED TITLE DATA
        st.markdown("---------------")
        st.markdown(f":newspaper: [link]({list_selected[1]})")
        st.markdown("---------------")
        st.write(f"{list_selected[2]}")
        st.markdown("---------------")


# Start annotating scraped text
        def annotating_sentence(phrases):
            annotated_sentence2 = []
            # st.write("PHRASES::::::::::::")
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            # fp = open("test.txt")
            single_comment_words = tokenizer.tokenize(phrases)
            for word in single_comment_words:
                single_phrase_lst = [word]
                single_phrase_df = pd.DataFrame(single_phrase_lst, columns=['content'])

                x_single_phrase_df = fast_encode(single_phrase_df.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
                single_phrase_df_dataset = (
                    tf.data.Dataset
                        .from_tensor_slices(x_single_phrase_df)
                        .batch(BATCH_SIZE)
                )
                single_phrase_df['toxic'] = model.predict(x_single_phrase_df, verbose=1)
                toxicity =  round(single_phrase_df.iloc[0]['toxic'] * 100, 2)
                annotation_color = '' #'#FFFFFF' # white
                if 33 < toxicity < 66:
                    annotation_color = '#fea'   # red
                elif toxicity >= 66:
                    annotation_color = '#faa'   # pink
                if toxicity < 33:
                    toxic_percentage = ''
                else:
                    toxic_percentage = str(toxicity) +'%'

                print(word, toxic_percentage , annotation_color)
                annotated_text((word, str(toxicity) +'%', annotation_color))
                annotated_sentence2.append((word, str(toxicity) +'%', annotation_color))
                print(annotated_sentence2)



        sentence = list_selected[3]
        sentence = sentence[1:-1]
        sentence =  sentence.replace('\\xa0', ' ')
        annotating_sentence(sentence)



    if add_selectbox_scrape != "-":
        scrape()

execute_streamlit()
