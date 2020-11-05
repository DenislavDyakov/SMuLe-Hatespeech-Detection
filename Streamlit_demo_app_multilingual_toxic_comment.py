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

# testGrigor = pd.read_csv('content/TestGrigor.csv',sep='\t')
# x_testGrigor = fast_encode(testGrigor.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
#
# testGrigor_dataset = (
#     tf.data.Dataset
#         .from_tensor_slices(x_testGrigor)
#         .batch(BATCH_SIZE)
# )
#
# testGrigor['toxic'] = model.predict(testGrigor_dataset, verbose=1)
#
# testGrigor.to_csv('testGrigorResult.csv', index=False)
st.markdown("# Multilingual toxicity phrase test")
single_phrase_lst = "Първоначална фраза"
single_comment = st.text_input('Въведете фраза за проверка', 'Първоначална фраза')
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

# st.write("Current text", single_comment)
# st.write(":hot_pepper: :hot_pepper: :hot_pepper: :hot_pepper:")
# st.dataframe(testGrigor)