import preprocessing

from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Concatenate, Dropout, GRU
)


def cnn_architecture(embedding_lyrics: Embedding) -> GlobalMaxPooling1D:
    """
    Constructs a Convolutional Neural Network (CNN) architecture for processing 
    embedded song lyrics.

    Args:
        embedding_lyrics (Embedding): An embedding layer output, representing 
        the input song lyrics in vectorized form.

    Returns:
        GlobalMaxPooling1D: The output of the final pooling layer, representing 
        the most prominent features extracted by the CNN.

    Architecture Details:
        - Conv1D Layer 1: 12 filters, kernel size of 4, 'relu' activation.
        - Conv1D Layer 2: 8 filters, kernel size of 6, 'relu' activation.
        - Conv1D Layer 3: 6 filters, kernel size of 8, 'relu' activation.
        - Final Layer: GlobalMaxPooling1D to reduce the output dimensionality 
          by selecting the maximum value across each feature map.
    """
    conv = Conv1D(
        filters= 12, kernel_size= 4, activation= 'relu',
        name= 'conv_layer1'
    ) (embedding_lyrics)
    
    conv = Conv1D(
        filters= 8, kernel_size= 6, activation= 'relu',
        name= 'conv_layer3'
    ) (conv)
    
    conv = Conv1D(
        filters= 6, kernel_size= 8, activation= 'relu',
        name= 'conv_layer4'
    ) (conv)

    return GlobalMaxPooling1D()(conv)
    

def rnn_architecture(embedding_lyrics: Embedding) -> GRU:
    """
    Builds a stacked GRU-based Recurrent Neural Network (RNN) architecture 
    for processing embedded song lyrics.

    Args:
        embedding_lyrics (Embedding): An embedding layer output, representing 
        the input song lyrics in vectorized form.

    Returns:
        GRU: The final GRU layer, representing the learned 
        features after passing through the stacked GRU layers.

    Architecture Details:
        - Layer 1: GRU with 128 units, 'tanh' activation, 'sigmoid' recurrent 
          activation, 40% dropout, and 30% recurrent dropout, returning sequences.
        - Layer 2: GRU with 64 units, similar configuration to Layer 1, also 
          returning sequences.
        - Layer 3: GRU with 32 units, 'tanh' activation, 'sigmoid' recurrent 
          activation, 40% dropout, returning a single vector output.
    """
    recurrent_layer = GRU(
        128, return_sequences= True,
        name= "recurrent_layer1",
        activation='tanh', recurrent_activation='sigmoid',
        dropout= 0.4, recurrent_dropout= 0.3
    ) (embedding_lyrics)
    
    recurrent_layer = GRU(
        64, return_sequences= True,
        name= "recurrent_layer2",
        activation='tanh', recurrent_activation='sigmoid',
        dropout= 0.4, recurrent_dropout= 0.3
    ) (recurrent_layer)
    
    return GRU(
        32, return_sequences= False,
        activation='tanh', recurrent_activation='sigmoid',
        name= "recurrent_layer3"
    ) (recurrent_layer)

    
def build_model(type: int) -> tuple:
    """
    Builds and returns a machine learning model along with its corresponding 
    preprocessing layers based on the specified model type.

    Args:
        type (int): An integer specifying the type of model to build. 
            - 0: Convolutional Neural Network (CNN) architecture.
            - 1: Recurrent Neural Network (RNN) architecture.

    Returns:
        tuple: A tuple containing:
            - model (keras.Model): The compiled Keras model ready for training.
            - preprocessing_layers (list): A list of layers required for 
              preprocessing the input data before feeding it into the model.

    Raises:
        ValueError: If the provided `type` is not 0 or 1.

    Functionality:
        - For type 0, constructs a CNN model with multiple convolutional layers 
          and pooling layers.
        - For type 1, constructs an RNN model using GRU layers with dropout 
          regularization.
    """
    lyrics_input = Input(shape=(preprocessing.max_seq_length,),name= 'text_input')

    embedding_lyrics = Embedding(
        input_dim= preprocessing.vocab_size, output_dim= 128,
    ) (lyrics_input)
    
    lyrics_dropout = Dropout(
        0.3 if type == 1
        else 0.5) (
            cnn_architecture(embedding_lyrics) if type == 1
            else rnn_architecture(embedding_lyrics)
        )

    # additional features
    stanza_number_input = Input(shape=(1,), name='stanza_number')

    bool_inputs = [
        Input(shape= (1,), name= name, dtype= 'int32')
        for name in [
            'is_country', 'is_pop', 'is_rap',
            'is_rb', 'is_rock', 'is_chorus'
        ]
    ]

    # title topic
    topic_input = Input(shape=(preprocessing.num_topics,), name="Topic_Input")

    # concatenate all inputs
    additional_input = Concatenate() (
        [stanza_number_input, topic_input] + bool_inputs
    )

    additional_input = Dense(
        32, activation= 'relu',
        name= 'additional_input'
    )(additional_input)

    # combine branches
    combined = Concatenate()([
        lyrics_dropout, additional_input
    ])
    dense1 = Dense(32, activation='relu')(combined)
    dropout = Dropout(0.3)(dense1)
    output = Dense(8, activation='softmax', name='output')(dropout)

    return (
        # inputs
        [lyrics_input, stanza_number_input, topic_input] + bool_inputs,
        # output
        output
    )