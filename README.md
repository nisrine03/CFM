#CFM Challenge Project
This project is focused on analyzing and predicting stock behavior based on order-book data, with a focus on constructing an optimized model for high-dimensional time-series classification. The project leverages deep learning techniques, specifically GRUs, to process the sequential data from various stock exchanges and predict target stock categories.

#Data Description
The dataset consists of 100 sequential order-book observations with 20 observations per stock, per day over 504 days across 24 stocks. The total number of rows is approximately 24.2 million. Below are descriptions of the key columns:

obs_id: A unique identifier for each sequence of 100 events.
venue: Encoded integer denoting the exchange (e.g., NASDAQ, BATY).
action: Indicates the type of event (A for added, D for deleted, U for updated).
order_id: Unique identifier for specific orders in a sequence, allowing tracking of an order's lifecycle.
side: The order book side (A or B) where the event took place.
price, bid, ask: Prices of the affected order, the best bid, and the best ask.
bid_size, ask_size: Volumes of orders at the best bid and ask.
flux: The volume change due to the event.
trade: Boolean indicating if the event was due to a trade or a cancellation.
To avoid data leakage, price-related fields are normalized by subtracting the initial best bid price from each event’s values in the sequence.

The target variable, Y, corresponds to eqt_code_cat, representing the stock identifier (integer values between 0 and 23) for classification.

#Benchmark Description
Model Architecture and Feature Construction
The model utilizes a GRU-based architecture with the following components:

Input Pre-processing: The dataset is transformed into a tensor of shape (100, 30), representing 100 events and a 30-dimensional input vector per observation.

Venue Embedding: [8 dimensions]
Action Embedding: [8 dimensions]
Trade Embedding: [8 dimensions]
Price Variables: Bid [1], Ask [1], Price [1]
Log-transformed Volumes: log(bid_size + 1) [1], log(ask_size + 1) [1]
Log-transformed Flux: log(flux) [1]
GRU Layers: Two 64-dimensional GRU cells are used:

Forward GRU: Processes the sequence forwards, producing a 64-dimensional output.
Backward GRU: Processes the sequence backwards, producing another 64-dimensional output.
Concatenation and Dense Layers:

The GRU outputs are concatenated into a 128-dimensional vector.
Two dense layers are applied:
128 to 64 Dimensions: Linear transformation with SeLU activation.
64 to 24 Dimensions: Linear layer to produce 24 logits, followed by a softmax layer for category probabilities.
Training
The model is trained as follows:

Loss Function: Cross-entropy for classification tasks.
Batching: 1000 random observations per batch of shape (1000, 100, 30).
Optimizer: Adam (default Optax parameters) with a learning rate of 3e-3.
Training Duration: 10,000 batches.
