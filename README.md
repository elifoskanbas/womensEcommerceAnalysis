# Women's E-Commerce Clothing Reviews — Sentiment Analysis


## Task Overview
**Task 1:** This task performs sentiment analysis on the Women's E-Commerce Clothing Reviews dataset using a BERT-based multilingual model.
It classifies customer reviews as positive, neutral, or negative, providing insights into customer satisfaction and feedback trends.

**Task 2:** This task builds a recommendation system that identifies products with positive user experiences and suggests similar products based on text embeddings and similarity scoring.
The system combines review data, sentiment analysis results, and sentence embeddings to find high-rated products and recommend related items within the same category.

## Dataset
**Source:** Kaggle - Women's E-Commerce Clothing Reviews

**Description:**
This dataset contains customer reviews and ratings for women's clothing items sold online.
It includes fields such as review text, rating, product class, and recommendation status.



## Task 1 Methodology 
**1.Data Preparation:**
- The redundant Unnamed: 0 column was dropped.
- Missing values in the Review Text column were removed.
- A clean_text column was created by combining Title and Review Text (handling NaN titles) and applying a cleaning function (clean_text) to remove HTML, URLs, and non-ASCII characters, and convert the text to lowercase.
- Reviews shorter than 5 characters were removed.

**2.Sentiment Prediction (BERT Model):**
- The pre-trained nlptown/bert-base-multilingual-uncased-sentiment model from the transformers library was used for sentiment classification. This model outputs a score from 1 (very negative) to 5 (very positive).
- A sentiment_result column was created with the predicted score (1-5) for each cleaned review text.

**3.Mapping Scores to Labels & True Sentiment:**

- The predicted scores (1-5) were mapped to the final sentiment labels:

  - Score 1 or 2: negative.

  - Score 3: neutral.

  - Score 4 or 5: positive.

- A true_sentiment column was created using the original Rating column (1-5) and the same mapping logic for comparison.

**4.Handling Conflicts (Recommendation Indicator):**

- The sentiment was adjusted based on the Recommended IND (1 for Recommended, 0 for Not Recommended) to resolve conflicting predictions:

  -  If negative was predicted, but the item was Recommended (Recommended IND = 1), the prediction was changed to neutral.

  - If positive was predicted, but the item was Not Recommended (Recommended IND = 0), the prediction was changed to neutral.

**5.Evaluation:**

- The model achieved an Accuracy of approximately **%84** .

- The full evaluation included a Classification Report and a Confusion Matrix to assess performance across the three classes:


| **Metric**  | **Negative** | **Neutral** | **Positive** |
|--------------|--------------|-------------|---------------|
| **Precision** | 0.66 | 0.40 | 0.98 |
| **Recall**    | 0.78 | 0.58 | 0.88 |
| **F1-Score**  | 0.71 | 0.47 | 0.93 |
| **Support**   | 2,370 | 2,823 | 17,448 |


## Task 2 Methodology

**1.Identify Positive Products:**

- The Average_Rating for each Clothing ID was calculated.

- Only products with an Average_Rating of 4.0 or higher were considered "products with positive user experiences" and kept in the dataset for recommendations.

**2.Text Embedding (Sentence-BERT):**

- The clean_text (combined title and review) for all reviews was converted into high-dimensional vectors (embeddings) using the pre-trained Sentence Transformer model sentence-transformers/all-MiniLM-L6-v2.

**3.Product-Level Embeddings:**

- An average embedding (avrEmbedding) was calculated for each unique Clothing ID by taking the mean of all individual review embeddings associated with that product. This vector represents the overall "sentiment and topic profile" of the product's reviews.

**4.Similarity Calculation:**

- A Cosine Similarity Matrix was computed using the avrEmbedding for all products.

**5.Recommendation System (Content-Based Filtering):**

- The system takes a target_id (e.g., 767).

- Filtering: Recommendations were restricted to products within the same Class Name as the target product to ensure relevancy.

- Sorting: The similarity scores for same-class products were sorted in descending order.

- Output: The top N most similar products (excluding the target itself) are returned.

**Example Recommendation**

For Product 767 (Class: Intimates), the top 5 most similar products are:

- Product 362 (Class: Intimates) → Similarity: 0.727

- Product 37 (Class: Intimates) → Similarity: 0.686

- Product 78 (Class: Intimates) → Similarity: 0.650

- Product 694 (Class: Intimates) → Similarity: 0.642

- Product 807 (Class: Intimates) → Similarity: 0.634


## Technologies Used
- Python 3.x
- Transformers (Hugging Face)
- PyTorch
- Pandas & NumPy
- Matplotlib & Seaborn
- BeautifulSoup4 (bs4)
- Kaggle API
- Regular Expressions (re)


## Installation & Setup
Clone this repository:
``` bash
git clone https://github.com/elifoskanbas/womens-ecommerce-sentiment-analysis.git
cd womens-ecommerce-sentiment-analysis
``` 
Install dependencies:
``` bash
pip install -r requirements.txt
``` 
Download the dataset from Kaggle:
``` bash
kaggle datasets download -d nicapotato/womens-ecommerce-clothing-reviews
unzip womens-ecommerce-clothing-reviews.zip -d clothing_reviews
``` 
Run the notebook or Python script.


## Future Improvements
- Fine-tune BERT on this dataset for improved accuracy.
- Add feature-based sentiment analysis (e.g., per product type).
- Build a web interface for real-time sentiment prediction.
- Incorporate SHAP/LIME for explainable AI insights.
