# Two-Tower E-Commerce Recommendation Engine

Production deep learning recommender:
- **User Tower**: Maps user interaction history and demographics to $d=32$ dense vectors.
- **Item Tower**: Maps product metadata and catalogs to $d=32$ dense vectors.
- **Serving**: Sub-5ms cosine retrieval with candidate re-ranking.
