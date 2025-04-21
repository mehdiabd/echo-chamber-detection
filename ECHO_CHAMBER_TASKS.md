# ✅ Echo Chamber Detection Project – Task Roadmap

## 1. Data Collection & Preprocessing
- [ ] Finalize data scraping from Telegram (using Telethon / Pyrogram)
- [ ] Represent users and interactions as a **graph**
- [ ] Normalize and weight interaction types (reply, forward, like, etc.)
- [ ] (Optional) Store structured data in SQLite / Pandas / NetworkX formats

## 2. Graph Construction
- [ ] Build the graph using `networkx` (nodes = users, edges = interactions)
- [ ] Assign edge weights based on interaction frequency/type
- [ ] Visualize basic graph statistics (e.g. degree distribution)

## 3. Community Detection Phase A – Louvain
- [x] Prototype Louvain algorithm (already done/tested in notebook)
- [ ] Apply Louvain on real Telegram graph
- [ ] Visualize detected communities
- [ ] Analyze modularity score + size distribution of communities

## 4. Community Detection Phase B – Node2Vec + KMeans
- [ ] Implement Node2Vec embedding on the user graph
- [ ] Reduce dimensions (e.g., with PCA or TSNE) for visualization
- [ ] Cluster embeddings with KMeans or DBSCAN
- [ ] Visualize & compare with Louvain results
- [ ] Interpret community structure from a semantic/social perspective

## 5. Echo Chamber Analysis
- [ ] Define criteria for echo chamber detection (e.g. high modularity, low inter-cluster interaction)
- [ ] Identify echo chamber clusters based on interaction patterns
- [ ] Analyze dominant topics (if messages are available)
- [ ] Measure polarization (optional: using graph metrics like assortativity)

## 6. Evaluation & Reporting
- [ ] Compare methods (Louvain vs. Node2Vec+KMeans) based on:
  - Modularity
  - Cohesiveness
  - Interpretability
- [ ] Create visual and textual reports
- [ ] (Optional) Export results in CSV or JSON

## 7. Refactor & Finalize Codebase
- [ ] Move code from notebooks to modular `.py` files
- [ ] Keep notebooks for experimentation
- [ ] Clean up project structure and add README
- [ ] Ensure all dependencies are captured in `requirements.txt`