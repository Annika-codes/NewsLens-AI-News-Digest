# NewsLens

A daily news digest with an old-newspaper front page. It pulls live stories on tech, AI, machine learning, career opportunities and top news. A tone classifier **trained on headlines I labelled myself** tags each story, and a small language model running **in the browser** finds related stories and ranks the page by what I read.

It is a static site (HTML, CSS, JavaScript). There is no server and no paid API.

## How it works

```
public RSS feeds -> rss2json (feed-to-JSON service) -> browser
        |
        v
   clean text, short excerpt -> tone classifier (my trained model) -> tone tag + words behind it
        |
        v
   MiniLM embeddings in the browser -> related stories + "For You" ranking
```

- **Daily edition:** the first visit each day fetches fresh stories and saves them in the browser; later visits that day load instantly.
- **Tone classifier:** TF-IDF (words and word pairs) plus logistic regression, trained with scikit-learn in `ml/train_tone.py`. The weights are exported to `tone_model.json`, and `index.html` runs the same maths in JavaScript. I checked that the browser output matches scikit-learn's probabilities to within 0.0001. The page also shows the words that pushed each label.
- **Smart Mode:** optional. Loads `all-MiniLM-L6-v2` through transformers.js (about 25 MB, cached after the first load). Each story becomes a 384-number vector.
  - *Related dispatches:* cosine similarity between story vectors.
  - *For You:* my "taste" is a weighted average of the vectors of stories I read (newer reads count more); stories are ranked by similarity to it. Needs 3+ reads.

## Run it locally

```
python -m http.server 8000      # inside this folder, then open http://localhost:8000
```

Use a local server, not a double-click. Browsers block `tone_model.json` from loading on `file://`, and the page then falls back to the keyword rules.

## Train the tone model

1. Open `ml/label_tool.html` (also through the local server or just double-click; it needs internet for the feeds).
2. Press **Fetch headlines** and label them with keys 1/2/3 (neutral, emotional, opinionated). Label the *wording*, not the news. Aim for 40+ of each label, more is better. Come back on other days to fetch new headlines.
3. Press **Download labels.csv** and put it in `ml/`.
4. Train:
   ```
   cd ml
   pip install -r requirements.txt
   python train_tone.py --data labels.csv --out ../tone_model.json
   ```
5. Read `ml/report.txt`: it has the cross-validated accuracy and macro-F1, a confusion matrix, the terms behind each label, and a comparison against the old keyword rules and an always-guess-the-majority baseline.
6. Reload the page. The ledger at the bottom shows **Tone Engine: Trained (F1 ...)** once the model loads.

## Deploy on Vercel

1. Put this folder in a GitHub repo (include `tone_model.json` after training).
2. In Vercel: **Add New, Project**, import the repo, **Deploy**. No build settings are needed.

## Limitations (honest ones)

- The tone labels reflect one person's judgement. A single labeller means no measure of agreement between labellers.
- The labelling tool can show likely-loaded headlines first to find the rarer labels. That makes the training set less like the natural mix of headlines, so real-world accuracy may differ from the cross-validated score.
- Feeds are read through a free third-party service (rss2json). If it is down or rate-limited, sections can be empty and the page shows sample stories.
- Summaries are the feeds' own excerpts, not generated.
- "For You" needs a few reads before it works, and it learns only in this browser, on this device.
- Smart Mode needs to download a model from the internet the first time.

## Things I can explain in an interview

- Why the model was evaluated with nested cross-validation and compared with two baselines.
- How TF-IDF and logistic regression produce a label and a confidence, and how I exported them to run in JavaScript.
- What an embedding is, why cosine similarity finds related stories, and how a mean vector can stand in for taste.
- The trade-offs of running ML on the client (no server cost and private, but a larger first load).
