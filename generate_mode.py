import sys
import numpy as np
import transformers
from scipy.stats import multivariate_normal
from tqdm import tqdm


# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# with open(sys.argv[-1]) as essay_file:
#     embeddings = model.encode(essay_file.read().splitlines())

# mean, cov = multivariate_normal.fit(embeddings)
# min_eig = np.linalg.eigvals(cov).real.min()
# if min_eig < 0:
#     cov -= 10 * min_eig * np.eye(*cov.shape)

# gaussian = multivariate_normal(mean, cov)
# print(gaussian.rvs())

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

embeddings = []
with open(sys.argv[-1]) as essay_file:
    for essay in tqdm(essay_file, desc="Encoding essays"):
        embeddings.append(tokenizer.encode(essay, padding="max_length", max_length=800))

mean, cov = multivariate_normal.fit(np.array(embeddings))
min_eig = np.linalg.eigvals(cov).real.min()
if min_eig < 0:
    cov -= 10 * min_eig * np.eye(*cov.shape)
gaussian = multivariate_normal(mean, cov, allow_singular=True)

print(tokenizer.decode(gaussian.rvs().astype(int), skip_special_tokens=True))
