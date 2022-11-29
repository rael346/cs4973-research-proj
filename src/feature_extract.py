import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import torch
from numpy import dot
from numpy.linalg import norm
import datetime
import math
from tqdm import tqdm

ANNOTATION = 'annotation'
GALLERY = 'gallery'
QUERY = 'query'
PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
IMAGE_BATCH_SIZE = 3072
num_images_not_found = 0


def get_model(finetuned=False):
    model = SentenceTransformer('clip-ViT-B-32')
    # TODO finetuning?
    if finetuned:
        pass
    return model

# Returns a list of 3-tuples containing 1 query feature vector (1 source image vector added to 1 feedback
# vector), 1 target feature vector, and 1 non-target feature vector. All 3 feature vectors are of size 128


def extract_features_annotation(annotation_df):
    feature_vectors = []
    model = get_model(finetuned=False)

    for _index, row in annotation_df.iterrows():
        # get data from dataframe
        source_id = row['Source Image ID']
        target_id = row['Target Image ID']
        non_target_id = row['Non-Target Image ID']
        feedback1 = row["Feedback 1"]
        feedback2 = row["Feedback 2"]
        feedback3 = row["Feedback 3"]
        feedbacks = [feedback1, feedback2, feedback3]

        # encode query (source image + 3 feedbacks), target image, non-target image
        query_emb = embed_query(model, get_img_path(
            source_id, ANNOTATION), feedbacks)
        target_emb = embed_image(model, get_img_path(target_id, ANNOTATION))
        non_target_emb = embed_image(
            model, get_img_path(non_target_id, ANNOTATION))

        feat_vec_triplet = (query_emb, target_emb, non_target_emb)
        feature_vectors.append(feat_vec_triplet)
    return feature_vectors

# Returns a list of 2-tuples containing 1 query feature vector (1 source image vector added to 1 feedback
# vector) and the list of candidate vectors. All feature vectors are of size 128


def extract_features_query_file(path: str, model: SentenceTransformer) -> list[tuple[torch.TensorType, list[torch.TensorType]]]:
    query_df = pd.read_json(path, lines=True)
    feature_vectors = []
    # model = get_model(finetuned=False)

    for _index, row in query_df.iterrows():
        # get data from dataframe
        source_id = row['source_pid']
        feedbacks = [row['feedback1'], row['feedback2'], row['feedback3']]
        candidates = row['candidates']

        candidate_embs = []
        for candidate in candidates:
            candidate_id = candidate['candidate_pid']
            candidate_emb = embed_image(
                model, get_img_path(candidate_id, QUERY), False)
            candidate_embs.append(candidate_emb)

        # encode query (source image + 3 feedbacks), target image, non-target image
        query_emb = embed_query(
            model, get_img_path(source_id, QUERY), feedbacks)

        feature_vectors.append((query_emb, candidate_embs))
    return feature_vectors

# Returns the image file path given the image name
# folder name should be either 'annotation', 'gallery', 'query'


def get_img_path(image_name, folder_name):
    assert (isinstance(image_name, str))
    assert (isinstance(folder_name, str))
    assert (folder_name in {'annotation', 'gallery', 'query'})
    path = './images/' + str(folder_name) + '/' + image_name + '.jpg'
    return path

# Encode image using sentence transformer model into 128-dimensional embedding


def embed_image(model: SentenceTransformer, image_path: str, scale_down: bool):
    if os.path.exists(image_path):
        # encoded as 512 dimensional
        emb = model.encode(Image.open(image_path))
        if scale_down:
            # 'scale' down by taking every 4th value (for now)
            emb = emb[0:emb.size:4]
        return emb
    else:
        global num_images_not_found
        num_images_not_found += 1
        print(image_path + ' was of ' +
              str(num_images_not_found) + ' images not found')
        return None         # image could not be found


def embed_text(model, text):
    emb = model.encode(text)  # encoded as 512 dimensional
    # 'scale' down by taking every 4th value (for now)
    size_128_emb = emb[0:emb.size:4]
    return size_128_emb


def embed_query(model: SentenceTransformer, image_path: str, feedbacks: list[str]):
    src_emb = embed_image(model, image_path, False)
    if src_emb is None:       # src_emb image_path was not found
        return None
    # we simply add feedback embedding vectors (for now)
    text_emb = sum(model.encode(feedbacks))
    # we simply add source and text vectors to get query embedding (for now)
    query_emb = src_emb + text_emb
    return query_emb


def get_img_emb(model: SentenceTransformer, path: str) -> dict[str, torch.Tensor]:
    img_names = os.listdir(path)
    img_emb_dict = {}

    num_batch = math.ceil(len(img_names) / IMAGE_BATCH_SIZE)
    for i in range(0, len(img_names), IMAGE_BATCH_SIZE):
        print("BATCH", i // IMAGE_BATCH_SIZE + 1, "OF", num_batch)
        batch_img = img_names[i: i + IMAGE_BATCH_SIZE]
        img_embs = model.encode([Image.open(path + "/" + name)
                                for name in batch_img], show_progress_bar=True)

        img_emb_dict.update({img_name.split(".")[0]: img_emb for img_name,
                             img_emb in zip(batch_img, img_embs)})

    return img_emb_dict


def get_feedback_emb_from_query(model: SentenceTransformer):
    query_df = pd.read_json(PATH_QUERY_FILE, lines=True)
    feedbacks = []
    source_pids = []
    for _, row in query_df.iterrows():
        source_pids.append(row['source_pid'])
        feedbacks.append(row['feedback1'])
        feedbacks.append(row['feedback2'])
        feedbacks.append(row['feedback3'])
    feedback_embs = model.encode(feedbacks, show_progress_bar=True)

    feedback_emb_dict = {}
    for i, pid in enumerate(source_pids):
        feedback1 = feedback_embs[3 * i]
        feedback2 = feedback_embs[3 * i + 1]
        feedback3 = feedback_embs[3 * i + 2]

        feedback_emb = feedback1 + feedback2 + feedback3
        feedback_emb_dict[pid] = feedback_emb

    return feedback_emb_dict


def evaluate(path: str, model: SentenceTransformer):
    query_df = pd.read_json(path, lines=True)
    query_df_scored = query_df.copy(deep=True)
    img_embs = get_img_emb(model, "images/query")
    print("IMG EMBS LENGTH:", len(img_embs))
    feedback_embs = get_feedback_emb_from_query(model)

    missing_img_count = 0

    for i_row, row in tqdm(query_df.iterrows()):
        source_pid = row["source_pid"]
        source_img_emb = img_embs.get(source_pid, None)

        if source_img_emb is None:
            missing_img_count += 1
            print("Missing source image", source_pid,
                  ", current count:", missing_img_count)
            source_emb = None
        else:
            feedback_emb = feedback_embs[source_pid]
            source_emb = source_img_emb + feedback_emb

        for i_c, c in enumerate(row["candidates"]):
            c_pid = c["candidate_pid"]
            c_emb = img_embs.get(c_pid, None)

            if c_emb is None:
                missing_img_count += 1
                print("Missing candidate pid image", c_pid,
                      ", current count:", missing_img_count)

            if c_emb is None or source_emb is None:
                score = 0
            else:
                score = util.cos_sim(source_emb, c_emb).item()

            query_df_scored.iloc[i_row]['candidates'][i_c]['score'] = score

    return query_df_scored


if __name__ == "__main__":
    model = SentenceTransformer('clip-ViT-B-32')
    # PATH_RESULTS_SAVE = './results/scored_query_file' + \
    #     datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jsonl'
    # scored = evaluate(PATH_QUERY_FILE, model)
    # scored.to_json(path_or_buf=PATH_RESULTS_SAVE, orient='records', lines=True)

    img_embs = get_img_emb(model, "images/query")
    print(img_embs["B1cy1Ci8XIS"] is None)
