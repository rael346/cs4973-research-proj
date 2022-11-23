import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import torch

ANNOTATION = 'annotation'
GALLERY = 'gallery'
QUERY = 'query'
PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
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
    path = '../images/' + str(folder_name) + '/' + image_name + '.jpg'
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

# Encode text feedbacks using sentence transformer model into 128-dimensional embedding


def embed_text(model, text):
    emb = model.encode(text)            # encoded as 512 dimensional
    # 'scale' down by taking every 4th value (for now)
    size_128_emb = emb[0:emb.size:4]
    return size_128_emb


def embed_query(model: SentenceTransformer, image_path: str, feedbacks: list[str]):
    assert (len(feedbacks) == 3)
    src_emb = embed_image(model, image_path, False)
    if src_emb is None:       # src_emb image_path was not found
        return None
    # we simply add feedback embedding vectors (for now)
    # text_emb = embed_text(model, feedbacks[0]) + embed_text(
    #     model, feedbacks[1]) + embed_text(model, feedbacks[2])
    text_emb = sum(model.encode(feedbacks))
    # we simply add source and text vectors to get query embedding (for now)
    query_emb = src_emb + text_emb
    return query_emb


if __name__ == "__main__":
    # load the annotations dataframe
    # annotation_df = pd.read_csv(PATH_APPAREL_TRAIN_ANNOTATION)
    # annotation_feature_vectors = extract_features_annotation(annotation_df.iloc[0:1])

    # load the query file dataframe

    # query_file_feature_vectors = extract_features_query_file(query_df.iloc[0:1])

    # ANNOTATION components
    # annotation_query_vectors = [f_v_triplet[0] for f_v_triplet in annotation_feature_vectors]
    # annotation_target_vectors = [f_v_triplet[1] for f_v_triplet in annotation_feature_vectors]
    # annotation_non_target_vectors = [f_v_triplet[2] for f_v_triplet in annotation_feature_vectors]

    # QUERY FILE components
    # query_file_query_vectors = [f_v_couple[0] for f_v_couple in query_file_feature_vectors]
    # query_file_candidate_vectors = [f_v_couple[1] for f_v_couple in query_file_feature_vectors]
    model = get_model()
    result = extract_features_query_file(PATH_QUERY_FILE, model)
    print(result)
