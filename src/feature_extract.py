import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image

ANNOTATION = 'annotation'
GALLERY = 'gallery'
QUERY = 'query'
PATH_APPAREL_TRAIN_ANNOTATION = '../dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = '../dataset/query_file_released.jsonl'

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
    query_emb = embed_query(model, get_img_path(source_id, ANNOTATION), feedbacks) 
    target_emb = embed_image(model, get_img_path(target_id, ANNOTATION))
    non_target_emb = embed_image(model, get_img_path(non_target_id, ANNOTATION))

    feat_vec_triplet = (query_emb, target_emb, non_target_emb)
    feature_vectors.append(feat_vec_triplet)
  return feature_vectors

# Returns a list of 2-tuples containing 1 query feature vector (1 source image vector added to 1 feedback
# vector) and the list of candidate vectors. All feature vectors are of size 128
def extract_features_query_file(query_df):
  feature_vectors = []
  model = get_model(finetuned=False)

  for _index, row in query_df.iterrows():
    # get data from dataframe
    source_id = row['source_pid']
    feedback1 = row['feedback1']
    feedback2 = row['feedback2']
    feedback3 = row['feedback3']
    feedbacks = [feedback1, feedback2, feedback3]
    candidates = row['candidates']
    
    candidate_embs = []
    for candidate in candidates:
      candidate_id = candidate['candidate_pid'] 
      candidate_emb = embed_image(model, get_img_path(candidate_id, QUERY))
      candidate_embs.append(candidate_emb)

    # encode query (source image + 3 feedbacks), target image, non-target image
    query_emb = embed_query(model, get_img_path(source_id, QUERY), feedbacks) 

    feat_vec_couple = (query_emb, candidate_embs)
    feature_vectors.append(feat_vec_couple)
  return feature_vectors  

# Returns the image file path given the image name
# folder name should be either 'annotation', 'gallery', 'query'
def get_img_path(image_name, folder_name):
  assert(isinstance(image_name, str))
  assert(isinstance(folder_name, str))
  assert(folder_name in {'annotation', 'gallery', 'query'})
  return '../images/' + str(folder_name) +  '/' + image_name + '.jpg'

# Encode image using sentence transformer model into 128-dimensional embedding
def embed_image(model, image_path):
  emb = model.encode(Image.open(image_path))    # encoded as 512 dimensional
  size_128_emb = emb[0:emb.size:4]    # 'scale' down by taking every 4th value (for now)
  return size_128_emb

# Encode text feedbacks using sentence transformer model into 128-dimensional embedding
def embed_text(model, text):
  emb = model.encode(text)            # encoded as 512 dimensional
  size_128_emb = emb[0:emb.size:4]    # 'scale' down by taking every 4th value (for now)
  return size_128_emb

def embed_query(model, image_path, feedbacks):
  assert(len(feedbacks) == 3)
  src_emb = embed_image(model, image_path)
  # we simply add feedback embedding vectors (for now)
  text_emb = embed_text(model, feedbacks[0]) + embed_text(model, feedbacks[1]) + embed_text(model, feedbacks[2]) 
  # we simply add source and text vectors to get query embedding (for now)
  query_emb = src_emb + text_emb
  return query_emb


if __name__ == "__main__":
  # load the annotations dataframe
  annotation_df = pd.read_csv(PATH_APPAREL_TRAIN_ANNOTATION)
  annotation_feature_vectors = extract_features_annotation(annotation_df.iloc[0:1])

  # load the query file dataframe
  query_df = pd.read_json(PATH_QUERY_FILE, lines = True)
  query_file_feature_vectors = extract_features_query_file(query_df.iloc[0:1])

  # ANNOTATION components 
  annotation_query_vectors = [f_v_triplet[0] for f_v_triplet in annotation_feature_vectors]
  annotation_target_vectors = [f_v_triplet[1] for f_v_triplet in annotation_feature_vectors]
  annotation_non_target_vectors = [f_v_triplet[2] for f_v_triplet in annotation_feature_vectors]

  # QUERY FILE components 
  query_file_query_vectors = [f_v_couple[0] for f_v_couple in query_file_feature_vectors]
  query_file_candidate_vectors = [f_v_couple[1] for f_v_couple in query_file_feature_vectors]
