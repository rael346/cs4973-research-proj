import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image

ANNOTATION = 'annotation'
GALLERY = 'gallery'
QUERY = 'query'
APPAREL_TRAIN_ANNOTATION_PATH = '../dataset/apparel_train_annotation.csv'

def get_model(finetuned=False):
  model = SentenceTransformer('clip-ViT-B-32')
  # TODO finetuning?
  if finetuned:
    pass
  return model

# Returns a list of 3-tuples containing 1 query feature vector (1 source image vector added to 1 feedback
# vector), 1 target feature vector, and 1 non-target feature vector. All 3 feature vectors are of size 128
def extract_features_training(annotation_df):
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
  annotation_df = pd.read_csv(APPAREL_TRAIN_ANNOTATION_PATH)
  feature_vectors = extract_features_training(annotation_df.iloc[0:1])

  # separate the triplets into separate lists
  query_vectors = [f_v_triplet[0] for f_v_triplet in feature_vectors]
  target_vectors = [f_v_triplet[1] for f_v_triplet in feature_vectors]
  non_target_vectors = [f_v_triplet[2] for f_v_triplet in feature_vectors]

  print("Number of query vectors: " + str(len(query_vectors)))
  print("Number of target vectors: " + str(len(target_vectors)))
  print("Number of non target vectors: " + str(len(non_target_vectors)))
  print("Size of feature vector: " + str(len(query_vectors[0])))
  print("Meaningless right now, but cosine similarity of the first query vector and target vector: " + str(util.cos_sim(query_vectors[0], target_vectors[0])[0][0]))
  print("Same thing but for query vector and nontarget vector: " + str(util.cos_sim(query_vectors[0], non_target_vectors[0])[0][0]))
