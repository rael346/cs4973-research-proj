from sentence_transformers import util
import datetime
import pandas as pd
from feature_extract import PATH_QUERY_FILE, extract_features_query_file

PATH_RESULTS_SAVE = '../results/scored_query_file' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jsonl'

# Given a query file data frame, returns a copy of the data frame with score values
# attached to each candidate. The score is the cosine similarity of its query vector
# and its candidate vector
def evaluate(query_df):
    query_df_scored = query_df.copy(deep=True)
    query_file_feature_vectors = extract_features_query_file(query_df)
    assert(len(query_file_feature_vectors) == len(query_df))

    # TODO speed this up with matrix-vector multiplication?
    for index, f_v in enumerate(query_file_feature_vectors):
        query_vector = f_v[0]           # remember, f_v is a 2-tuple
        candidate_vectors = f_v[1]
        for c_v_index, c_v in enumerate(candidate_vectors):
            score = util.cos_sim(query_vector, c_v)
            query_df_scored.iloc[index]['candidates'][c_v_index]['score'] = score
    return query_df_scored

if __name__ == "__main__":
    query_df = pd.read_json(PATH_QUERY_FILE, lines=True)
    scored = evaluate(query_df)
    scored.to_json(path_or_buf=PATH_RESULTS_SAVE, orient='records', lines=True)
    