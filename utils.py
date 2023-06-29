from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from collections import OrderedDict
import pandas as pd
import docx2txt
from string import punctuation
import spacy
import re
import os
# from multiprocessing import Pool

nlp = spacy.load("en_core_web_sm")


def select_profile():
    profiles = pd.read_csv("./Data/Profiles.csv", index_col="ID")
    id = input(f"Select any job profile from the list\n{profiles}\n\n>>>")
    selected_profile = profiles.Profile.iloc[int(id)-1]
    print(f"\nSelected profile is {selected_profile}")
    return id


def preProcess(text):
    text = re.sub(r'[^\x00-\x7f]', r' ',
                  text)  # remove unwanted chars
    text = re.sub('\n', ' ', text)  # remove next lines
    text = re.sub('[%s]' % re.escape(
        """!"$%&'()*,-/:;<=>?@[]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop:
            continue
        filtered_tokens.append(token.text)
    text = " ".join(filtered_tokens).strip()
    return text.lower()


def combine_features(df):
    columns = ['skills', 'experience', 'designation', 'degree']
    df_new = df.copy()
    df_new = df_new[columns]
    df_new['features combined'] = [
        ' '.join(row.astype(str)) for row in df_new[df_new.columns[0:]].values]
    # print(df_new['features combined'])
    return df_new['features combined']


def generate_jd_text(df, root_path):
    file_formats = ["pdf", "docx", "txt"]
    remove_columns = ['email', 'mobile number']
    try:
        for file_format in file_formats:
            jd_path = f"{root_path}/JD.{file_format}"
            if os.path.exists(jd_path):
                jd_dict = extractor(df, jd_path, root_path)
                jd_text = ' '.join(
                    [(value+value) for key, value in jd_dict.items() if not key in remove_columns])
                df_jd = pd.concat([pd.Series(v, name=k)
                                   for k, v in jd_dict.items()], axis=1)
                df_jd.drop(columns=remove_columns, axis=1, inplace=True)
                return {'text': jd_text, 'df': df_jd}
    except:
        print("Invalid or empty J.D. file")
        return ""


def shortlist_resumes(root_path):
    model_df = pd.read_csv(
        f"{root_path}/Model Data.csv", encoding="unicode_escape")
    jd_result = generate_jd_text(model_df, root_path)
    jd_df = jd_result['df']
    jd_text = jd_result['text']
    df = parse_resumes(jd_df, root_path)

    features_combined = combine_features(df)
    sim_percent = []
    for index, resume in enumerate(features_combined):
        # # Vectorizing our features using CountVectorizer and generating the count matrix
        cv = CountVectorizer()
        try:
            count_matrix = cv.fit_transform([resume, jd_text])
            match_percentage = cosine_similarity(count_matrix)[0][1]*100
        except:
            match_percentage = 0.0
        sim_percent.append(round(match_percentage, 2))
    df['match percentage'] = sim_percent
    df = df.sort_values(by="match percentage",
                        ascending=False).reset_index(drop=True)

    final_df = df.filter(['ID', 'match percentage',
                         'mobile number', 'email'], axis=1)
    # final_df = final_df.head(n=10)
    print(
        f"{len(df['match percentage'])} RESUMES SIMILARITY PERCENTAGE TO THE PROVIDED JOB DESCRIPTION :\n{final_df}")

    final_df.to_csv(f'{root_path}/Shortlisted/Shortlisted.csv',
                    encoding='utf8', index=False)


def resumeParser(path_file):
    try:
        if ".pdf" in path_file:
            text = extract_text(path_file)
        elif ".doc" in path_file or ".docx" in path_file:
            text = docx2txt.process(path_file)
        elif ".txt" in path_file:
            with open(path_file, encoding='utf8') as f:
                text = f.read()
        return text
    except:
        print("only PDF, DOCX or TEXT file are valid")
        return "N/A"


def extract_model_data_words(df):
    # df = df.apply(preProcess)
    df_dict = {}
    for column in df.columns:
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(str.lower)
        list_column_words = list(df[column])
        separated_words = " ".join(list_column_words).strip().split()
        df_dict[column] = list(OrderedDict.fromkeys(separated_words))
    return df_dict


def separateEntities(df, text):
    dict_resume = {}
    words = text.lower().split()
    df_dict = extract_model_data_words(df)
    for column in df.columns:
        column_words = df_dict[column]
        ents_found = []
        for column_word in column_words:
            if column_word in words:
                if column_word in punctuation:
                    continue
                ents_found.append(column_word)
        ents_found = list(OrderedDict.fromkeys(ents_found))
        text = " ".join(ents_found).strip()
        dict_resume[column.lower()] = text
    return dict_resume


def get_personal_ents(text):
    dict_resume = {}
    phonenum_regex = r'[\+]?\d{10}|\(\d{3}\)\s?-\d{6}'
    email_regex = r'[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+'
    mob_num = re.findall(phonenum_regex, text)
    email_found = re.findall(email_regex, text)
    try:
        dict_resume["mobile number"] = mob_num[0]
    except:
        dict_resume["mobile number"] = ""
    try:
        email = email_found[0]
        if email[0] in string.punctuation:
            email = email[1:]
        dict_resume["email"] = email
    except:
        dict_resume["email"] = ""
    return dict_resume


def extractor(df, path_file, root_path):
    extracted = resumeParser(path_file)
    personal_ents = get_personal_ents(extracted)
    pre_processed_text = preProcess(extracted)
    features_ent = separateEntities(df, pre_processed_text)
    res_parsed = personal_ents | features_ent
    # res_parsed["Cleaned Text"] = pre_processed_text
    return res_parsed


def parse_resumes(df, root_path):
    print('\nPARSING RESUME(s)\n')
    res_dict = {}
    resumes_path = []

    files_resume = os.listdir(f"{root_path}/Resumes")
    for file_name in files_resume:
        path_file = f"{root_path}/Resumes/{file_name}"
        resumes_path.append(path_file)

    # # create the process pool
    # results = {}
    # with Pool(processes=5) as pool:
    #     # call the same function with different data in parallel
    #     for index, result in enumerate(pool.starmap(extractor, resumes_path)):
    #         # report the value to show progress
    #         results[index+1] = result

    for index, path_file in enumerate(resumes_path):
        res_parsed = extractor(df, path_file, root_path)
        res_dict[index+1] = res_parsed

    df = pd.DataFrame.from_dict(res_dict)
    df = df.T
    df = df.rename_axis('ID').reset_index()
    df.to_csv(f'{root_path}/Parsed/Parsed.csv',
              encoding='utf8', index=False)
    return df
