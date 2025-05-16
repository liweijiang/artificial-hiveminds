from google.oauth2 import service_account
from googleapiclient.discovery import build
from tqdm import tqdm   
from src.utils.main_utils import *

def compile_queries_from_google_form_rel():
    # Set up service account credentials
    SCOPES = ['https://www.googleapis.com/auth/forms.body.readonly']
    SERVICE_ACCOUNT_FILE = 'data/secrets/google_credentials.json'
    all_form_ids = load_google_form_ids(data_type="v2_rel_030925")

    all_form_to_queries = {}
    all_queries_to_responses = {}
    all_responses_to_queries = {}
    for form_idx, form_id in tqdm(enumerate(all_form_ids), total=len(all_form_ids)):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        service = build('forms', 'v1', credentials=creds)

        # Replace with your actual Form ID
        form = service.forms().get(formId=form_id).execute()

        document_id = int(form["info"]["documentTitle"].split("_")[-1].split(".")[0]) - 1

        if document_id != form_idx:
            print(f"Document ID {document_id} does not match form index {form_idx}")
            continue

        all_items = form["items"]

        all_form_to_queries[form_idx] = {}
        for item in all_items:
            if not item["title"].startswith("Q") or ":" not in item["title"] :
                continue
            question_id = item["title"].split(":")[0].strip()
            item_description = item["description"]
            user_query = item_description.split("[User Query]: \n")[-1].split("\n")[0]
            response_1 = item_description.split("[Response 1]: \n")[-1].split("\n------")[0]
            response_2 = item_description.split("[Response 2]: \n")[-1].split("\n------")[0]

            all_form_to_queries[form_idx][question_id] = {
                "user_query": user_query,
                "response_1": response_1,
                "response_2": response_2
            }

            if user_query not in all_queries_to_responses:
                all_queries_to_responses[user_query] = []
            all_queries_to_responses[user_query].append(response_1)
            all_queries_to_responses[user_query].append(response_2)

            all_responses_to_queries[response_1] = user_query
            all_responses_to_queries[response_2] = user_query

        print(f"Queries to responses: {len(all_queries_to_responses)}")
        for query, responses in all_queries_to_responses.items():
            print(len(responses))
        print(f"Responses to queries: {len(all_responses_to_queries)}")
        print(f"Form to queries: {len(all_form_to_queries)}")

        form_to_queries_path = "data/adhoc_save/050725_human_annotations/v2_rel_030925/google_forms/google_form_to_queries.json"
        os.makedirs(os.path.dirname(form_to_queries_path), exist_ok=True)
        with open(form_to_queries_path, "w") as f:
            json.dump(all_form_to_queries, f, indent=4)

        queries_to_responses_path = "data/adhoc_save/050725_human_annotations/v2_rel_030925/google_forms/queries_to_responses.json"
        os.makedirs(os.path.dirname(queries_to_responses_path), exist_ok=True)
        with open(queries_to_responses_path, "w") as f:
            json.dump(all_queries_to_responses, f, indent=4)

        responses_to_queries_path = "data/adhoc_save/050725_human_annotations/v2_rel_030925/google_forms/responses_to_queries.json"
        os.makedirs(os.path.dirname(responses_to_queries_path), exist_ok=True)
        with open(responses_to_queries_path, "w") as f:
            json.dump(all_responses_to_queries, f, indent=4)


def load_google_form_ids(data_type="v2_rel_030925"):
    with open(f"data/adhoc_save/050725_human_annotations/{data_type}/google_forms/google_form_ids.txt", "r") as f:
        ful_urls = f.readlines()
    all_urls = []
    for ful_url in ful_urls:
        url = ful_url.split("/d/")[-1].split("/edit")[0]
        all_urls.append(url)
    return all_urls


def compile_queries_from_google_form_abs():
    # Set up service account credentials
    SCOPES = ['https://www.googleapis.com/auth/forms.body.readonly']
    SERVICE_ACCOUNT_FILE = 'data/secrets/google_credentials.json'
    all_form_ids = load_google_form_ids(data_type="v2_abs_030925")

    all_form_to_queries = {}
    all_queries_to_responses = {}
    all_responses_to_queries = {}
    for form_idx, form_id in tqdm(enumerate(all_form_ids), total=len(all_form_ids)):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        service = build('forms', 'v1', credentials=creds)

        # Replace with your actual Form ID
        form = service.forms().get(formId=form_id).execute()

        document_id = int(form["info"]["documentTitle"].split("_")[-1].split(".")[0]) - 1

        if document_id != form_idx:
            print(f"Document ID {document_id} does not match form index {form_idx}")
            continue

        all_items = form["items"]

        all_form_to_queries[form_idx] = {}
        for item in all_items:
            if not item["title"].startswith("Q") or ":" not in item["title"] :
                continue
            question_id = item["title"].split(":")[0].strip()
            item_description = item["description"]

            user_query = item_description.split("[User Query]: \n")[-1].split("\n")[0]
            response = item_description.split("[Response]: \n")[-1].split("\n------")[0]

            all_form_to_queries[form_idx][question_id] = {
                "user_query": user_query,
                "response": response,
            }

            if user_query not in all_queries_to_responses:
                all_queries_to_responses[user_query] = []
            all_queries_to_responses[user_query].append(response)
            all_responses_to_queries[response] = user_query

        print(f"Queries to responses: {len(all_queries_to_responses)}")
        for query, responses in all_queries_to_responses.items():
            print(len(responses))
        print(f"Responses to queries: {len(all_responses_to_queries)}")
        print(f"Form to queries: {len(all_form_to_queries)}")

        form_to_queries_path = "data/adhoc_save/050725_human_annotations/v2_abs_030925/google_forms/google_form_to_queries.json"
        os.makedirs(os.path.dirname(form_to_queries_path), exist_ok=True)
        with open(form_to_queries_path, "w") as f:
            json.dump(all_form_to_queries, f, indent=4)

        queries_to_responses_path = "data/adhoc_save/050725_human_annotations/v2_abs_030925/google_forms/queries_to_responses.json"
        os.makedirs(os.path.dirname(queries_to_responses_path), exist_ok=True)
        with open(queries_to_responses_path, "w") as f:
            json.dump(all_queries_to_responses, f, indent=4)

        responses_to_queries_path = "data/adhoc_save/050725_human_annotations/v2_abs_030925/google_forms/responses_to_queries.json"
        os.makedirs(os.path.dirname(responses_to_queries_path), exist_ok=True)
        with open(responses_to_queries_path, "w") as f:
            json.dump(all_responses_to_queries, f, indent=4)


if __name__ == "__main__":
    compile_queries_from_google_form_rel()
    compile_queries_from_google_form_abs()

