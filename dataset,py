from datasets import load_dataset

def load_data():
    dataset = load_dataset("multi_news")

    data_train = dataset['train']
    data_validation = dataset['validation']
    data_test = dataset['test']

    return data_train, data_validation, data_test


def main():
    data_train, data_validation, data_test = load_data()

    documents = []
    summaries = []
    for entry in data_train:
        document = entry["document"]
        summary = entry["summary"]

        # Append the document and summary to their respective lists
        documents.append(document)
        summaries.append(summary)

    return 0



if __name__ == "__main__":
    main()
