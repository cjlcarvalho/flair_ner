from flair.data_fetcher import NLPTaskDataFetcher
from flair.models.text_classification_model import TextClassifier
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.data import Sentence
from flair.trainers import ModelTrainer

import argparse
import os


def train(corpus_dir: str, output_dir: str):

    corpus = NLPTaskDataFetcher.load_classification_corpus(
        corpus_dir, train_file="train.txt", test_file="test.txt", dev_file="dev.txt"
    )

    label_dict = corpus.make_label_dictionary()

    glove_embedding = WordEmbeddings("glove")

    embeddings = [glove_embedding]

    document_embedding = DocumentRNNEmbeddings(
        embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256
    )

    model = TextClassifier(
        document_embedding, label_dictionary=label_dict, multi_label=False
    )

    trainer = ModelTrainer(model, corpus)

    trainer.train(
        output_dir,
        learning_rate=0.1,
        mini_batch_size=32,
        anneal_factor=0.5,
        patience=5,
        max_epochs=150,
    )


def predict(content: str, model_dir: str):

    sentence = Sentence(content)

    model = TextClassifier.load_from_file(os.path.join(model_dir, "final-model.pt"))
    model.predict(sentence)

    print(sentence.labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    train_group = group.add_argument_group("train")
    train_group.add_argument("--train_corpus_dir", type=str)
    train_group.add_argument("--output_model_dir", type=str)

    test_group = group.add_argument_group("test")
    test_group.add_argument("--predict_sentence", type=str)
    test_group.add_argument("--load_model_dir", type=str)

    args = parser.parse_args()

    if args.train_corpus_dir:
        train(args.train_corpus_dir, args.output_model_dir)
    elif args.predict_sentence:
        predict(args.predict_sentence, args.load_model_dir)
