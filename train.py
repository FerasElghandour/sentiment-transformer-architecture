"""Train or evaluate the released model using fixed prepared data splits."""
from sentiment_transformer import SentimentTransformer
from training import main

def build_model(c):
    return SentimentTransformer(vocabulary=c["vocabulary"],width=c["width"],heads=c["heads"],layers=c["layers"],context=c["context"])

if __name__=="__main__":
    main(build_model,{'classes': 2, 'vocabulary': 20000, 'width': 128, 'heads': 4, 'layers': 4, 'context': 256},binary=False,tokens=True)
