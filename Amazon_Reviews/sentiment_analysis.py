# First attempt of analyzing a data set with a sentiment analyzer (VADER)
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt

analyser = SentimentIntensityAnalyzer()

dataFrame = pd.read_csv("./tripadvisor_hotel_reviews.csv")
x = np.linspace(0,len(dataFrame),len(dataFrame))

listOfReviews = dataFrame['Review'].to_list()

def generatePolarities(lst) -> list:
    rlst = []
    for i in lst:
        rlst.append(analyser.polarity_scores(i))
    return rlst

def extractPolarities(ilst,choice) -> list:
    rlst = []
    for i in ilst:
        rlst.append(i[choice])
    return rlst

rawPolarityList = generatePolarities(listOfReviews)
polarityList = extractPolarities(rawPolarityList,'compound')
posPolarityList = extractPolarities(rawPolarityList,'pos')

print("\nPlotting the complete figure plots")
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.title("Compound Polarity")
plt.plot(x,polarityList,'.')
plt.subplot(122)
plt.title("Positive Polarity")
plt.plot(x,posPolarityList,'.')
plt.show()

# Future task -> see if the polarity reflects the balance
# From the analysis it would seem that the reviews are mostly positive