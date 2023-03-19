from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import json
import numpy as np


def load_mldataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    feature_matrix = np.array(data['Features'])  # convert list to numpy array
    target_vector = np.array(data['Targets'])  # convert list to numpy array

    return feature_matrix, target_vector

X, y = load_mldataset("C:\\AuthentiWrite\\datasets\\example_2_dataset.json")

# Define the mutual information-based feature selector
selector = SelectKBest(mutual_info_classif, k=10)

# Fit the selector
selector.fit(X, y)

# Get the selected features
selected_features = selector.get_support()

# Create a new feature matrix with only the selected features
X_selected = X[:, selected_features]

# Print the shape of the original and selected feature matrices
print(f"Original feature matrix shape: {X.shape}")
print(f"Selected feature matrix shape: {X_selected.shape}")

# To view the selected features, you can use the following approach

feature_names = np.array(["Count_Paragraphs", "Count_Sentences", "Count_Periods", "Count_QuestionMarks", "Count_ExclamationMarks", "Count_Colons", "Count_Semicolons", "Count_Commas", "Count_Ellipses", "Count_CapitalLetters", "Count_LowercaseLetters", "Density_Periods", "Density_QuestionMarks", "Density_ExclamationMarks", "Density_Colons", "Density_Semicolons", "Density_Commas", "Density_Ellipses", "Density_CapitalLetters", "Density_LowercaseLetters", "Avg_WordsPerSentence", "Count_1SyllableWords", "Count_2SyllableWords", "Count_3SyllableWords", "Count_4PlusSyllableWords", "Readability_FleshKincaid", "Readability_DaleChall", "Readability_GunningFog", "Readability_SMOG", "Readability_ColemanLiau", "Readability_FleschReadingEase", "Readability_AutomatedReadabilityIndex", "Readability_LinsearWrite", "Readability_FryReadabilityGraph", "Readability_Raygor", "Count_Words", "Count_UniqueWords", "Ratio_UniqueWords", "Count_UniqueBigrams", "Count_UniqueTrigrams", "Count_Characters", "Count_Syllables", "Avg_WordLength", "Count_Contractions", "Count_StudentWords", "SentimentValue", "SentimentStrength", "Count_EasyWords", "Count_HardWords", "Ratio_HardWords", "Count_Conjunctions", "Count_DiscourseMarkers", "Count_HedgeWords", "Statistics_ShannonEntropy", "LD_Gini_Words", "LD_Gini_Bigrams", "LD_Gini_Trigrams", "LD_TTR", "LD_VariationCoefficient", "LD_CarrollsCTTR", "LD_SichelS", "LD_DugastU", "LD_SummersS", "LD_MaasA", "LD_GuiraudR", "LD_BrunetW", "LD_HerdanC", "LD_YulesK", "LD_HonoresStatistic", "LD_SimpsonsDiversityIndex", "LD_BergerParkerIndex", "LD_HapaxLegomenaRatio", "LD_HapaxDislegomenaRatio", "LD_KBand50", "LD_KBand75", "LD_KBand100", "LD_RepeatingWordRatio", "Count_SlangWords", "Count_GrammaticalErrors"])
selected_feature_names = feature_names[selected_features]

print("Selected features:")
for feature_name in selected_feature_names:
    print(feature_name)