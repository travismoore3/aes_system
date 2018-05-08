# aes_system
This is a program I wrote to score ESL essays automatically.
feature_extractor.py takes an input file of essays and extracts features from them and prints those to a tab separated output file (output_file.tsv).
train_model.py takes the output file and builds an ML model based on the extracted features. Then it trains on 90% of the files and tests on 10%. It then prints out some evaluation stats. 