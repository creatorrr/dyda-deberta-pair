from simpletransformers.classification import ClassificationModel

model = ClassificationModel("deberta", "diwank/dyda-deberta-pair")
convert_to_label = lambda n: ["__dummy__ (0), inform (1), question (2), directive (3), commissive (4)".split(', ')[i] for i in n]

predictions, raw_outputs = model.predict([["Say what is the meaning of life?", "I dont know"]])
convert_to_label(predictions)  # inform (1)
