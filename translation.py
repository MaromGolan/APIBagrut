from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
article = """you've gotta dance like theres nobody watching,Love like you'll never be hurt,
Sing like there's nobody listening,
And live like it's heaven on earth"""
print(article)
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["heb_Hebr"], max_length=500)

print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])



"""
tokenizer1 = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",src_lang="heb_Hebr")
model1 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
article1 = """'בתי הביולוגית'"""
print(article1)
inputs1 = tokenizer1(article1, return_tensors="pt")
translated_tokens1 = model1.generate(**inputs1, forced_bos_token_id=tokenizer1.lang_code_to_id["eng_Latn"], max_length=30)
print(tokenizer1.batch_decode(translated_tokens1, skip_special_tokens=True)[0])

"""

"""
tokenizer1 = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",src_lang="heb_Hebr")

model1 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

article1 = """'הרצפה הייתה מלאה בכדורים לא של כדורסל ולא של אקמול'"""
print(article1)
inputs1 = tokenizer1(article1, return_tensors="pt")

translated_tokens1 = model.generate(**inputs1, forced_bos_token_id=tokenizer1.lang_code_to_id["eng_Latn"], max_length=30)

print(tokenizer1.batch_decode(translated_tokens1, skip_special_tokens=True)[0])"""


#@article{nllb2022,
 # title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  #author={{NLLB Team} and Costa-jussà, Marta R. and Cross, James and Çelebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Mejia-Gonzalez, Gabriel and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzmán, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff},
#  year={2022}
#}

