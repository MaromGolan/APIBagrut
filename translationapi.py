from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import flask
from flask import request, jsonify
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>RAVELTRANSLATE</h1>
                <p> Ravel-Translate's Translator   </p>'''

def api_translation():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    getarticle = request.get_json()
    article = """you've gotta dance like theres nobody watching,
    Love like you'll never be hurt,
    Sing like there's nobody listening,
    And live like it's heaven on earth"""
    #print(article)
    inputs = tokenizer(getarticle, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["heb_Hebr"], max_length=500)
    #print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
    translation = string(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
    return  jsonify(translation)@app.route('/', methods=['GET'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
















#@article{nllb2022,
 # title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  #author={{NLLB Team} and Costa-jussà, Marta R. and Cross, James and Çelebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Mejia-Gonzalez, Gabriel and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzmán, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff},
#  year={2022}
#}





