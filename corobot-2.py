
# -*- coding: utf-8 -*-

import nltk
import re
import string
import numpy as np
import random

f=open('data/infos_corona.txt','r',errors = 'ignore', encoding = "utf8")
texte=f.read()

#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
# on avait remarqué une phrase à tort à cause de l'acronyme n.c.a, on corrige : 
texte = re.sub('n.c.a.', 'nca', texte)

# Tokenisation en phrases et mots
phrases_token = nltk.sent_tokenize(texte, language = "french")

# on a beaucoup de questions ici : on les enlève
for i in sorted(range(len(phrases_token)), reverse = True):
    if re.search(r"\?", phrases_token[i]):
        del(phrases_token[i])

# on enlève les doublons
phrases_token = list(set(phrases_token)) 

# On crée une liste nettoyée mais qui ne sera pas celle dans laquelle
# on ira chercher les réponses, simplement pour la création de la
# matrice TF-IDF
def nettoyage(texte):
    texte = texte.lower()
    # on remplace covid-19 par coronavirus
    texte = re.sub('covid-19| virus|covid |sars-cov', 'coronavirus', texte)
    # on remplace les "coronavirus coronavirus" par coronavirus
    texte = re.sub('coronavirus coronavirus', 'coronavirus', texte)
    texte = re.sub(f"[{string.punctuation}]", " ", texte)
    texte = re.sub('[éèê]', 'e', texte)
    texte = re.sub('[àâ]', 'a', texte)
    texte = re.sub('[ô]', 'o', texte)
    texte = re.sub('mort(\w){0,3}|deces|deced(\w){1,5}', 'deces', texte)
    texte = re.sub('remedes?|traitements?|antidotes?', 'traitement', texte)
    texte = re.sub('medec(\w){1,5}|medic(\w){1,3}', 'medical', texte)
    return texte

phrases_nettoyees = []
for i in range(len(phrases_token)):
    phrases_nettoyees.append(nettoyage(phrases_token[i]))
      
# on récupère les stop words
from stop_words import get_stop_words
french_stop_words = get_stop_words('french')

# Stemmer : on prend la racine des mots, 
# lemmer : on fait quelque chose de plus "propre" : infinitif pour les verbes...
from nltk.stem.snowball import FrenchStemmer
french_stem = FrenchStemmer()

def stem_tokens(tokens):
    return [french_stem.stem(token) for token in tokens]
def stem_norm(text):
    return stem_tokens(nltk.word_tokenize(text))
    #return re.sub(f"[{string.punctuation}]", " ", stem_tok)


# générer des réponses à partir de la matrice tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tf_idf = TfidfVectorizer(tokenizer=stem_norm, stop_words = french_stop_words)
tf_idf_chat = tf_idf.fit(phrases_nettoyees)

def reponse_corona(user_sentence):
    user_sentence = [user_sentence]
    phrases_tf = tf_idf_chat.transform(phrases_nettoyees)
    user_tf = tf_idf_chat.transform(user_sentence)
    similarity = cosine_similarity(user_tf, phrases_tf).flatten()
    index_max_sim = np.argmax(similarity)
    if(similarity[index_max_sim] == 0):
        robo_response = "Je n'ai pas trouvé cette information, désolé!"
    elif(similarity[index_max_sim] <= 0.30):
        robo_response = """Je ne suis pas sûr d'avoir trouvé exactement ce que vous vouliez dire, voilà ce que j'ai trouvé : \n"""+phrases_token[index_max_sim] 
    else:
        simil_index = []
        for i in range(len(similarity)):
            if similarity[i] > 0.3:
                simil_index.append(i)
        robo_response = '\n'.join([phrases_token[i] for i in simil_index])
    return robo_response

# gérer les salutations
salutations_user = r"bonjour.*|salut.*|hello.*|hey.*|coucou.*|bonsoir.*"
salutations_robot = ["Bonjour, bienvenue sur ce chatbot!"]
def salutations(user_sentence):
    if re.fullmatch(salutations_user, user_sentence):
        return random.choice(salutations_robot)
# gérer les demandes de nouvelles
nouvelles_user = r".*[çs]a va.*\?|.*la pêche\?|.*la forme\?"
nouvelles_robot = ["Je suis un robot, ça va jamais vraiment",
                   "Un peu marre du confinement",
                   "On fait aller!",
                   "J'ai une pêche d'enfer!!!"]
def nouvelles(user_sentence):
    if re.fullmatch(nouvelles_user, user_sentence):
        return random.choice(nouvelles_robot)
    
# On définit une porte de sortie pour l'utilisateur
exit_user = ["au revoir", "bye", "bye bye", "à +", "ciao"]
exit_bot = ["au revoir!", "j'espère vous avoir été utile!","à une prochaine fois :)"]

# on définit enfin notee chatbot : 
flag = True
print("""> Corona-bot : Je suis le corona-bot, je réponds à vos questions sur l'épidémie ! 
Pour quitter, vous pouvez juste me dire au revoir""")
while(flag == True):
    user_sentence = input("> Vous :  ")
    user_sentence = user_sentence.lower()
    if (user_sentence == "infos"):
        user_info = input("> Vos informations :  ")
        phrases_token.append(user_info)
        user_info = user_info.lower()
        user_info = nettoyage(user_info)
        phrases_nettoyees.append(user_info) 
        tf_idf_chat = tf_idf.fit(phrases_nettoyees)
        print("Merci c'est noté!")
    elif not (user_sentence in exit_user):
        if (salutations(user_sentence) != None):
            print("> Corona-bot : " + salutations(user_sentence))
        elif (nouvelles(user_sentence) != None):
            print("> Corona-bot : " + nouvelles(user_sentence))
        else:
            user_sentence = nettoyage(user_sentence)
            print("> Corona-bot : " + reponse_corona(user_sentence))
    else:
        flag = False
        print("> Corona-bot : " + random.choice(exit_bot))
