#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: majamarjamaa
"""
import numpy as np
from matplotlib import *
from pylab import *
import pandas as pd
import math
from scipy.optimize import curve_fit


#leser covid-data fra url
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url)
total = df.drop(['Province/State', 'Lat', 'Long'], axis=1).groupby('Country/Region').sum().max(axis=1).sort_values(ascending=False)
countries = total.index.to_list()
smitte = pd.DataFrame()


#reformaterer dataen 
for country in countries:
    tid = df[df['Country/Region'] == country][df.columns[4:]].T.sum(axis=1)
    tid.index = pd.to_datetime(tid.index) 
    tid = tid.to_frame(country)
    smitte = pd.concat([smitte, tid], axis=1)
    

#henter ut data fra interessante land
norge_full = smitte["Norway"].values.tolist()
sørafrika_full = smitte["South Africa"].values
sverige_full = smitte["Sweden"].values
kina_full = smitte["China"].values
usa_full = smitte["US"].values


#lager ny liste med utvikling etter første smittetilfelle
def nyListe(x,y):
    for i in range(len(x)):
        if (x[i]!=0):
            y.append(x[i])

norge = []
nyListe(norge_full,norge)

sørafrika = []
nyListe(sørafrika_full,sørafrika)

sverige = []
nyListe(sverige_full,sverige)

kina = []
nyListe(kina_full,kina)

usa = []
nyListe(usa_full,usa)



#funksjon som finner minste antallet observasjoner (for plotting)
def minstVerdi(a):
    u = len(a[0])
    for i in range(1,len(a),1):
        if (len(a[i])<u):
            u = len(a[i])
    return u

#plotter utvikling (antall dager etter første tilfelle)
liste = [norge,sørafrika,sverige,kina,usa] #liste for å finne mulig x-akse

tid = linspace(0,minstVerdi(liste),minstVerdi(liste))
plot(tid,norge[0:minstVerdi(liste)], label="Norge" )
plot(tid,sørafrika[0:minstVerdi(liste)], label = "Sør Afrika")
plot(tid,sverige[0:minstVerdi(liste)], label = "Sverige")
plot(tid,kina[0:minstVerdi(liste)], label = "Kina")
plot(tid,usa[0:minstVerdi(liste)], label = "USA")
xlabel("Dager etter første smittetilfelle")
title("Antall smittede")
legend()
show()


#glatter kurvene med glidende gjennomsnitt
def snitt(x,y,z):
    start = x
    slutt = len(y)-x
    for i in range (start,slutt):
        z.append(mean(y[(i-x):(i+x)]))

k = 10

norgeSnitt = []
snitt(k, norge, norgeSnitt)

sørafrikaSnitt = []
snitt(k, sørafrika, sørafrikaSnitt)

sverigeSnitt = []
snitt(k, sverige, sverigeSnitt)

kinaSnitt = []
snitt(k, kina, kinaSnitt)

usaSnitt = []
snitt(k, usa, usaSnitt)

#plotter glattede kurver
liste2 = [norgeSnitt,sørafrikaSnitt,sverigeSnitt,kinaSnitt,usaSnitt] #liste for å finne mulig x-akse

tid = linspace(0,minstVerdi(liste2),minstVerdi(liste2))
plot(tid,norge[0:minstVerdi(liste2)], label="Norge" )
plot(tid,sørafrika[0:minstVerdi(liste2)], label = "Sør Afrika")
plot(tid,sverige[0:minstVerdi(liste2)], label = "Sverige")
plot(tid,kina[0:minstVerdi(liste2)], label = "Kina")
plot(tid,usa[0:minstVerdi(liste2)], label = "USA")
xlabel("Dager etter første smittetilfelle")
title(f"Antall smittede glidende gjennomsnitt, k= {k}")
legend()
show()


#leser befolkningsdata fra excel
befolkning = pd.read_excel("Befolkningstall_2020.xlsx")

Land = befolkning["Land"].values.tolist()
Befolkning = befolkning["Befolkning"].values.tolist()

def befolkningstall(x):
    for i in range(0,len(Land),1):
        if (Land[i] == x):
            y = Befolkning[i]
            return  y

norgeB = befolkningstall("Norway")
sørafrikaB = befolkningstall("South Africa")
sverigeB = befolkningstall("Sweden")
kinaB = befolkningstall("China")
usaB = befolkningstall("United States")


#finner antall smittede per capita i landet
def perCapita(x,y,z):
    for i in range(0,len(x),1):
        z.append(x[i]/y)

norgeCapita = []
perCapita(norgeSnitt,norgeB,norgeCapita)

sørafrikaCapita = []
perCapita(sørafrikaSnitt,sørafrikaB,sørafrikaCapita)

sverigeCapita = []
perCapita(sverigeSnitt,sverigeB,sverigeCapita)

kinaCapita = []
perCapita(kinaSnitt,kinaB,kinaCapita)

usaCapita = []
perCapita(usaSnitt,usaB,usaCapita)


#plotter utviklingen per capita
liste3 = [norgeCapita, sørafrikaCapita, sverigeCapita, kinaCapita, usaCapita]
tid = linspace(0,minstVerdi(liste3),minstVerdi(liste3)) 
plot(tid,norgeCapita[0:minstVerdi(liste3)], label="Norge" )
plot(tid,sørafrikaCapita[0:minstVerdi(liste3)], label = "Sør Afrika")
plot(tid,sverigeCapita[0:minstVerdi(liste3)], label = "Sverige")
plot(tid,kinaCapita[0:minstVerdi(liste3)], label = "Kina")
plot(tid,usaCapita[0:minstVerdi(liste3)], label = "USA")
xlabel("Dager etter første smittetilfelle")
title("Antall smittede per capita")
legend()
show()


#deriverer Capita-listene
def der(a,b):
    for i in range(0,len(a)-1,1):
        deltay = a[i+1]-a[i]
        deltax = 1  #observasjonene er daglige
        b.append(deltay/deltax)

norgeDer = []
der(norgeCapita,norgeDer)

sørafrikaDer = []
der(sørafrikaCapita,sørafrikaDer)

sverigeDer = []
der(sverigeCapita,sverigeDer)

kinaDer = []
der(kinaCapita,kinaDer)


usaDer = []
der(usaCapita,usaDer)


#glatter ut den deriverte
norgeDerSnitt = []
snitt(k,norgeDer,norgeDerSnitt)


sørafrikaDerSnitt = []
snitt(k,sørafrikaDer,sørafrikaDerSnitt)

sverigeDerSnitt = []
snitt(k,sverigeDer,sverigeDerSnitt)

kinaDerSnitt = []
snitt(k,kinaDer,kinaDerSnitt)

usaDerSnitt = []
snitt(k,usaDer,usaDerSnitt)


#plotter den glatte deriverte
liste4 = [norgeDerSnitt, sørafrikaDerSnitt,sverigeDerSnitt,kinaDerSnitt,usaDerSnitt]
tid = linspace(0,minstVerdi(liste4),minstVerdi(liste4))
plot(tid,norgeDerSnitt[0:minstVerdi(liste4)], label="Norge" )
plot(tid,sørafrikaDerSnitt[0:minstVerdi(liste4)], label = "Sør Afrika")
plot(tid,sverigeDerSnitt[0:minstVerdi(liste4)], label = "Sverige")
plot(tid,kinaDerSnitt[0:minstVerdi(liste4)], label = "Kina")
plot(tid,usaDerSnitt[0:minstVerdi(liste4)], label = "USA")
xlabel("Dager etter første smittetilfelle")
title(f"Endringen i antall smittede per capita glidende gjennomsnitt, k = {k}")
legend()
show()



#lager en logistisk modell for å forsøke å uttrykke smitteveksten i Norge
#finner først startverdiene a, b og c
c_s = max(norge)
a_s = c_s - 1

#finner b_s
y_k = 720000   #velger en verdi som ligger i midten av y-verdiene
for i in range(0,len(norge),1):
    if (norge[i] > y_k):
        x_k = i
        break
b_s = (1/x_k)*(log((a_s*y_k)/(c_s-y_k)))

#definerer logistisk funksjon
def f(x,a,b,c):
    return c/(1+a*(e**(-b*x)))

antallDager = []
for i in range(0,len(norge)):
    antallDager.append(i)
    
#bestemmer og skriver ut a b og c
[a,b,c] = curve_fit(f, antallDager, norge, p0=[a_s,b_s,c_s])[0]
print("a=", round(a,1))
print("b=", round(b,2))
print("c=", round(c,1))

#plotter den logistiske kurven
plot(antallDager, norge)
x = linspace(0,len(norge),len(norge))
plot(x,f(x,a,b,c))
title(f"Logistisk modell med valgt Y_k = {y_k}")
xlabel("Antall dager etter første smittetilfelle")
show()


#beregner minste kvadrat for utledet modell
Q = 0
for i in range(0,len(norge),len(norge)):
    Q1 = Q + (norge[i]-f(i,a,b,c))**2
print(f"Minste kvadrat for modellen er {Q1}")        
        



#estimerer y_k som gir lavest minste kvadrat
Q = 0
Qm = 1
y_k_m = 0
for y_k in range(200000,1400000,1000):
    for i in range(0,len(norge),1):
        if (norge[i] > y_k):
            x_k = i
            break
    b_s = (1/x_k)*(log((a_s*y_k)/(c_s-y_k)))
    [a,b,c] = curve_fit(f, antallDager, norge, p0=[a_s,b_s,c_s])[0]
    
    for i in range(0,len(norge),len(norge)):
        Q1 = Q + (norge[i]-f(i,a,b,c))**2
    if (Q1<Qm):
        Qm = Q1
        y_k_m = y_k

print("")
print(f"y_k som gir best modell: {y_k_m}")
print(f"minste kvadrat ved beste mulige modell: {Qm}")


#plotter den nye logistiske kurven
#finner b_s
y_k = y_k_m   
for i in range(0,len(norge),1):
    if (norge[i] > y_k):
        x_k = i
        break
b_s = (1/x_k)*(log((a_s*y_k)/(c_s-y_k)))

#definerer logistisk funksjon
def f(x,a,b,c):
    return c/(1+a*(e**(-b*x)))

antallDager = []
for i in range(0,len(norge)):
    antallDager.append(i)
    
#bestemmer og skriver ut a b og c
[a,b,c] = curve_fit(f, antallDager, norge, p0=[a_s,b_s,c_s])[0]
print("a=", round(a,1))
print("b=", round(b,2))
print("c=", round(c,1))

#plotter den logistiske kurven
plot(antallDager, norge)
x = linspace(0,len(norge),len(norge))
plot(x,f(x,a,b,c))
title(f"Logistisk modell med valgt Y_k = {y_k}")
xlabel("Antall dager etter første smittetilfelle")
show()





