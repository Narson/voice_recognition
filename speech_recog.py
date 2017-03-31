# -*- coding: utf8 -*-
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
from numpy.linalg import inv
from dpcore import dp
import numpy

#obliczenie dwóch macierzy mfcc z plików wav

def get_distance(file1, file2):
    #czytanie pliku
    (rate,sig) = wav.read(file1)
    #funkcja mfcc zwraca macierz
    mfcc_feat = mfcc(sig,rate)

    #to samo dla drugiego pliku
    (rate,sig) = wav.read(file2)
    mfcc_feat2 = mfcc(sig,rate)

    #obliczenie roznicy długości miacierzy
    mfcc_diff = abs(mfcc_feat.shape[0] - mfcc_feat2.shape[0])

    #przypisanie na poczatek dluzszej a potem krotszej macierzy
    #wszystke macierze mają wymiary N, 13, bo liczba wektorów moze się rożnić w zależnoći od próbki ale skłądniki z mfcc sa ustawione defaultowo na 13
    ma = mfcc_feat if mfcc_feat.shape[0] >= mfcc_feat2.shape[0] else mfcc_feat2
    mb = mfcc_feat2 if mfcc_feat.shape[0] >= mfcc_feat2.shape[0] else mfcc_feat

    
    #print 'ma ', ma.shape
    #print 'mb ', mb.shape

    distances = []
    #liczenie euklidesowego dysnansu dla każdej z próbek dla dwóch macierzy
    #zwracany minimalny i maksymalny średni dystans
    for start in range(0, mfcc_diff+1):
        calculated = []
        for idx, f1 in enumerate(ma[start:]):
            
            if idx >= mb.shape[0]:
                break
            calculated.append(numpy.linalg.norm(f1-mb[idx]))
        if len(calculated) > 0:    
            distances.append(numpy.average(calculated))#sum(calculated)/len(calculated))

    #print distances

    return min(distances), max(distances)

#funkcja porównująca dystansy dla niznanego pliku i dla próbek
def compare_with(file_to_compare):
    file_distances = []
    for i in range(0, 16):
        f = 'y\\y%d.wav' % i
        file_distances.append(get_distance(f, file_to_compare)[0])
    return numpy.average(file_distances)#sum(file_distances)/len(file_distances)


if __name__ == "__main__":
    # krzyzowe porownanie dystansów, wyciągniecie statystki
    values = []
    for i in range(0, 16):
        f = 'y\\y%d.wav' % i
        values.append(compare_with(f))

    #tutaj wpisuje sie nazwe pliku, ktory chce sie porownac z wzorcowymi
    file_name = 'gloski\\y_test3.wav'
    
    maximum = max(values)
    minimum = min(values)
    average = numpy.average(values)
    current_value = compare_with(file_name)
    
    print(minimum, maximum, average)
    
    fit = 100 - abs(average - current_value)

    print("value:", current_value)

    if current_value < maximum and current_value > minimum:
        print("That is my Y. ", fit, "% fit")
    else:
        print("That is not my Y. ", fit, "% fit")
                 
