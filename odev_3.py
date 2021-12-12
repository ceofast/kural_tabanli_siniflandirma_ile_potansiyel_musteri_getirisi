### KURAL TABANLI SINIFLANDIRMA İLE POTANSİYEL MÜŞTERİ GETİRİSİ HESAPLAMA ###
################################### VBO #####################################

# İŞ PROBLEMİ

# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based)
# yeni müşteri tanımları (persona) oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup
# bu segmentlere göre yeni gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceğini tahmin
# etmek istemektedir.

# Örneğin Türkiye'den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar
# kazandırabileceği belirlenmek isteniyor.

# VERİ SETİ HİKAYESİ

# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın
# alan kullanıcıların bazı demografik bilgilerini barındırmaktadır.

# Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir.

# Bunun anlamı tablo tekilleştirilmemiştir.

# Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# GÖREV 1 #

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

def load_persona():
    df = pd.read_csv(r"/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_2/persona.csv")
    return df

df = load_persona()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Değişkenler #
# PRICE   - Müşterinin harcama tutarı
# SOURCE  - Müşterinin bağlandığı cihaz türü
# SEX     - Müşterinin cinsiyeti
# COUNTRY - Müşterinin ülkesi
# AGE     - Müşterinin yaşı

# Veride 5000 gözlem birimi ve 5 değişken bulunmaktadır. Herhangi bir eksik veri ise bulunmamaktadır.

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].unique()
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# SOURCE değişkeninin android ve ios adında iki değişkeni bulunmaktadır. Frekansları ise;
# android: 2974
# ios    : 2026

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].unique()

# Soru 4: Hangi PRICE'dan kaç tane satış gerçekleşmiştir?
df["PRICE"].value_counts()
df["PRICE"].nunique()

# PRICE değişkeninin [29, 39, 49, 19, 59, 9] olarak 6 değişkeni bulunmaktadır. Gerçekleşen satış frekansları ise;
# 29 : 1305
# 39 : 1260
# 49 : 1031
# 19 : 992
# 59 : 212
# 9  : 200

# Soru 5: Hangi ülkeden kaç tane satış olmuştur?
# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmıştır?
df[["COUNTRY", "PRICE"]].head()
df[["COUNTRY", "PRICE"]].groupby("COUNTRY").agg(["count","sum"])
# Ülkelere göre satışlardan toplam şu kadar kazanılmıştır;

#bra       : 1496 adet satıştan 51354 kazanılmıştır.
#can       : 230  adet satıştan 7730 kazanılmıştır.
#deu       : 455  adet satıştan 15485 kazanılmıştır.
#fra       : 303 adet satıştan  10177 kazanılmıştır.
#tur       : 451 adet satıştan 15689 kazanılmıştır.
#usa       : 2065 adet satıştan 70225 kazanılmıştır.


# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df[["SOURCE","PRICE"]].groupby("SOURCE").agg(["count"])
# Ülkelere göre SOURCE frekanslarının satış sayıları şunlardır;
#android  : 2974
#ios      : 2026

# Soru 8: Ülkelere göre PRICE ortalamaları nelerdir?
df[["COUNTRY", "PRICE"]].groupby("COUNTRY").agg(["mean"])
# Ülkelere göre PRICE ortalamaları şunlardır;
# bra      34.327540
# can      33.608696
# deu      34.032967
# fra      33.587459
# tur      34.787140
# usa      34.007264

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df[["SOURCE", "PRICE"]].groupby("SOURCE").agg(["mean"])
# SOURCE'lara göre PRICE ortalamaları şunlardır;
# android  : 34.174849
# ios      : 34.069102

# Soru 10: COUNTRY - SOURCE kırılımında PRICE ortalamaları nedir?
df[["SOURCE","COUNTRY","PRICE"]].groupby(["COUNTRY","SOURCE"]).agg(["mean"])
# COUNTRY - SOURCE kırılımında PRICE ortalamaları aşağıdaki tabloda gösterilmiştir;
# bra     android  :34.387029
#         ios      :34.222222
# can     android  :33.330709
#         ios      :33.951456
# deu     android  :33.869888
#         ios      :34.268817
# fra     android  :34.312500
#         ios      :32.776224
# tur     android  :36.229437
#         ios      :33.272727
# usa     android  :33.760357
#         ios      :34.371703

# GÖREV 2 #

#       - COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg(["mean"])
# bra     android female 15   38.714286
#                        16   35.944444
#                        17   35.666667
#                        18   32.255814
#                        19   35.206897
#                               ...
# usa     ios     male   42   30.250000
#                        50   39.000000
#                        53   34.000000
#                        55   29.000000
#                        59   46.500000
# [348 rows x 1 columns]

# GÖREV 3 ve 4 #

#       - Çıktıyı PRICE'a göre sıralayınız.
#       - Index'te yer alan isimleri değişken ismine çeviriniz.
agg_df = pd.DataFrame(df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg(["mean"])).reset_index()
agg_df.head()

agg_df.columns = ["COUNTRY", "SOURCE", "SEX", "AGE", "PRICE"]
agg_df.sort_values("PRICE", ascending=False, inplace=True)
agg_df.head()

# GÖREV 5 #

#       - AGE değişkenini kategorik değişkene çeviriniz ve agg_df'ye ekleyiniz.
#       - AGE sayısal değişkenini kategorik değişkene çeviriniz.
#       - Aralıkları ikna edici şekilde oluşturunuz.
#       - Örneğin: "0_18", "19_23", "24_30", "31_40", 41_70"

agg_df["AGE"].max()
agg_df["AGE"].min()

pd.set_option('display.max_columns', None)
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins = [0, 18, 23, 30, 40, 66], labels = ['0_18', '19_23', '24_30', '31_40', '41_66'])
agg_df.head()

# GÖREV 6 #

#       - Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
#       - Yeni eklenecek değişkenin adı: customers_level_based
#       - Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based
# değişkenini oluşturmanız gerekmektedir.

agg_df["customers_level_based"] = [str(row[0]).upper() + "_" + str(row[1]).upper() + "_" + str(row[2]).upper() + "_" + str([5]).upper() for row in agg_df.values]
agg_df.head()

df_cat = agg_df[["customers_level_based", "PRICE"]]
df_cat.head()

df_cat = df_cat.groupby("customers_level_based").agg({"PRICE": "mean"})
df_cat = df_cat.reset_index()
df_cat.head()

# GÖREV 7 #

#       - Yeni müşterileri (personaları) segmentlere ayırınız.
#       - Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
#       - Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
#       - Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
#       - C segmentini analiz ediniz (Veri setinden sadece C segmentini çekip analiz ediniz).

df_cat["SEGMENT"] = pd.qcut(df_cat["PRICE"], 4, labels=["D", "C", "B", "A"])
df_cat.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})
df_cat.loc[df_cat['SEGMENT'] == 'C']

# SEGMENT
# D        31.736938  32.909615  190.421628
# C        33.762553  34.092250  202.575319
# B        34.628458  35.193482  207.770749
# A        36.122921  36.918483  216.737527

#                                 PRICE SEGMENT
# 2       BRA_IOS_FEMALE_[5]  33.915453       C
# 7         CAN_IOS_MALE_[5]  33.568668       C
# 9     DEU_ANDROID_MALE_[5]  33.849489       C
# 20  USA_ANDROID_FEMALE_[5]  34.024831       C
# 21    USA_ANDROID_MALE_[5]  34.092250       C
# 22      USA_IOS_FEMALE_[5]  33.124630       C

# GÖREV 8 #

#       - Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve ne kadar gelir getirebileceğini tahmin ediniz.
#       - 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
#       - 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?

def new_customer(dataframe,new_user):
    print(dataframe[dataframe["customers_level_based"] == new_user])

new_customer(agg_df,"TUR_ANDROID_FEMALE_31_40")

new_customer(agg_df,"FRA_IOS_FEMALE_31_40")

