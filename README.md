# Užduotis

Sukurkite duomenų produktą - analitinę aplikaciją, skirtą banko paskolos įvertinimui mašininio mokymosi algoritmų pagalba.

<img src="practice/img/data_product.png" width="500"/>

#### Projekte ugdomi gebėjimai

-   analizuoti ir interpretuoti skaičiavimų rezultatus, įvertinti ar galima taikyti duomenų tyrybos metodą konkretiems duomenims ir jeigu ne, pasiūlyti kitą tinkamą metodą;\
-   apibendrinti gautus rezultatus ir pateikti pagrįstas išvadas ir prognozes, rengti ataskaitas;\
-   atlikti duomenų tyrybą panaudojant duomenų tyrybos programinę įrangą;
-   paaiškinti naudotų duomenų tyrybos metodų sąvokas ir teoriją ;
-   sudaryti duomenų tyrybos modelius realiems didiesiems verslo duomenims tirti.

#### Projekto vertinimo kriterijai

1.  Mokslinės literatūros tinkamumas projekto temai.\
2.  Domenų tyrybos metodų parinkimas projekto uždavinių sprendimui.
3.  Sukurtų duomenų tyrybos modelių tikslumas ir kokybė
4.  Modelių programų kokybė.
5.  Gautų rezultatų patikimumas, interpretavimo teisingumas, išvadų pagrįstumas.
6.  Rekomendacijos sukurtų modelių tobulinimui.
7.  Projekto ataskaitos atitikimas nustatytiems reikalavimams.

#### Reikalingi įrankiai projektui

-   R (shiny, tidyverse, rmarkdown)
-   git, github
-   h2o mašininio mokymosi platforma <https://www.h2o.ai/>
-   projekto ataskaita rengiama markdown formatu (arba <https://rmarkdown.rstudio.com/>)

------------------------------------------------------------------------

# Vertinimo schema *(10 balų skalė)*

1.  GitHub projektas (1 t.)
2.  Duomenų nuskaitymas ir apjungimas (1 t.)
3.  Duomenų žvalgomoji analizė (1 t.)
4.  Modelio parinkimas (1 t.)
5.  Hyperparametrų optimizavimas (1 t.)
6.  WEB aplikacijos sukūrimas paskolos spėjimui (2 t.)
7.  Modelio tikslumas naujiems duomenims (2 t.)

Sum = 9

#### Papildomi taškai

1.  Shiny aplikacijos dizainas (1 t.)
2.  Shiny aplikacijos funkcionalumas (1 t.)
3.  Papildomų ML paketu taikymas (SparkML, SciKit-Learn, Keras, Tensorflow) (1 t.)
4.  Shiny aplikacijos pateikimas (Docker, Shiny-Apps, Shiny-Server) (1 t.)

sum = 4

**Top 3 komandos/dalyviai gaus maksimalų įvertinimą**

<img src="practice/img/vertinimas.png" width="500"/>

# Duomenys

Paskolų įsipareigojimo nevykdymas (loan default)

-   id - unikalus ID
-   y - ar įvykdytas paskolos įsipareigojimas (0 - TAIP, 1 - NE)
-   kiti kintamieji - paskolos parametrai (pvz. kredito istorija, paskolos tipas)

Duomenis galite atsisiųsti iš:

[Google Drive](https://drive.google.com/drive/folders/17NsP84MecXHyctM94NLwps_tsowld_y8?usp=sharing)